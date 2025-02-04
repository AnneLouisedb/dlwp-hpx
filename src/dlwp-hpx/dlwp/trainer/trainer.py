#!/usr/bin/env python3
import gc
import os
import threading
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# amp
from torch.cuda import amp

# distributed stuff
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  

# custom
from dlwp.utils import write_checkpoint


class Trainer():
    """
    A class for DLWP model training
    """

    def __init__(
            self,
            model: torch.nn.Module,  # Specify... (import)
            data_module: torch.nn.Module,  # Specify... (import)
            criterion: torch.nn.Module,  # Specify... (import)
            optimizer: torch.nn.Module,  # Specify... (import)
            lr_scheduler: torch.nn.Module,  # Specify... (import)
            num_refinement_steps: int = 1,
            min_epochs: int = 100,
            max_epochs: int = 500,
            early_stopping_patience: int = None,
            amp_mode: str = "none",
            graph_mode: str = "none",
            device: torch.device = torch.device("cpu"),
            output_dir: str = "/outputs/"
            ):
        """
        Constructor.

        :param model: 

        :param criterion: A PyTorch loss module
        :param optimizer: A PyTorch optimizer module
        :param lr_scheduler: A PyTorch learning rate scheduler module
        """
        self.device = device
        self.amp_enable = False if (amp_mode == "none") else True
        self.amp_dtype = torch.float16 if (amp_mode == "fp16") else torch.bfloat16
        self.output_variables = data_module.output_variables
        self.early_stopping_patience = early_stopping_patience

        self.model = model.to(device=self.device)

        if dist.is_initialized():
            self.dataloader_train, self.sampler_train = data_module.train_dataloader(num_shards=dist.get_world_size(),
                                                                                     shard_id=dist.get_rank())
            self.dataloader_valid, self.sampler_valid = data_module.val_dataloader(num_shards=dist.get_world_size(),
                                                                                   shard_id=dist.get_rank())
        else:
            self.dataloader_train, self.sampler_train = data_module.train_dataloader()
            self.dataloader_valid, self.sampler_valid = data_module.val_dataloader()
        self.output_dir_tb = os.path.join(output_dir, "tensorboard")

        # set the other parameters
        self.optimizer = optimizer
        self.criterion = criterion.to(device=self.device)
        self.lr_scheduler = lr_scheduler
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

        # add gradient scaler
        self.gscaler = amp.GradScaler(enabled=(self.amp_enable and self.amp_dtype == torch.float16))

        # use distributed data parallel if requested:
        self.print_to_screen = True
        self.train_graph = None
        self.eval_graph = None

        if dist.is_initialized():

            capture_stream = torch.cuda.Stream()
            with torch.cuda.stream(capture_stream):
                self.model = DDP(self.model,
                                 device_ids = [self.device.index],
                                 output_device = [self.device.index],
                                 broadcast_buffers = True,
                                 find_unused_parameters = False,
                                 gradient_as_bucket_view = True)
                capture_stream.synchronize()

            self.print_to_screen = dist.get_rank() == 0

            # capture graph if requested
            if graph_mode in ["train", "train_eval"]:
                if self.print_to_screen:
                    print(f"Capturing model for training ...")
                # get the shapes
                inp, tar = next(iter(self.dataloader_train))
                
                self._train_capture(capture_stream, [x.shape for x in inp], tar.shape)

                if graph_mode == "train_eval":
                    if self.print_to_screen:
                        print(f"Capturing model for validation ...")
                    self._eval_capture(capture_stream)

        # Set up tensorboard summary_writer or try 'weights and biases'
        # Initialize tensorbaord to track scalars
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.writer = SummaryWriter(log_dir=self.output_dir_tb)

    def compute_loss(self, prediction, target):
        d = ((target-prediction)**2).mean(dim=(0, 1, 2, 4, 5)) #*self.loss_weights
        return torch.mean(d)

    def fit(
            self,
            epoch: int = 0,
            validation_error: torch.Tensor = torch.inf,
            iteration: int = 0,
            epochs_since_improved: int = 0
            ):

        # Perform training by iterating over all epochs
        best_validation_error = validation_error
        for epoch in range(epoch, self.max_epochs):
            torch.cuda.nvtx.range_push(f"training epoch {epoch}")
            
            # Track epoch and learning rate in tensorboard
            #writer.add_scalar(tag="Epoch", scalar_value=epoch, global_step=iteration)
            #writer.add_scalar(tag="Learning Rate", scalar_value=optimizer.state_dict()["param_groups"][0]["lr"],
            #                  global_step=iteration)
            
            if self.sampler_train is not None:
                self.sampler_train.set_epoch(epoch)

            # Train: iterate over all training samples
            training_step = 0
            self.model.train()
            #for inputs, target in tqdm(self.dataloader_train, disable=(not self.print_to_screen)):
            for inputs, target in (pbar := tqdm(self.dataloader_train, disable=(not self.print_to_screen))):
                #for inputs, target in self.dataloader_train:
                pbar.set_description(f"Training   epoch {epoch+1}/{self.max_epochs}")

                if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                    self.writer.add_scalar(tag="epoch", scalar_value=epoch, global_step=iteration)

                torch.cuda.nvtx.range_push(f"training step {training_step}") 
                
                inputs = [x.to(device=self.device) for x in inputs]
                target = target.to(device=self.device)
                
                # do optimizer step
                if self.train_graph is not None:
                    # copy data into entry nodes
                    for idx, inp in enumerate(inputs):
                        self.static_inp[idx].copy_(inp)
                    self.static_tar.copy_(target)

                    # replay
                    self.train_graph.replay()

                    # extract loss
                    output = self.static_gen_train
                    train_loss = self.static_loss_train
                else:
                    # zero grads
                    self.model.zero_grad(set_to_none=True)

                    if self.amp_enable:
                        with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                            output = self.model(inputs)
                            train_loss = self.compute_loss(prediction=output, target=target)
                    else:
                        output = self.model(inputs)
                        train_loss = self.compute_loss(prediction=output, target=target)
                
                    self.gscaler.scale(train_loss).backward()

                # Gradient clipping                
                self.gscaler.unscale_(self.optimizer)
                curr_lr = self.optimizer.param_groups[-1]["lr"] if self.lr_scheduler is None else self.lr_scheduler.get_last_lr()[0]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), curr_lr)
                
                # Optimizer step
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
                                
                pbar.set_postfix({"Loss": train_loss.item()})

                torch.cuda.nvtx.range_pop()

                if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                    self.writer.add_scalar(tag="loss", scalar_value=train_loss, global_step=iteration)
                iteration += 1
                training_step += 1

            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"validation epoch {epoch}")  
            
            # Validate (without gradients)
            if self.sampler_valid is not None:
                self.sampler_valid.set_epoch(epoch)

            self.model.eval()
            with torch.no_grad():

                validation_stats = torch.zeros((2+len(self.output_variables)), dtype=torch.float32,
                                               device=self.device)
                for inputs, target in (pbar := tqdm(self.dataloader_valid, disable=(not self.print_to_screen))):
                    pbar.set_description(f"Validation epoch {epoch+1}/{self.max_epochs}")
                    inputs = [x.to(device=self.device) for x in inputs]
                    target = target.to(device=self.device)
                    bsize = float(target.shape[0])

                    # do eval step
                    if self.eval_graph is not None:
                        # copy data into entry nodes
                        for idx, inp in enumerate(inputs):
                            self.static_inp[idx].copy_(inp)
                        self.static_tar.copy_(target)

                        # replay graph
                        self.eval_graph.replay()

                        # increase the loss
                        validation_stats[0] += self.static_loss_eval * bsize

                        # Same for the per-variable loss
                        for v_idx, v_name in enumerate(self.output_variables):
                            validation_stats[1+v_idx] += self.static_losses_eval[v_idx] * bsize
                    else:
                        if self.amp_enable:
                            with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                                output = self.model(inputs)
                                validation_stats[0] += self.compute_loss(prediction=output, target=target) * bsize
                                for v_idx, v_name in enumerate(self.output_variables):
                                    validation_stats[1+v_idx] += self.criterion(
                                        output[v_idx], target[v_idx]
                                        ) * bsize
                        else:
                            output = self.model(inputs)
                            validation_stats[0] += self.compute_loss(prediction=output, target=target) * bsize
                            for v_idx, v_name in enumerate(self.output_variables):
                                validation_stats[1+v_idx] += self.criterion(
                                    output[v_idx], target[:, :, :, v_idx]
                                    ) * bsize

                    pbar.set_postfix({"Loss": (validation_stats[0]/validation_stats[-1]).item()})

                    # increment sample counter
                    validation_stats[-1] += bsize

                if dist.is_initialized():
                    dist.all_reduce(validation_stats)

                validation_error = (validation_stats[0] / validation_stats[-1]).item()

                # Record error per variable
                validation_errors = []
                for v_idx, v_name in enumerate(self.output_variables):
                    validation_errors.append((validation_stats[1+v_idx]/validation_stats[-1]).item())

                # Track validation improvement to later check early stopping criterion
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    epochs_since_improved = 0
                else:
                    epochs_since_improved += 1


            torch.cuda.nvtx.range_pop()

            # Logging and checkpoint saving
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(tag="learning_rate", scalar_value=self.optimizer.param_groups[0]['lr'],
                                           global_step=iteration)
                self.writer.add_scalar(tag="val_loss", scalar_value=validation_error, global_step=iteration)
                
                # Per-variable loss
                for v_idx, v_name in enumerate(self.output_variables):
                    self.writer.add_scalar(tag=f"val_loss/{v_name}", scalar_value=validation_errors[v_idx],
                                           global_step=iteration)

                # Write model checkpoint to file, using a separate thread
                thread = threading.Thread(
                    target=write_checkpoint,
                    args=(self.model.module if dist.is_initialized() else self.model,
                          self.optimizer,
                          self.lr_scheduler, epoch+1,
                          iteration,
                          validation_error,
                          epochs_since_improved,
                          self.output_dir_tb, )
                    )
                thread.start()

            # Update learning rate
            if self.lr_scheduler is not None: self.lr_scheduler.step()

            # Check early stopping criterium
            if self.early_stopping_patience is not None and epochs_since_improved >= self.early_stopping_patience:
                print(f"Hit early stopping criterium by not improving the validation error for {epochs_since_improved}"
                       " epochs. Finishing training.")
                break

        # Wrap up
        #if dist.get_rank() == 0:
        #    try:
        #        thread.join()
        #    except UnboundLocalError:
        #        pass
            self.writer.flush()
            self.writer.close()
