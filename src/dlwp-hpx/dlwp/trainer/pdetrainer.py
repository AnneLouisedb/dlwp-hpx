#!/usr/bin/env python3
import gc
import os
import sys
import threading
import healpy as hp
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from tqdm import tqdm
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
import math
from typing import List
# amp
from torch.cuda import amp
from torch.optim.lr_scheduler import _LRScheduler
from remap.healpix import HEALPixRemap

# distributed stuff
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  

# custom
from dlwp.utils import write_checkpoint

# diffusion 
from diffusers.schedulers import DDPMScheduler
import torch.nn.functional as F
from dlwp.utils import plot_single_step_frequency_spectrum


# These are from the PDE ARENA
def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)


class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        MSEloss = torch.nn.MSELoss(reduction=self.reduction)
        return MSEloss(input, target, reduction=self.reduction)


class Trainer():
    """
    A class for DLWP model training, This class trains a model to predict th next step in a diffusion process?
    """

    def __init__(
            self,
            model: torch.nn.Module,  # Specify... (import)
            data_module: torch.nn.Module,  # Specify... (import)
            criterion: torch.nn.Module,  # Specify... (import)
            optimizer: torch.nn.Module,  # Specify... (import)
            lr_scheduler: torch.nn.Module,  # Specify... (import) # use warmup steps!!
            num_refinement_steps: int = 3,
            min_epochs: int = 100,
            max_epochs: int = 500,
            early_stopping_patience: int = None,
            amp_mode: str = "none", # Training with mixed precision
            graph_mode: str = "none",
            device: torch.device = torch.device("cpu"),
            output_dir: str = "/outputs/",
            padding_mode: str = "zeros",
            predict_difference: bool = False,
            difference_weight: float = 1.0,
            min_noise_std: float = 4e-7,
            ema_decay: float = 0.995,
            writing: bool = True
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

        self.model = model.to(device=self.device) # Neural Operator
        self.train_criterion = CustomMSELoss()

        #self.ema = ExponentialMovingAverage(self.model, decay=self.hparams.ema_decay)
        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        # scheduling the addition of noise
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction", # shouldnt this be epsilon?
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / num_refinement_steps
        
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
            print("initialize distributed training?..")

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
        else:
            print("DIST is NOT initialized!")

        # Set up tensorboard summary_writer or try 'weights and biases'
        # Initialize tensorbaord to track scalars
        if writing:
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                self.writer = SummaryWriter(log_dir=self.output_dir_tb)
    

    # Make sure this code aligns!!         
    def compute_rolloutloss(self, batch, ):
        (u, v, cond, grid) = batch

        losses = {k: [] for k in self.rollout_criterions.keys()}
        for start in range(0, self.max_start_time + 1, self.hparams.time_future + 1):

            end_time = start + self.hparams.time_history
            target_start_time = end_time + 1 # time step
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

            # input sequence
            init_u = u[:, start:end_time, ...]
            # ground truth sequence
            targ_u = u[:, target_start_time:target_end_time, ...]

            init_v = None
            targ_traj = targ_u

            pred_traj = cond_rollout2d(
                self,
                init_u,
                init_v,
                None,
                cond,
                grid,
                self.pde,
                self.hparams.time_history,
                min(targ_u.shape[1], self.hparams.max_num_steps),
            )

            for k, criterion in self.rollout_criterions.items():
                loss = criterion(pred_traj, targ_traj)
                loss = loss.mean(dim=(0,) + tuple(range(2, loss.ndim)))
                losses[k].append(loss)
        loss_vecs = {k: sum(v) / max(1, len(v)) for k, v in losses.items()}
        return loss_vecs

    def predict_next_solution(self, inputs, save = None):
        """ This should be called once the model is trained! Call in the evaluation!"""
        if isinstance(inputs, list):
            # Get the first element of the list
            first_element = inputs[0]
            # Generate a random tensor with the same shape and type as the first element
            y_noised = torch.randn_like(first_element,device=self.device)
        
        elif isinstance(inputs, torch.Tensor):
            # Generate a random tensor with the same shape and type as the input tensor
            y_noised = torch.randn_like(inputs, device=self.device)

        storing = []
        for k_scalar in self.scheduler.timesteps:
            batch_size = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]
            time_tensor = torch.full((batch_size,), k_scalar, device=inputs[0].device if isinstance(inputs, list) else inputs.device)
            x_in = inputs[1] + y_noised
            # x_in = torch.cat([inputs[1], y_noised], axis=2) # so we only noise the second time step in this forecast

            x_in = [inputs[0], x_in] # conditioning input, actual input

            pred = self.model(x_in, time=  time_tensor * self.time_multiplier) 
            print("PRD OUTPUT?", pred.shape) # torch.Size([16, 12, 1, 1, 32, 32])
            
            y_noised = self.scheduler.step(pred, k_scalar, y_noised).prev_sample
            print("Y_NOISED SHAPE", y_noised.shape)  # torch.Size([16, 12, 2, 1, 32, 32]) WHY??

            if save:
                storing.append(y_noised)

        y = y_noised # apparently this has dimension: output shape torch.Size([16, 12, 2, 1, 32, 32])
        # i was expecting output shape torch.Size([16, 12, 1, 1, 32, 32])

        if save:

            remapper = HEALPixRemap(
            latitudes=181,
            longitudes=360,
            nside=32
            )

            for idx, image in enumerate(storing):
                first_item = image[0]  # Shape: [12, 1, 1, 32, 32]
                first_item_squeezed = first_item.squeeze()  # Shape: [12, 32, 32]
                remapper.hpx2ll(first_item_squeezed,  visualize = True, title = f"{idx}")
                print(f"Images saved to directory.")
                        

        return y 
           

    def _eval_capture(self, capture_stream, num_warmup_steps=20):
        self.model.eval()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):

            with torch.no_grad():
                for _ in range(num_warmup_steps):
                
                    # FW
                    with amp.autocast(enabled = self.amp_enable, dtype = self.amp_dtype):
                        # input from a single time step -> is the static input
                        self.static_gen_eval = self.predict_next_solution()

                        self.static_loss_eval = self.criterion(self.static_gen_eval, self.static_tar)
                        
                        self.static_losses_eval = []
                        for v_idx in range(len(self.output_variables)):
                            self.static_losses_eval.append(self.criterion(self.static_gen_eval[:, :, :, v_idx],
                                                                          self.static_tar[:, :, :, v_idx]))
                            
            # sync here
            capture_stream.synchronize()

            gc.collect()
            torch.cuda.empty_cache()

            # create graph
            self.eval_graph = torch.cuda.CUDAGraph()

            # start capture:
            with torch.cuda.graph(self.eval_graph, pool=self.train_graph.pool()):

                # FW
                with torch.no_grad():
                    with amp.autocast(enabled = self.amp_enable, dtype = self.amp_dtype):
                        
                        self.static_gen_eval = self.predict_next_solution()

                        self.static_loss_eval = self.criterion(self.static_gen_eval, self.static_tar)
                        
                        self.static_losses_eval = []
                        for v_idx in range(len(self.output_variables)):
                            # store the loss per output variable.. 
                            self.static_losses_eval.append(self.criterion(self.static_gen_eval[:, :, :, v_idx],
                                                                          self.static_tar[:, :, :, v_idx]))
                            
        # wait for capture to finish
        torch.cuda.current_stream().wait_stream(capture_stream) 

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
            torch.cuda.nvtx.range_push(f"training (refiner) epoch{epoch}")
            
            if self.sampler_train is not None:
                self.sampler_train.set_epoch(epoch)

            # Train: iterate over all training samples
            training_step = 0
            self.model.train()
            
            for inputs, target in (pbar := tqdm(self.dataloader_train, disable=(not self.print_to_screen))):
                print("what is the input shape then?") # two tensors are the input
               
                for inp in inputs:
                    print("C:")
                    print(inp.shape)
                # output = self.model(inputs) - old version!!!

                inp_shapes = [x.shape for x in inputs]
                self.static_inp = [torch.zeros(x_shape, dtype=torch.float32, device=self.device) for x_shape in inp_shapes]
                self.static_tar = torch.zeros(target.shape, dtype=torch.float32, device=self.device)


                #for inputs, target in self.dataloader_train:
                pbar.set_description(f"Training  epoch {epoch+1}/{self.max_epochs}")

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

                    # extract loss - these are defined in the train graph
                    output = self.static_gen_train
                    train_loss = self.static_loss_train
                else:
                    
                    # zero grads
                    self.model.zero_grad(set_to_none=True)

                    if self.amp_enable:
                        with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):

                            print("Manual training!!?")  

                            k = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=self.device)
                            k_scalar = k.item()
                            batch_size = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]
                            time_tensor = torch.full((batch_size,), k_scalar, device=inputs[0].device if isinstance(inputs, list) else inputs.device)
                            
                            noise_factor = self.scheduler.alphas_cumprod.to(self.device)[k]
                            noise_factor = noise_factor.view(-1, *[1 for _ in range(self.static_inp[0].ndim - 1)]) 
                            signal_factor = 1 - noise_factor

                            print("dimension STATIC TAR", self.static_tar.shape) # dimension STATIC TAR torch.Size([16, 12, 4, 1, 32, 32])
                            noise = torch.randn_like(self.static_tar)

                            y_noised = self.scheduler.add_noise(target, noise, k)
                            print("y_noised", y_noised.shape) #  torch.Size([8, 12, 1, 1, 32, 32])

                            x_in = inputs[1] + y_noised
                            x_in = [inputs[0], x_in]
                
                            output = self.model(x_in, time= time_tensor * self.time_multiplier) # used to be x_in
                            print("OUTPUT SHAPES,", output.shape) # OUTPUT SHAPES, torch.Size([8, 12, 1, 1, 32, 32])

                            train_loss = self.compute_loss(prediction=output, target=target)# self.criterion(output, target

                            # target = (noise_factor**0.5) * noise - (signal_factor**0.5) * target
                            # train_loss = self.train_criterion(pred, target)

                            

                    else:
                        output = self.model(inputs, 2)
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
                    # why does the dataloader return a list of two time steps, check this out?
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
                                
                                output = self.predict_next_solution(inputs)
                                # this is the output of the autoregressive time step
                                plot_single_step_frequency_spectrum(output, target, spatial_domain_size = 1000)

                                
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
        # if dist.get_rank() == 0:
        #     try:
        #         thread.join()
        #     except UnboundLocalError:
        #         pass
        self.writer.flush()
        self.writer.close()

