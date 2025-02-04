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
import wandb
 
from dlwp.trainer.losses import *
        
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
            diffusion: bool = True
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

    
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            
            wandb.init(project="your_project_name")

    
    def predict_next_solution(self, inputs):
        """ This should be called once the model is trained! """
        if isinstance(inputs, list):
            # Get the first element of the list
            first_element = inputs[0]
            # Generate a random tensor with the same shape and type as the first element
            y_noised = torch.randn_like(first_element,device=self.device)
        
        elif isinstance(inputs, torch.Tensor):
            # Generate a random tensor with the same shape and type as the input tensor
            y_noised = torch.randn_like(inputs, device=self.device)

        for k_scalar in self.scheduler.timesteps:
            batch_size = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]
            time_tensor = torch.full((batch_size,), k_scalar, device=inputs[0].device if isinstance(inputs, list) else inputs.device)
          
            # this has to be a torch cat over the third dimension TO DO             
            x_in = torch.cat([inputs[0], y_noised], axis=3)  
            # append the x_in as inputs[0]

            x_in = [x_in] + inputs[1:]
              
            pred = self.model(x_in, time=  time_tensor * self.time_multiplier) 
            y_noised = self.scheduler.step(pred, k_scalar, y_noised).prev_sample
        
           
        y = y_noised 
                
        return y 
  
          
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
                
                
                inp_shapes = [x.shape for x in inputs]
                self.static_inp = [torch.zeros(x_shape, dtype=torch.float32, device=self.device) for x_shape in inp_shapes]
                self.static_tar = torch.zeros(target.shape, dtype=torch.float32, device=self.device)


                #for inputs, target in self.dataloader_train:
                pbar.set_description(f"Training  epoch {epoch+1}/{self.max_epochs}")

               
                wandb.log({"epoch": epoch}, step=iteration)
                    
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

                    with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):

                        # make a k value
                        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=self.device)
                        k_scalar = k.item()
                        batch_size = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]
                        time_tensor = torch.full((batch_size,), k_scalar, device=inputs[0].device if isinstance(inputs, list) else inputs.device)
                        # constructing the noise factor
                        noise_factor = self.scheduler.alphas_cumprod.to(self.device)[k]
                        noise_factor = noise_factor.view(-1, *[1 for _ in range(self.static_inp[0].ndim - 1)]) 
                        signal_factor = 1 - noise_factor

                        noise = torch.randn_like(self.static_tar)
                        y_noised = self.scheduler.add_noise(target, noise, k)
                        
                        
                        x_in = torch.cat([inputs[0], y_noised], axis=3)
                        # adding the insolation to the input
                        x_in = [x_in] + inputs[1:]
                        
                        
                        output = self.model(x_in, time= time_tensor * self.time_multiplier)
                        
                        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * target

                       
                        train_loss = self.train_criterion(input = output, target = target) 
                        wandb.log({f"train_loss_k{k_scalar}": train_loss.item() 
                        })
                         
                       
                    
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
                    #self.writer.add_scalar(tag="loss", scalar_value=train_loss, global_step=iteration)
                    wandb.log({"train loss": train_loss}, step=iteration)

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
                        
                        with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                            
                            output = self.predict_next_solution(inputs) 

                            output_example = output[0] # [B, F, T, C, H, W] -> [F, T, C, H, W]
                
                            # this has to be compute loss!!
                            validation_stats[0] += self.train_criterion(input = output, target = target)  * bsize  
                            
                            for v_idx, v_name in enumerate(self.output_variables):
                                validation_stats[1+v_idx] += self.criterion(
                                    output[v_idx], target[v_idx]
                                    ) * bsize
                       
                    
                    # increment sample counter
                    validation_stats[-1] += bsize

                wandb.log({f"validation loss": (validation_stats[0]/validation_stats[-1]).item() },  step=iteration)
                pbar.set_postfix({"Loss": (validation_stats[0]/validation_stats[-1]).item()})
                
                if dist.is_initialized():
                    dist.all_reduce(validation_stats)

                validation_error = (validation_stats[0] / validation_stats[-1]).item()
                wandb.log({ "val_loss": validation_error}, step=iteration)

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
                    wandb.log({ "learning_rate": self.optimizer.param_groups[0]['lr']}, step=iteration)
                    
                
                wandb.log({ "val_loss": validation_error}, step=iteration)

                #Per-variable loss
                for v_idx, v_name in enumerate(self.output_variables):
                    
                    wandb.log({f"val_loss/{v_name}": validation_errors[v_idx]}, step=iteration)


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





# Make sure this code aligns!! PDE ARENA     
    # def compute_rolloutloss(self, batch, ):
    #     (u, v, cond, grid) = batch

    #     losses = {k: [] for k in self.rollout_criterions.keys()}
    #     for start in range(0, self.max_start_time + 1, self.hparams.time_future + 1):

    #         end_time = start + self.hparams.time_history
    #         target_start_time = end_time + 1 # time step
    #         target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

    #         # input sequence
    #         init_u = u[:, start:end_time, ...]
    #         # ground truth sequence
    #         targ_u = u[:, target_start_time:target_end_time, ...]

    #         init_v = None
    #         targ_traj = targ_u

    #         pred_traj = cond_rollout2d(
    #             self,
    #             init_u,
    #             init_v,
    #             None,
    #             cond,
    #             grid,
    #             self.pde,
    #             self.hparams.time_history,
    #             min(targ_u.shape[1], self.hparams.max_num_steps),
    #         )

    #         for k, criterion in self.rollout_criterions.items():
    #             loss = criterion(pred_traj, targ_traj)
    #             loss = loss.mean(dim=(0,) + tuple(range(2, loss.ndim)))
    #             losses[k].append(loss)
    #     loss_vecs = {k: sum(v) / max(1, len(v)) for k, v in losses.items()}
    #     return loss_vecs
       
        

