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
import wandb
import xarray as xr

class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        
        path = '/home/adboer/dlwp-hpx/src/dlwp-hpx/data/era5_1deg_1D_HPX64_1979-2024_grid_area.nc'
        weights_map = xr.open_dataset(path) 
        weights = torch.tensor(weights_map.cell_area.values , dtype=torch.float32)
        self.spatial_weights = weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
   

    def forward(self, input, target):
        # B, T, C, (F), H, W
        channel_weights = torch.tensor([0.1, 1.0, 0.1, 0.1, 0.1, 0.1], device=input.device)
        # weight targets by area - do this for each item in the batch (16) and across the channel and time dimension (1 and 6)
        self.spatial_weights = torch.tensor(self.spatial_weights, device = input.device)
        d = ((target-input)**2)*self.spatial_weights # torch.Size([16, 12, 1, 6, 64, 64])
        d = d.mean(dim=(0, 1, 2, 4, 5))*channel_weights 
        
       
        for i, channel_loss in enumerate(d):
            wandb.log({f"loss_channel_{i}": channel_loss.item()})
        
        return torch.mean(d)

   


         
# def compute_loss(self, prediction, target):
#         # B, T, C, (F), H, W
#     d = ((target- prediction)**2).mean(dim=(0, 1, 2, 4, 5)) #*self.loss_weights
#     # store the loss per channel in wandb before taking the mean
#         # Log the loss per channel in wandb
    
#     for i, channel_loss in enumerate(d):
#         wandb.log({f"loss_channel_{i}": channel_loss.item()})
    

#     return torch.mean(d)
