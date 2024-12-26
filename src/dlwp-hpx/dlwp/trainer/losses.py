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

#from typing import Mapping

#from graphcast import xarray_tree
import numpy as np
#from typing_extensions import Protocol
import xarray as xr

# These are from the PDE ARENA
# def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
#     loss = F.mse_loss(input, target, reduction="none")
#     # avg across space
#     reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
#     # sum across time + fields
#     reduced_loss = reduced_loss.sum(dim=(1, 2))
#     # reduce along batch
#     if reduction == "mean":
#         return torch.mean(reduced_loss)
#     elif reduction == "sum":
#         return torch.sum(reduced_loss)
#     elif reduction == "none":
#         return reduced_loss
#     else:
#         raise NotImplementedError(reduction)


class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

        #path = '/home/adboer/dlwp-hpx/src/dlwp-hpx/data/era5_1deg_1D_HPX64_1940-2024_grid_area.nc'
        path = '/home/adboer/dlwp-hpx/src/dlwp-hpx/data/era5_1deg_1D_HPX64_1979-2024_snorm_cell_area.nc'
        weights_map = xr.open_dataset(path)
      
        self.weights = torch.tensor(weights_map.cell_area.values , dtype=torch.float32)
        
        # self.variable_weights = {
        # 'msl': 0.1,
        # 'sst': 0.1,
        # 'stream250': 1.0,
        # 'stream500': 1.0,
        # 't2m': 1.0,
        # 'ttr': 1.0
        # }


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.weights = self.weights.to(target.device)

        # Ensure weights are on the same device as input
        squared_error = (input - target)**2

        weights_expanded = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        weights_expanded = weights_expanded.expand(input.shape[0], -1, input.shape[2], input.shape[3], -1, -1)

        # # Apply grid area weighting
        weighted_squared_error = squared_error * weights_expanded

        # # Apply per-variable weights
        # for idx, (var, weight) in enumerate(self.variable_weights.items()):
        #     weighted_squared_error[:, :, :, idx, :, :] *= weight

        # print('weighted error?', weighted_squared_error.shape)
        
        reduced_loss = torch.mean(weighted_squared_error)
     
    
        return reduced_loss 
       


#MSEloss = torch.nn.MSELoss(reduction=self.reduction) # MSEloss(input, target)
# LossAndDiagnostics = tuple[xarray.DataArray, xarray.Dataset]




