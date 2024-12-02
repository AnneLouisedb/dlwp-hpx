from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th
import einops
from dlwp.model.modules.utils import zero_module
from dlwp.model.modules.healpix import HEALPixLayer
from dlwp.model.modules.utils import Interpolate, ConditionedBlock

class Swish(th.nn.Module):
    """
    ### Swish activation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * th.sigmoid(x)

class ConvNeXtBlock(ConditionedBlock):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

       
        if in_channels == out_channels:
            self.skip_module = lambda x: x
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
            )

        convblock = []
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
        ))
        if activation is not None: convblock.append(activation)
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
        ))
        if activation is not None: convblock.append(activation)
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
        ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x, time_emb):
        
        h = self.convblock(x)
        h = h + time_emb 
        return self.skip_module(x) + h
    

class MiddleBlock(ConditionedBlock):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
        use_scale_shift_norm (bool, optional): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1. Defaults to False.
    """

    def __init__(
        self,
        n_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )
        self.attn =  th.nn.Identity() # AttentionBlock(n_channels) if has_attn else NO ATTENTION FOR NOW
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x



        



# Conditional Residual block for the Modern Unet used in the PDE-REfiner
class ConditionalResidualBlock(ConditionedBlock):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1.
        # TO DO FINISH DOCSTRING

    Note:
        This conditional residual block is used in the Modern U-Net, PDE-REfiner paper. (Figure 10: ResNet block of the Modern U-Net)
    """
    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        cond_channels_main: int = 0,
        time_embed_dim: int = 1024, 
        activation: th.nn.Module = th.nn.GELU(),
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        use_scale_shift_norm: bool = False, # n_dims = 1
        norm: bool = False, 
        n_groups: int = 32,
        n_layers = None,
        ):
        super().__init__()

        self.use_scale_shift_norm = use_scale_shift_norm
        self.activation = activation
       
        # Main convolution layers
        self.conv1 = geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels + cond_channels_main,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            
        )
        # this one has to go through the zero_module
        self.conv2 = zero_module(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
        ))
        

         # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
            )
        else:
            self.shortcut = th.nn.Identity()


        if norm: # check if 32 is the default group norm?
            self.norm1 = th.nn.GroupNorm(n_groups, in_channels + cond_channels_main)
            self.norm2 = th.nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = th.nn.Identity()
            self.norm2 = th.nn.Identity()

        self.cond_emb = th.nn.Linear(time_embed_dim, 2 * out_channels if use_scale_shift_norm else out_channels)
      

    def forward(self, x: th.Tensor, emb: th.Tensor, cond: th.Tensor = None):
        """
        1. GroupNorm
        2. GELU
        3. Convolution
        4. Groupnorm
        5. Scale-and-Shift (conditioning features)
        6. GELU
        7. Convolution
        8. +Adding residual input (shortcut)
        """
        # Step 1 - 3
        if cond:
            h = self.conv1(self.activation(self.norm1(th.cat([x, cond], dim=1))))
        else:
            h = self.conv1(self.activation(self.norm1(x)))
            
        emb_out = self.cond_emb(emb)
        emb_out =  emb_out.unsqueeze(-1).unsqueeze(-1)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            # Step 4 - 5
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
            # Step 6 -7
            h = self.conv2(self.activation(h))
        else:
            # Step 4 - 7
            h = h + emb_out
            h = self.conv2(self.activation(self.norm2(h)))
       
        # Step 8
        return h + self.shortcut(x)
    
class BasicConvBlock(ConditionedBlock):
    """
    Convolution block consisting of n subsequent convolutions and activations
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            n_layers: int = 1,
            latent_channels: int = None,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        if latent_channels is None: latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            # Conv2D
            convblock.append(geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels if n == 0 else latent_channels,
                out_channels=out_channels if n == n_layers - 1 else latent_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
            # Attention
            if activation is not None: convblock.append(activation)
        self.out_channels = out_channels
        self.convblock = th.nn.Sequential(*convblock)
         # in channels should be the number of conditional channels
        time_channels = 4 # this should be the initial feature map * 4
        self.cond_emb = th.nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

    def forward(self, x, time_emb):
        emb_out = self.cond_emb(time_emb)
        first_layer = self.convblock[0]
        x = first_layer(x)

        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x = x+ emb_out  

        # Process through the remaining layers
        for layer in self.convblock[1:]:
            x = layer(x)

        return x
       
       

class ConvGRUBlock(ConditionedBlock):
    """
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            kernel_size: int = 1,
            downscale_factor: int = 4,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels + self.channels,
            out_channels=2*self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.conv_can = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels+self.channels,
            out_channels=self.channels, # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence, time_emb: th.Tensor) -> Sequence:
        # time embeddings are calculated in the encoder!

        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)

        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate*self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        # do the time embeddings go through an activation as well?
        # batch normalize the entire thing?

        return inputs + h_next + time_emb

    def reset(self):
        self.h = th.zeros_like(self.h)