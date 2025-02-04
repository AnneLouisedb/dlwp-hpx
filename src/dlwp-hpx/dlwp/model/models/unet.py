import logging
from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch as th
import pandas as pd
import math
import torch
from dlwp.model.modules.healpix import HEALPixPadding, HEALPixLayer
from dlwp.model.modules.encoder import UNetEncoder, UNet3Encoder, ConditionalUNetEncoder
from dlwp.model.modules.decoder import UNetDecoder, UNet3Decoder, ConditionalUNetDecoder
from dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from dlwp.model.modules.losses import LossOnStep
from dlwp.model.modules.utils import Interpolate
from dlwp.model.modules.utils import fourier_embedding
from diffusers.schedulers import DDPMScheduler
logger = logging.getLogger(__name__)


class CubeSphereUNet(th.nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
    ):
        """
        The Deep Learning Weather Prediction (DLWP) UNet model on the cube sphere mesh.

        :param encoder: dictionary of instantiable parameters for the U-net encoder (see UnetEncoder docs)
        :param decoder: dictionary of instantiable parameters for the U-net decoder (see UnetDecoder docs)
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Build the model layers
        self.encoder = instantiate(encoder, input_channels=self._compute_input_channels())
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(decoder, input_channels=self.encoder.n_channels,
                                   output_channels=self._compute_output_channels())

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
        :param step: step number in the sequence of integration_steps
        :return: reshaped Tensor in expected shape for model encoder
        """
        if not (self.n_constants > 0 or self.decoder_input_channels > 0):
            return inputs[0].flatten(start_dim=1, end_dim=2)
        if self.n_constants == 0:
            result = [
                inputs[0].flatten(start_dim=1, end_dim=2),  # inputs
                inputs[1][:, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim)].flatten(1, 2)  # DI
            ]
            return th.cat(result, dim=1)
        if self.decoder_input_channels == 0:
            result = [
                inputs[0].flatten(start_dim=1, end_dim=2),  # inputs
                inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1]))  # constants
            ]
            return th.cat(result, dim=1)
        result = [
            inputs[0].flatten(start_dim=1, end_dim=2),  # inputs
            inputs[1][:, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim)].flatten(1, 2),  # DI
            inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
        ]
        return th.cat(result, dim=1)

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:
        shape = tuple(outputs.shape)
        return outputs.view(shape[0], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[2:])
    

    def forward(self, inputs: Sequence, output_only_last=False) -> th.Tensor:
        # Reshape required for compatibility of CubedSphere model with modified dataloader
        # [B, F, T, C, H, W] -> [B, T, C, F, H, W]
        inputs[0] = th.permute(inputs[0], dims=(0, 2, 3, 1, 4, 5))
        inputs[1] = th.permute(inputs[1], dims=(0, 2, 3, 1, 4, 5))
        inputs[2] = th.swapaxes(inputs[2], 0, 1)

        outputs = []
        for step in range(self.integration_steps):
            print("STEP?")
            if step == 0:
                input_tensor = self._reshape_inputs(inputs, step)
                print("TENSOR at 0", input_tensor.shape)
            else:
                input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:]), step)
                print("TENSOR", step, input_tensor.shape)

            hidden_states = self.encoder(input_tensor)
            outputs.append(self._reshape_outputs(self.decoder(hidden_states)))

        # On return, undo reshape from above
        if output_only_last:
            return outputs[-1].permute(dims=(0, 3, 1, 2, 4, 5))
        return th.cat(outputs, dim=1).permute(dims=(0, 3, 1, 2, 4, 5))


class HEALPixUNet(th.nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            presteps: int = 0,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
    ):
        """
        Deep Learning Weather Prediction (DLWP) UNet on the HEALPix mesh.

        :param encoder: dictionary of instantiable parameters for the U-net encoder
        :param decoder: dictionary of instantiable parameters for the U-net decoder
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param enable_nhwc: Model with [N, H, W, C] instead of [N, C, H, W] oder
        :param enable_healpixpad: Enable CUDA HEALPixPadding if installed
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad

        print('input time dim?', input_time_dim)
        print('outputtime dim', output_time_dim)
        
        #self.output_dim = self.output_channels*self.input_time_dim

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Build the model layers
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.hidden_channels = encoder.n_channels[0] #not sure about this
        time_embed_dim = self.hidden_channels * 4 # number of hidden channels times 4?
        self.encoder = instantiate(config=encoder,
                                   input_channels=self._compute_input_channels(),
                                   enable_nhwc=self.enable_nhwc,
                                   enable_healpixpad=self.enable_healpixpad,
                                   time_embed_dim = time_embed_dim)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(config=decoder,
                                   output_channels=self._compute_output_channels(),
                                   enable_nhwc = self.enable_nhwc,
                                   enable_healpixpad = self.enable_healpixpad)
        
        
        self.time_embed = th.nn.Sequential(
            th.nn.Linear(self.hidden_channels, time_embed_dim),
            self.encoder.activation,
            th.nn.Linear(time_embed_dim, time_embed_dim),)
        
    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)
    
    @integration_steps.setter
    def integration_steps(self, value):
        self._integration_steps = value

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
        :param step: step number in the sequence of integration_steps
        :return: reshaped Tensor in expected shape for model encoder
        """
        if not (self.n_constants > 0 or self.decoder_input_channels > 0):
            return inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1)
        if self.n_constants == 0:
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim+1)  # DI
            ]
            res = th.cat(result, dim=self.channel_dim)

            # fold faces into batch dim
            res = self.fold(res)
            
            return res
        if self.decoder_input_channels == 0:
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape)*[-1]))  # constants
                #th.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
            ]
            res = th.cat(result, dim=self.channel_dim)

            # fold faces into batch dim
            res = self.fold(res)
            
            return res
        
        result = [
            inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
            inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim+1),  # DI
            inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
            #th.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
        ]
        res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        #print('shape of fole', res.shape)
        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:

        # unfold:
        outputs = self.unfold(outputs)
        
        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        
        return res
    

    def multiple_forward(self, inputs: Sequence, time: th.Tensor = None, z: th.Tensor= None, output_only_last=False) -> th.Tensor:
        """Function is called in the forecast.py file. At each integration step the full diffusion process (k) is called. 
        Outputs for each integration step are stacked in the outputs list. """
        
        num_refinement_steps = 3
        time_multiplier = 1000 / num_refinement_steps
        betas = [4e-7 ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        scheduler = DDPMScheduler(
                num_train_timesteps=num_refinement_steps + 1,
                trained_betas=betas,
                prediction_type="v_prediction", # shouldnt this be epsilon?
                clip_sample=False,
            )
        
        outputs = []
        
        for step in range(self.integration_steps): # [B, F, C*T, H, W]
            print(f"Integration step {step}")
            
            if step == 0:
                # Add noise to the original input
                inputs_0 = inputs[0]
                # print(f"size of output UNET after {step} step", inputs_0.shape)

                y_noised = torch.randn_like(inputs_0, device=inputs[0].device)
                # inputs_first = inputs_0 + y_noised 
                inputs_first = torch.cat([inputs_0, y_noised], axis=3)
                inputs = [inputs_first] + inputs[1:]
                input_tensor = self._reshape_inputs(inputs, step) 
                
            else:
                
                inputs_0 = outputs[-1]
                # print(f"size of output UNET after {step} step", inputs_0.shape)

                y_noised = torch.randn_like(inputs_0, device=inputs[0].device)
                inputs_first = inputs_0 + y_noised 
                inputs_first = torch.cat([inputs_0, y_noised], axis=3)
                
                input_tensor = self._reshape_inputs([inputs_first] + list(inputs[1:]), step) # replace x_in with outputs[-1] (original)
                
                
            for k_scalar in scheduler.timesteps:
            
                #print(f"diffusion step {k_scalar}")

                batch_size = input_tensor[0].shape[0] if isinstance(input_tensor, list) else input_tensor.shape[0]
                time_tensor = torch.full((1,), k_scalar, device= input_tensor.device)
                # forward step without reshaping of inputs
                pred = self.special_forward(input_tensor, time=  time_tensor * time_multiplier) 
                #print("predicted value shape?",pred.shape)
                y_noised = scheduler.step(pred, k_scalar, y_noised).prev_sample
                #print("noised shape?", y_noised.shape)
            
                
            decodings = y_noised 
            B, F, C, T, H, W = decodings.shape
            # collapse the input [B, F, C, T, H, W] -> [B*F, C*T, H, W]
            decodings = decodings.reshape(B*F, C*T, H, W)
            reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)  
            outputs.append(reshaped)
            
  
        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)

        # output after all integration steps (stacking the solutions) # torch.Size([1, 12, 7, 4, 64, 64]) - 7 = nr. integration steps
        print("output of unet, shape?", res.shape)

        return res
    
    def special_forward(self, inputs: Sequence, time: th.Tensor = None, z: th.Tensor= None, output_only_last=False) -> th.Tensor:
        
        outputs = []

        input_tensor = inputs
    
        if time is not None:
            
            fourier = fourier_embedding(time, self.hidden_channels, device = input_tensor.device) # replaced 1 with 
            time_emb = self.time_embed(fourier) 
            encodings = self.encoder(input_tensor, time_emb)
            decodings = self.decoder(encodings, time_emb)
        else:
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
        

        #reshaped = self._reshape_outputs(decodings)  # Absolute prediction
        
        reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)  # Residual prediction
        outputs.append(reshaped)

  
        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)

        return res
    
  

    def forward(self, inputs: Sequence, time: th.Tensor = None, z: th.Tensor= None, output_only_last=False) -> th.Tensor:

        outputs = []
       
        for step in range(self.integration_steps): # [B, F, C*T, H, W]
            
           
            if step == 0:
                inputs_0 = inputs[0]
                input_tensor = self._reshape_inputs(inputs, step) # Squashes the time/channel dimension and concatenates in constants and decoder inputs. (would decoder inputs be 4?)
                # N, F, C, H, W -> N*F, C, H, W
               # print("reshaping input tensor", input_tensor.shape) # ([192, 4, 32, 32]) -> 12 x 16 = 192 #  (B*F, T*C, H, W)
               
            
            else:
                
                raise ValueError
                
                
            if time is not None:
                print("STEP integration", step)
                
                fourier = fourier_embedding(time, self.hidden_channels, device = input_tensor.device)
                time_emb = self.time_embed(fourier) 
                #print("what is the dimension of input?", input_tensor.shape)
                #print("shape of the time embedding?", time_emb.shape)
                encodings = self.encoder(input_tensor, time_emb)
                decodings = self.decoder(encodings, time_emb)
            else:
                encodings = self.encoder(input_tensor)
                decodings = self.decoder(encodings) #  B, F, T, C, H, W 
            # to this formag?
            #  B, F, T*C, H, W 
            #reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            
            reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)  # Residual prediction
            outputs.append(reshaped)

  
        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)

        print("Output of UNET?")
        print(res.shape)

        return res
    
   
class HEALPixRecUNet(th.nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            delta_time: str = "6H",
            reset_cycle: str = "24H",
            presteps: int = 1,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
    ):
        """
        Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh.

        :param encoder: dictionary of instantiable parameters for the U-net encoder
        :param decoder: dictionary of instantiable parameters for the U-net decoder
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param delta_time: hours between two consecutive data points
        :param reset_cycle: hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
            to never reset the hidden states.
        :param presteps: number of model steps to initialize recurrent states.
        :param enable_nhwc: Model with [N, H, W, C] instead of [N, C, H, W]
        :param enable_healpixpad: Enable CUDA HEALPixPadding if installed
        """
        super().__init__()
        self.channel_dim = 2  # Now 2 with [B, F, T*C, H, W]. Was 1 in old data format with [B, T*C, F, H, W]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds()//3600)
        self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds()//3600)
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Build the model layers
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.encoder = instantiate(config=encoder,
                                   input_channels=self._compute_input_channels(),
                                   enable_nhwc=self.enable_nhwc,
                                   enable_healpixpad=self.enable_healpixpad)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(config=decoder,
                                   output_channels=self._compute_output_channels(),
                                   enable_nhwc = self.enable_nhwc,
                                   enable_healpixpad = self.enable_healpixpad)
        
        
        hidden_channels = self.encoder.n_channels[-1]
        time_embed_dim = hidden_channels * 4 # number of hidden channels times 4?
        self.time_embed = th.nn.Sequential(
            th.nn.Linear(hidden_channels, time_embed_dim),
            self.encoder.activation,
            th.nn.Linear(time_embed_dim, time_embed_dim),)

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)# + self.presteps

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
        :param step: step number in the sequence of integration_steps
        :return: reshaped Tensor in expected shape for model encoder
        """
       
        if not (self.n_constants > 0 or self.decoder_input_channels > 0):
            return self.fold(prognostics)

        if self.n_constants == 0:
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim+1
                    )  # DI
            ]
            res = th.cat(result, dim=self.channel_dim)

            # fold faces into batch dim
            res = self.fold(res)
            
            return res

        if self.decoder_input_channels == 0:
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape)*[-1]))  # constants
            ]
            res = th.cat(result, dim=self.channel_dim)

            # fold faces into batch dim
            res = self.fold(res)
            
            return res

        
        result = [
            inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
            inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                start_dim=self.channel_dim, end_dim=self.channel_dim+1
                ),  # DI
            inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
        ]
        
        res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        
        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:

        # unfold:
        outputs = self.unfold(outputs)
        
        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        
        return res

    def _initialize_hidden(self, inputs: Sequence, outputs: Sequence, step: int) -> None:
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                input_tensor = self._reshape_inputs(
                    inputs=[inputs[0][:, :, s*self.input_time_dim:(s+1)*self.input_time_dim]] + list(inputs[1:]),
                    step=step+prestep
                    )
            else:
                s = step - self.presteps + prestep
                input_tensor = self._reshape_inputs(
                    inputs=[outputs[s-1]] + list(inputs[1:]),
                    step=s+1
                    )
            
            # Forward the data through the model to initialize hidden states
            self.decoder(self.encoder(input_tensor))

    def forward(self, inputs: Sequence, time: th.Tensor = None, z: th.Tensor= None, output_only_last=False) -> th.Tensor:
        """Forwarding the UNET in its entirity. This forward function should take the inputs, time and z (for a conditional model)"""
        assert not (time is None and z is None)

        self.reset()
        outputs = []
        
        for step in range(self.integration_steps):
            
            # (Re-)initialize recurrent hidden states
            if (step*(self.delta_t*self.input_time_dim)) % self.reset_cycle == 0:
                self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)
                # code does not reach here
                
            
            # Construct input: [prognostics|TISR|constants]
            if step == 0:
                s = self.presteps
                input_tensor = self._reshape_inputs(
                    inputs=[inputs[0][:, :, s*self.input_time_dim:(s+1)*self.input_time_dim]] + list(inputs[1:]),
                    step=s
                    )
            else:
                input_tensor = self._reshape_inputs(
                    inputs=[outputs[-1]] + list(inputs[1:]),
                    step=step+self.presteps
                    )
                
            if time is not None:
                hidden_channels = self.encoder.n_channels[-1]
                time_emb = self.time_embed(fourier_embedding(time, hidden_channels, device = input_tensor.device))
                th.cuda.nvtx.range_push(f"Forward encoder with diffusion")  
                print("Forwarding with a time embedding?")
                encodings = self.encoder(input_tensor, time_emb)
            else:

                print("Forwarding WITHOUT time embedding?")
                print(input_tensor.shape)
                encodings = self.encoder(input_tensor)


            decodings = self.decoder(encodings)

            
            # Absolute prediction
            #reshaped = self._reshape_outputs(decodings)
            # Residual prediction
            reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)
            outputs.append(reshaped)

        if output_only_last:
            return outputs[-1]
        print
        
        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()
