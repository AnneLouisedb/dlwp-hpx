import torch
import pytest
#import hydra
#from hydra.utils import instantiate

from dlwp.trainer.pdetrainer import DummyTrainer
#from dlwp.data.data_loading import DataModule
from dlwp.data.data_loading import TimeSeriesDataset
from dlwp.data.modules import TimeSeriesDataModule
from omegaconf import OmegaConf
from dlwp.model.models.unet import HEALPixRecUNet
#@hydra.main(config_path='../configs', config_name='config', version_base=None)


def test_trainer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    time_history = 1
    train_criterion = torch.nn.MSELoss()
    
    encoder = {
    "conv_block": {
        "activation": {
            "_target_": "dlwp.model.modules.activations.CappedGELU",
            "cap_value": 10
        },
        "_target_": "dlwp.model.modules.conditional_blocks.ConvNeXtBlock",
        "_recursive_": True,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "dilation": 1,
        "upscale_factor": 4
    },
    "down_sampling_block": {
        "_target_": "dlwp.model.modules.blocks.AvgPool",
        "pooling": 2
    },
    "recurrent_block": {
        "_target_": "dlwp.model.modules.conditional_blocks.ConvGRUBlock",
        "_recursive_": False,
        "in_channels": 3,
        "kernel_size": 1,
        "downscale_factor": 4
    },
    "_target_": "dlwp.model.modules.encoder.ConditionalUNetEncoder", 
    "_recursive_": False,
    "n_channels": [136, 68, 34],
    "dilations": [1, 2, 4]
    }

    decoder = {
    "conv_block": {
        "activation": {
            "_target_": "dlwp.model.modules.activations.CappedGELU",
            "cap_value": 10
        },
        "_target_": "dlwp.model.modules.conditional_blocks.ConvNeXtBlock",
        "_recursive_": True,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "dilation": 1,
        "upscale_factor": 4
    },
    "up_sampling_block": {
        "activation": {
            "_target_": "dlwp.model.modules.activations.CappedGELU",
            "cap_value": 10
        },
        "_target_": "dlwp.model.modules.blocks.TransposedConvUpsample",
        "in_channels": 3,
        "out_channels": 1,
        "upsampling": 2
    },
    "recurrent_block": {
        "_target_": "dlwp.model.modules.blocks.ConvGRUBlock",
        "_recursive_": False,
        "in_channels": 3,
        "kernel_size": 1,
        "downscale_factor": 4
    },
    "output_layer": {
        "_target_": "dlwp.model.modules.blocks.BasicConvBlock",
        "in_channels": 3,
        "out_channels": 2,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1
    },
    "_target_": "dlwp.model.modules.decoder.UNetDecoder",
    "_recursive_": False,
    "n_channels": [34, 68, 136],
    "dilations": [4, 2, 1],
    #"presteps": 1,
    #"input_time_dim": 2,
   # "output_time_dim": 4,
   # "delta_time": "24h",
   # "input_channels": 7,
   # "output_channels": 7,
   # "n_constants": 2,
   # "decoder_input_channels": 1,
   # "enable_nhwc": False,
    # "enable_healpixpad": False
    }
    model = HEALPixRecUNet(encoder = encoder , decoder = decoder, input_channels= 7, output_channels= 7, n_constants= 2, decoder_input_channels= 1, input_time_dim = 2, output_time_dim=  4)

    scaling_dict = {
        "t2m0": {
            "mean": 287.8665771484375,
            "std": 14.86227798461914
        },
        "t2m": {
            "mean": 287.8665771484375,
            "std": 14.86227798461914
        },
        "t850": {
            "mean": 281.2710266113281,
            "std": 12.04991626739502
        },
        "tau300-700": {
            "mean": 61902.72265625,
            "std": 2559.8408203125
        },
        "tcwv0": {
            "mean": 24.034976959228516,
            "std": 16.411935806274414
        },
        "z1000": {
            "mean": 952.1435546875,
            "std": 895.7516479492188
        },
        "z250": {
            "mean": 101186.28125,
            "std": 5551.77978515625
        },
        "z500": {
            "mean": 55625.9609375,
            "std": 2681.712890625
        },
        "tp6": {
            "mean": 0.0,
            "std": 1.0,
            "log_epsilon": 1e-6
        }
    }
    date_ranges = {
    "train_date_start": "2012-01-01",
    "train_date_end": "2012-12-31T18:00",
    "val_date_start": "2016-01-01",
    "val_date_end": "2016-12-31T18:00",
    "test_date_start": "2018-01-01",
    "test_date_end": "2018-12-31T18:00"
    }
    # Convert scaling_dict to an OmegaConf object
    scaling = OmegaConf.create(scaling_dict)
    dates = OmegaConf.create(date_ranges)

    data_module = TimeSeriesDataModule(src_directory = "/home/adboer/dlwp-hpx/src/dlwp-hpx/data", 
                                       dst_directory = "/home/adboer/dlwp-hpx/src/dlwp-hpx/data",
                                        dataset_name = "era5_hpx64_1var_24h_24h",
                                        prefix = 'era5_1deg_1D_HPX32_1940-2024_', 
                                        scaling = scaling,
                                        suffix =  '',
                                        data_format = 'classic',
                                        input_variables = ['t2m'],
                                        splits = dates)
    data_module.setup()                                   
    optimizer=torch.optim.Adam(model.parameters())
    num_epochs = 2

    trainer = DummyTrainer(
        model=model,
        data_module=data_module,
        criterion=train_criterion,
        optimizer= optimizer,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs),
        num_refinement_steps=3,
        output_variables = None,
        device=device,
        writing = False)
   
    
    assert trainer is not None
    assert isinstance(trainer.model, torch.nn.Module)
    
    # Test training step
    #th.randn(1, 12, 1, 32, 32)  16, 12, 2, 1, 32, 32
    batch = (
        [torch.randn(16, 12, 4, 1, 64, 64, device=device), torch.randn(16, 12, 8, 1, 64, 64, device=device)],
        torch.randn(16, 12, 2, 1, 64, 64, device=device) 
    )
    # 36, 2, 1, 64, 64
    inp_shapes = [x.shape for x in batch[0]]
    trainer.model.train()
    print("start training?")#  this is the input for the forward [16, 12, 4, 1, 32, 32]), torch.Size([16, 12, 8, 1, 32, 32]
    testing = trainer.fit()
    print("TESTING")
    print(testing)
    #loss = trainer._train_capture(inp_shapes , batch[1].shape)

    #[torch.Size([16, 12, 4, 1, 32, 32]), torch.Size([16, 12, 8, 1, 32, 32])]
    #assert loss is not None
    
    # Test evaluation step
    trainer.model.eval()
    with torch.no_grad():
        loss = trainer._eval_capture()
    assert loss is not None
    
    # Test fit method
    #trainer.fit(epoch=0, validation_error=torch.inf, iteration=0, epochs_since_improved=0)
    #assert trainer.model.training == False  # Model should be in eval mode after fit
    
    # Test predict_next_solution
    prediction = trainer.predict_next_solution()
    assert prediction.shape == (8, 1, 64, 64)

test_trainer()