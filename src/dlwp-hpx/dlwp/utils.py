import logging
import re
import os
import glob
import xarray as xr
import numpy as np
import torch as th
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from remap.healpix import HEALPixRemap

def plot_single_step_frequency_spectrum(output, target, spatial_domain_size, sample_index=0, filename=None):
    """
    Plot the Fourier spectrum comparison between output and target for a single step.
    
    Args:
    output (torch.Tensor): Predicted output tensor of shape (batch_size, spatial_points)
    target (torch.Tensor): Target tensor of shape (batch_size, spatial_points)
    spatial_domain_size (float): Size of the spatial domain
    sample_index (int): Index of the sample to plot (default: 0)
    filename (str): Custom filename for saving the plot (default: None, generates a timestamped filename)
    
    Returns:
    matplotlib.figure.Figure: The generated figure object
    """
    # Check input shapes
    print(output.shape, "OUTPUT SHAPE!")
    print(target.shape, "Target shape!")

    assert output.shape == target.shape, "Output and target shapes must match"
    # output is from the predict next solution step.

    remapper = HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=32)

    first_item = output[0]  # Shape: [12, 1, 1, 32, 32]
    first_item_squeezed = first_item.squeeze()  # Shape: [12, 32, 32]
    output = remapper.hpx2ll(first_item_squeezed,  visualize = True, title = f"output_diffusion_step")

    first_item = target[0]  # Shape: [12, 1, 1, 32, 32]
    first_item_squeezed = first_item.squeeze()  # Shape: [12, 32, 32]
    target = remapper.hpx2ll(first_item_squeezed,  visualize = True, title = f"output_target_step")
    
    print("REMAPPER OUTPUT")
    print(output.shape, "OUTPUT SHAPE!")
    print(target.shape, "Target shape!")
    #   :return: An array of shape [height=latitude, width=longitude] containing the latlon data
    
    # take the 23rd item in the latitude (first dimension)
    # such that we have the array over the longitude
    # Move tensors to CPU and convert to numpy arrays

    # Assuming output and target are 3D tensors after squeezing
    output_lat = output[23, :]  # Extracting 23rd latitudex
    target_lat = target[23, :]  # Extracting 23rd latitude

    output_np = output_lat 
    target_np = target_lat 

    # Compute FFT
    output_fft = np.fft.fft(output_np)
    target_fft = np.fft.fft(target_np)

    # Compute amplitude spectrum
    amplitude_spectrum_output =  np.abs(output_fft)
    amplitude_spectrum_target = np.abs(target_fft)

    # set the dimensions to the x and y axis length of the original input
    # these have to be the latitude and longitude values of the xarray
    fft_da = xr.DataArray(amplitude_spectrum_output, dims = ['latitude', 'longitude'] )
    fft_target = xr.DataArray(amplitude_spectrum_target, dims = ['latitude', 'longitude'])


    fig, ax = plt.subplots(figsize=(20, 10))
    # Prepare data for boxplot
    data = []
    for lat in fft_da.latitude.values:
        for wn in range(1, 15):  # Adjust range as needed
            amplitudes = fft_da.sel(latitude=lat).isel(longitude=wn).values
            data.extend([(lat, wn, amp) for amp in amplitudes])

    # Create DataFrame
    df_output = pd.DataFrame(data, columns=['Latitude', 'Wavenumber', 'Amplitude'])

    data_target = []
    for lat in fft_target.latitude.values:
        for wn in range(1, 15):
            amplitudes = fft_target.sel(latitude=lat).isel(longitude=wn).values
            data_target.extend([(lat, wn, amp) for amp in amplitudes])

    df_target = pd.DataFrame(data_target, columns=['Latitude', 'Wavenumber', 'Amplitude'])

    # Create combined boxplot
    sns.boxplot(x='Wavenumber', y='Amplitude', hue='Latitude', data=pd.concat([df_output, df_target]), ax=ax)

    # Customize the plot
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Zonal FFT Amplitude Spectrum Distribution')
    ax.legend(title='Latitude')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 80000)


    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'fourier_spectrum_output_comparison_{timestamp}.png'

    # Save the plot
    plt.savefig(filename)
    plt.close(fig)

    return 



def configure_logging(verbose=1):
    verbose_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.NOTSET
    }
    if verbose not in verbose_levels.keys():
        verbose = 1
    current_logger = logging.getLogger()
    current_logger.setLevel(verbose_levels[verbose])
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s][PID=%(process)d]"
        "[%(levelname)s %(filename)s:%(lineno)d] - %(message)s"))
    handler.setLevel(verbose_levels[verbose])
    current_logger.addHandler(handler)


def remove_chars(in_str):
    """
    Remove characters from a string that have unintended effects on file paths.
    :param in_str: str
    :return: str
    """
    return ''.join(re.split('[$/\\\\]', in_str))


def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.
    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to
        xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    for var in ds.data_vars:
        if 'coordinates' in ds[var].encoding:
            del ds[var].encoding['coordinates']
        ds[var].encoding['contiguous'] = False
        ds[var].encoding['original_shape'] = ds[var].shape
        ds[var].encoding['chunksizes'] = tuple([chunk_dict[d] for d in ds[var].dims])
        ds[var].encoding['chunks'] = tuple([chunk_dict[d] for d in ds[var].dims])
    return ds


def encode_variables_as_int(ds, dtype='int16', compress=0, exclude_vars=()):
    """
    Adds encoding to Dataset variables to export to int16 type instead of float at write time.
    :param ds: xarray Dataset
    :param dtype: as understood by numpy
    :param compress: int: if >1, also enable compression of the variable with this compression level
    :param exclude_vars: iterable of variable names which are not to be encoded
    :return: xarray Dataset with encoding applied
    """
    for var in ds.data_vars:
        if var in exclude_vars:
            continue
        var_max = float(ds[var].max())
        var_min = float(ds[var].min())
        var_offset = (var_max + var_min) / 2
        var_scale = (var_max - var_min) / (2 * np.iinfo(dtype).max)
        if var_scale == 0:
            logger.warning("min and max for variable %s are both %f", var, var_max)
            var_scale = 1.
        ds[var].encoding.update({
            'scale_factor': var_scale,
            'add_offset': var_offset,
            'dtype': dtype,
            '_Unsigned': not np.issubdtype(dtype, np.signedinteger),
            '_FillValue': np.iinfo(dtype).min,
        })
        if 'valid_range' in ds[var].attrs:
            del ds[var].attrs['valid_range']
        if compress > 0:
            ds[var].encoding.update({
                'zlib': True,
                'complevel': compress
            })
    return ds


def insolation(dates, lat, lon, S=1., daily=False, enforce_2d=False, clip_zero=True):  # pylint: disable=invalid-name
    """
    Calculate the approximate solar insolation for given dates.

    For an example reference, see:
    https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html

    :param dates: 1d array: datetime or Timestamp
    :param lat: 1d or 2d array of latitudes
    :param lon: 1d or 2d array of longitudes (0-360deg). If 2d, must match the shape of lat.
    :param S: float: scaling factor (solar constant)
    :param daily: bool: if True, return the daily max solar radiation (lat and day of year dependent only)
    :param enforce_2d: bool: if True and lat/lon are 1-d arrays, turns them into 2d meshes.
    :param clip_zero: bool: if True, set values below 0 to 0
    :return: 3d array: insolation (date, lat, lon)
    """
    # pylint: disable=invalid-name
    if len(lat.shape) != len(lon.shape):
        raise ValueError("'lat' and 'lon' must have the same number of dimensions")
    if len(lat.shape) >= 2 and lat.shape != lon.shape:
        raise ValueError(f"shape mismatch between lat ({lat.shape} and lon ({lon.shape})")
    if len(lat.shape) == 1 and enforce_2d:
        lon, lat = np.meshgrid(lon, lat)
    n_dim = len(lat.shape)

    # Constants for year 1995 (standard in climate modeling community)
    # Obliquity of Earth
    eps = 23.4441 * np.pi / 180.
    # Eccentricity of Earth's orbit
    ecc = 0.016715
    # Longitude of the orbit's perihelion (when Earth is closest to the sun)
    om = 282.7 * np.pi / 180.
    beta = np.sqrt(1 - ecc ** 2.)

    # Get the day of year as a float.
    start_years = np.array([pd.Timestamp(pd.Timestamp(d).year, 1, 1) for d in dates], dtype='datetime64')
    days_arr = (np.array(dates, dtype='datetime64') - start_years) / np.timedelta64(1, 'D')
    for d in range(n_dim):
        days_arr = np.expand_dims(days_arr, -1)
    # For daily max values, set the day to 0.5 and the longitude everywhere to 0 (this is approx noon)
    if daily:
        days_arr = 0.5 + np.round(days_arr)
        new_lon = lon.copy().astype(np.float32)
        new_lon[:] = 0.
    else:
        new_lon = lon.astype(np.float32)
    # Longitude of the earth relative to the orbit, 1st order approximation
    lambda_m0 = ecc * (1. + beta) * np.sin(om)
    lambda_m = lambda_m0 + 2. * np.pi * (days_arr - 80.5) / 365.
    lambda_ = lambda_m + 2. * ecc * np.sin(lambda_m - om)
    # Solar declination
    dec = np.arcsin(np.sin(eps) * np.sin(lambda_))
    # Hour angle
    h = 2 * np.pi * (days_arr + new_lon / 360.)
    # Distance
    rho = (1. - ecc ** 2.) / (1. + ecc * np.cos(lambda_ - om))

    # Insolation
    sol = S * (np.sin(np.pi / 180. * lat[None, ...]) * np.sin(dec) -
               np.cos(np.pi / 180. * lat[None, ...]) * np.cos(dec) *
               np.cos(h)) * rho ** -2.
    if clip_zero:
        sol[sol < 0.] = 0.

    return sol.astype(np.float32)


def get_best_checkpoint_path(path: str) -> str:
    """
    Returns the string of the best checkpoint in a given directory.

    :param path: The path to a checkpoints directory
    :return: The absolute path of the best checkpoint
    """
    path = os.path.abspath(path)
    ckpt_paths = np.array(glob.glob(path + "/epoch*.ckpt"))

    best_path = ""
    best_error = np.inf
    for ckpt_path in ckpt_paths:
        if "NAN" in ckpt_path:
            continue
        # Read the scientific number from the checkpoint name and perform update if appropriate
        curr_error = float(re.findall("-?\d*\.?\d+E[+-]?\d+", os.path.basename(ckpt_path))[0])
        if curr_error < best_error:
            best_path = ckpt_path
            best_error = curr_error

    return best_path


def write_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch: int,
        iteration: int,
        val_error: float,
        epochs_since_improved: int,
        dst_path: str,
        keep_n_checkpoints: int = 5
    ):
    """
    Writes a checkpoint including model, optimizer, and scheduler state dictionaries along with current epoch,
    iteration, and validation error to file.
    
    :param model: The network model
    :param optimizer: The pytorch optimizer
    :param scheduler: The pytorch learning rate scheduler
    :param epoch: Current training epoch
    :param iteration: Current training iteration
    :param val_error: The validation error of the current training
    :param epochs_since_improved: The number of epochs since the validation error improved
    :param dst_path: Path where the checkpoint is written to
    :param keep_n_checkpoints: Number of best checkpoints that will be saved (worse checkpoints are overwritten)
    """
    ckpt_dst_path = os.path.join(
        dst_path, "checkpoints",
        f"epoch={str(epoch).zfill(4)}-val_loss=" + "{:.4E}".format(val_error) + ".ckpt"
        )
    root_path = os.path.dirname(ckpt_dst_path)
    os.makedirs(root_path, exist_ok=True)
    th.save(obj={"model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "epoch": epoch + 1,
                 "iteration": iteration,
                 "val_error": val_error,
                 "epochs_since_improved": epochs_since_improved},
            f=ckpt_dst_path)
    th.save(obj={"model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "epoch": epoch + 1,
                 "iteration": iteration,
                 "val_error": val_error,
                 "epochs_since_improved": epochs_since_improved},
            f=os.path.join(root_path, "last.ckpt"))

    # Only keep top n checkpoints
    ckpt_paths = np.array(glob.glob(root_path + "/epoch*.ckpt"))
    if len(ckpt_paths) > keep_n_checkpoints + 1:
        worst_path = ""
        worst_error = -np.inf 
        for ckpt_path in ckpt_paths:
            if "NAN" in ckpt_path:
                os.remove(ckpt_path)
                continue
            # Read the scientific number from the checkpoint name and perform update if appropriate
            curr_error = float(re.findall("-?\d*\.?\d+E[+-]?\d+", os.path.basename(ckpt_path))[0])
            if curr_error > worst_error:
                worst_path = ckpt_path
                worst_error = curr_error
        os.remove(worst_path)

