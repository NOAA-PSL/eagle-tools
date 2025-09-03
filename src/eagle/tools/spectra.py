import logging

import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from anemoi.training.diagnostics.plots import compute_spectra as compute_array_spectra, equirectangular_projection

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_dataset, open_anemoi_inference_dataset
from eagle.tools.metrics import postprocess

logger = logging.getLogger("eagle.tools")


def compute_power_spectrum(xds, latlons, min_delta):

    pc_lat, pc_lon = equirectangular_projection(latlons)

    pc_lat = np.array(pc_lat)
    # Calculate delta_lat on the projected grid
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]
    min_delta_lat = np.min(abs(non_zero_delta_lat))

    if min_delta_lat < min_delta:
        min_delta_lat = min_delta

    # Define a regular grid for interpolation
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / min_delta_lat))
    n_pix_lon = (n_pix_lat - 1) * 2 + 1  # 2*lmax + 1
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    nds = dict()
    for varname in xds.data_vars:

        varlist = []
        for time in xds.time.values:
            yp = xds[varname].sel(time=time).values.squeeze()
            nan_flag = np.isnan(yp).any()

            method = "linear" if nan_flag else "cubic"
            yp_i = griddata((pc_lon, pc_lat), yp, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

            # Masking NaN values
            if nan_flag:
                mask = np.isnan(yp_i)
                if mask.any():
                    yp_i = np.where(mask, 0.0, yp_i)

            amplitude = np.array(compute_array_spectra(yp_i))
            varlist.append(amplitude)

        xamp = xr.DataArray(
            np.array(varlist),
            coords={"time": xds.time.values, "k": np.arange(len(amplitude))},
            dims=("time", "k",),
        )

        nds[varname] = xamp
    return postprocess(xr.Dataset(nds))


def main(config):
    """Compute the Power Spectrum averaged over all initial conditions

    Note:
        The arguments documented here are passed via a config dictionary.

    Config Args:
        min_delta_lat (float, optional): The minimum delta latitude used as a
            parameter for the power spectrum computation. Defaults to 0.0003.

    Config Args common to metrics.py:
        model_type (str): The type of model grid, one of: "global", "lam",
            "nested-lam", "nested-global".
            This determines how grid cell area weights, edge trimming, and coordinates are handled.
        verification_dataset_path (str): The path to the zarr verification
            dataset used as targets for comparison.
        forecast_path (str): The directory path containing the forecast datasets.
        output_path (str): The directory where the output NetCDF files will be saved, as
            f"{output_path}/spectra.{model_type}.nc"
        start_date (str): The first initial condition date to process, in any format
            interpretable by pandas.date_range.
        end_date (str): The last initial condition date to process, in any format
            interpretable by pandas.date_range.
        freq (str): The frequency string for generating the date range between
            start_date and end_date (e.g., "6h"), passed to pandas.date_range.
        lead_time (str): A string representing the forecast lead time (e.g., "240h")
            used as part of the forecast input filename.
        from_anemoi (bool, optional): If True, opens forecast data using the
            anemoi inference dataset format. Otherwise, assumes layout of dataset
            created by ufs2arco using a base target layout. Defaults to True.
        lam_index (int, optional): For nested models (e.g., model_type="nested-lam"), this integer
            specifies the number of grid points belonging to the LAM domain.
            Defaults to None.
        levels (list, optional): A list of vertical levels to subset from the
            datasets. If None, all levels are used. Defaults to None.
        vars_of_interest (list[str], optional): A list of variable names to
            include in the analysis. If None, all variables are used. Defaults to None.
        trim_edge (int, optional): Specifies the number of grid points to trim
            from the edges of the verification dataset. Only used for LAM or Nested configurations.
            Defaults to None.
        trim_forecast_edge (int, optional): Specifies the number of grid points to
            trim from the edges of the forecast dataset. Defaults to None.
    """

    setup_simple_log()

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    min_delta = config.get("min_delta_lat", 0.0003)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
    }

    # Verification dataset
    vds = open_anemoi_dataset(
        path=config["verification_dataset_path"],
        trim_edge=config.get("trim_edge", None),
        **subsample_kwargs,
    )
    latlons = np.stack([vds["latitude"].values, vds["longitude"].values], axis=1)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])

    pspectra = None
    logger.info(f" --- Computing Spectra --- ")
    logger.info(f"Initial Conditions:\n{dates}")
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")
        if config.get("from_anemoi", True):

            fds = open_anemoi_inference_dataset(
                f"{config['forecast_path']}/{st0}.{config['lead_time']}.nc",
                model_type=model_type,
                lam_index=lam_index,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                **subsample_kwargs,
            )
        else:

            fds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                **subsample_kwargs,
            )

        this_pspectra = compute_power_spectrum(fds, latlons=latlons, min_delta=min_delta)

        if pspectra is None:
            pspectra = this_pspectra / len(dates)

        else:
            pspectra += this_pspectra / len(dates)

        logger.info(f"Done with {st0}")
    logger.info(f" --- Done Computing Spectra --- ")

    logger.info(f" --- Combining & Storing Results --- ")
    fname = f"{config['output_path']}/spectra.{config['model_type']}.nc"
    pspectra.to_netcdf(fname)
    logger.info(f"Stored result: {fname}")
    logger.info(f" --- Done Storing Spectra --- \n")
