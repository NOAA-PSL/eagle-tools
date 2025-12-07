import logging

import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from anemoi.training.diagnostics.plots import compute_spectra as compute_array_spectra, equirectangular_projection

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_dataset_with_xarray, open_anemoi_inference_dataset
from eagle.tools.metrics import postprocess
from eagle.tools.nested import prepare_regrid_target_mask

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
            if len(yp.shape) > 1:
                yp = yp.flatten()
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

    See ``eagle-tools spectra --help`` or cli.py for help
    """

    setup_simple_log()

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    min_delta = config.get("min_delta_lat", 0.0003)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "lcc_info": config.get("lcc_info", None),
    }

    if model_type == "nested-global":
        config["forecast_regrid_kwargs"]["target_grid_path"] = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=config["forecast_regrid_kwargs"],
        )

    # Verification dataset
    vds = open_anemoi_dataset_with_xarray(
        path=config["verification_dataset_path"],
        model_type=model_type,
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
                horizontal_regrid_kwargs=config.get("forecast_regrid_kwargs", None),
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
    fname = f"{config['output_path']}/spectra.predictions.{config['model_type']}.nc"
    pspectra.to_netcdf(fname)
    logger.info(f"Stored result: {fname}")
    logger.info(f" --- Done Storing Spectra --- \n")
