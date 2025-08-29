import logging
import sys
from math import ceil

import numpy as np
import xarray as xr
import pandas as pd

import ufs2arco.utils

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_dataset, open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import get_gridcell_area_weights

logger = logging.getLogger("eagle.tools")


def postprocess(xds, keep_t0=None):

    if keep_t0:
        t0 = pd.Timestamp(xds["time"][0].values)
        xds["t0"] = xr.DataArray(t0, coords={"t0": t0})
        xds = xds.set_coords("t0")
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds["fhr"] = xr.DataArray(
        xds["lead_time"].values.astype("timedelta64[h]").astype(int),
        coords=xds.time.coords,
        attrs={"description": "forecast hour, aka lead time in hours"},
    )
    xds = xds.swap_dims({"time": "fhr"}).drop_vars("time")
    xds = xds.set_coords("lead_time")
    return xds


def rmse(target, prediction, weights=1., keep_t0=False):
    result = {}
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean("ensemble")
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds, keep_t0)


def mae(target, prediction, weights=1., keep_t0=False):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean("ensemble")
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds, keep_t0)


def main(config):
    """Compute grid cell area weighted RMSE and MAE

    Note that the arguments documented here are passed via a config yaml as in

    Args:
        forecast_path (str): directory containing forecast datasets to compare against a verification dataset. For now, the convention is that, within this directory, each forecast is in a separate netcdf file named as "<initial_date>.<lead_time>.nc", where initial_date = "%Y-%m-%dT%H" and lead_time is defined below
        lead_time (str): a string indicating length, e.g. 240h or 90d, it doesn't matter what format, just make it the same as what was saved during forecast time
        verification_dataset_path (str): path to the zarr verification dataset
        model_type (str): "nested-lam", "nested-global", or "global"
        lam_index (int): number of points in nested domain that are dedicated to LAM
        output_path (str): directory to save rmse.nc and mae.nc
        start_date (str): date of first last IC to grab, in %Y-%m-%dTH format
        end_date (str): date of last last IC to grab, in %Y-%m-%dTH format
        freq (str): frequency over which to grab initial condition dates, passed to pandas.date_range
    """
    setup_simple_log()

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    keep_t0 = config.get("keep_t0", False)
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

    # Area weights
    latlon_weights = get_gridcell_area_weights(vds, model_type)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])

    rmse_container = list() if keep_t0 else None
    mae_container = list() if keep_t0 else None

    logger.info(f" --- Computing Spatial Error Metrics --- ")
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
                **subsample_kwargs,
            )
        else:

            fds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                **subsample_kwargs,
            )

        tds = vds.sel(time=fds.time.values).load()

        this_rmse = rmse(target=tds, prediction=fds, weights=latlon_weights, keep_t0=keep_t0)
        this_mae = mae(target=tds, prediction=fds, weights=latlon_weights, keep_t0=keep_t0)

        if rmse_container is None:
            rmse_container = this_rmse / len(dates)
            mae_container = this_mae / len(dates)

        else:
            if keep_t0:
                rmse_container.append(this_rmse)
                mae_container.append(this_mae)
            else:
                rmse_container += this_rmse / len(dates)
                mae_container += this_mae / len(dates)

        logger.info(f"Done with {st0}")
    logger.info(f" --- Done Computing Metrics --- \n")

    logger.info(f" --- Combining & Storing Results --- ")
    for varname, xda in zip(["rmse", "mae"], [rmse_container, mae_container]):
        if keep_t0:
            fname = f"{config['output_path']}/spatial.{varname}.perIC.{config['model_type']}.nc"
            xda = xr.concat(xda, dim="t0")
        else:
            fname = f"{config['output_path']}/spatial.{varname}.{config['model_type']}.nc"
        xda.to_netcdf(fname)
        logger.info(f"Stored result: {fname}")
    logger.info(f" --- Done Computing Spatial Error Metrics --- \n")
