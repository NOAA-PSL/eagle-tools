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
    """Compute spatial maps of RMSE and MAE

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
        keep_t0 (bool, optional): If True, keeps the initial condition time (t0)
            as a separate dimension in the output file. This can produce very large results
            and requires a lot of memory. If False, the metrics
            are averaged over all initial conditions. Defaults to False.

    \b
    Config Args common to metrics.py:
        model_type (str): The type of model grid, one of: "global", "lam",
            "nested-lam", "nested-global".
            This determines how grid cell area weights, edge trimming, and coordinates are handled.
        \b
        verification_dataset_path (str): The path to the anemoi dataset with target data
            used for comparison.
        \b
        forecast_path (str): The directory path containing the forecast datasets.
        \b
        output_path (str): The directory where the output NetCDF files will be saved, as
            f"{output_path}/spatial.rmse.{model_type}.nc" and
            f"{output_path}/spatial.mae.{model_type}.nc"
            or if keep_t0=True, then as
            f"{output_path}/spatial.rmse.perIC.{model_type}.nc" and
            f"{output_path}/spatial.mae.perIC.{model_type}.nc"
        \b
        start_date (str): The first initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        end_date (str): The last initial condition date to process, in any format
            interpretable by pandas.date_range.
        \b
        freq (str): The frequency string for generating the date range between
            start_date and end_date (e.g., "6h"), passed to pandas.date_range.
        \b
        lead_time (str): A string representing the forecast lead time (e.g., "240h")
            used as part of the forecast input filename.
        \b
        from_anemoi (bool, optional): If True, opens forecast data using the
            anemoi inference dataset format. Otherwise, assumes layout of dataset
            created by ufs2arco using a base target layout. Defaults to True.
        \b
        lam_index (int, optional): For nested models (e.g., model_type="nested-lam"), this integer
            specifies the number of grid points belonging to the LAM domain.
            Defaults to None.
        \b
        levels (list, optional): A list of vertical levels to subset from the
            datasets. If None, all levels are used. Defaults to None.
        \b
        vars_of_interest (list[str], optional): A list of variable names to
            include in the analysis. If None, all variables are used. Defaults to None.
        \b
        trim_edge (int, optional): Specifies the number of grid points to trim
            from the edges of the verification dataset. Only used for LAM or Nested configurations.
            Defaults to None.
        \b
        trim_forecast_edge (int, optional): Specifies the number of grid points to
            trim from the edges of the forecast dataset. Defaults to None.
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
    logger.info(f" --- Done Storing Spatial Error Metrics --- \n")
