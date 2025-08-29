import logging
import sys
from math import ceil

import numpy as np
import xarray as xr
import pandas as pd

import ufs2arco.utils
from ufs2arco.mpi import MPITopology

from eagle.tools.data import open_anemoi_dataset, open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import get_gridcell_area_weights
from eagle.tools.postprocess import regrid_nested_to_global

logger = logging.getLogger("ufs2arco")


def postprocess(xds, keep_t0=None):

    keep_t0 = keep_t0 if keep_t0 is not None else "cell" not in xds.dims
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


def compute_spatial_metrics():
    """Compute grid cell area weighted RMSE and MAE

    Note that the arguments documented here are passed via a config yaml as in

    Example:
        >>> python compute_error_metrics.py recipe.yaml

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

    if len(sys.argv) != 2:
        raise Exception("Did not get an argument. Usage is:\npython compute_error_metrics.py recipe.yaml")

    config = open_yaml_config(sys.argv[1])
    topo = MPITopology(log_dir=config.get("log_path", "./logs/spatial-metrics"))

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
    if "global" in model_type:
        latlon_weights = get_gridcell_area_weights(vds)

    elif model_type == "nested-lam":
        latlon_weights = 1. # Assume LAM is equal area

    elif model_type == "lam":
        latlon_weights = 1. # Assume LAM is equal area

    else:
        raise NotImplementedError

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])

    rmse_container = list() if keep_t0 else None
    mae_container = list() if keep_t0 else None

    n_dates = len(dates)
    n_batches = ceil(n_dates / topo.size)

    logger.info(f" --- Starting Metrics Computation --- ")
    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break
        try:
            t0 = dates[date_idx]
        except:
            logger.info(f"trying to get this date: {date_idx} / {n_dates}")
            raise
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"\tProcessing {st0}, batch {batch_idx} / {n_batches}")
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
        if model_type == "nested-global":
            fds = regrid_nested_to_global(
                fds,
                ds_out=vds.coords.to_dataset().load(),
                lam_index=lam_index,
                regrid_weights_filename=config.get("regrid_weights_path", "conservative_weights.nc"),
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

        logger.info(f"\tDone with {st0}")
    logger.info(f" --- Done Computing Metrics --- \n")
    if keep_t0:
        rmse_container = topo.gather(rmse_container)
        mae_container = topo.gather(mae_container)
    else:
        root_rmse = np.zeros_like(rmse_container
        root_mae = 0*mae_container
        topo.sum(rmse_container, root_rmse)
        topo.sum(mae_container, root_mae)


    if keep_t0:
        rmse_container = sorted(rmse_container, key=lambda xds: xds.coords["t0"])
        rmse_container = xr.concat(rmse_container, dim="t0")
        fname = f"{config['output_path']}/spatial.rmse.perIC.{config['model_type']}.nc"
    else:
        rmse_container = root_rmse
        fname = f"{config['output_path']}/spatial.rmse.{config['model_type']}.nc"

    rmse_container.to_netcdf(fname)

    if keep_t0:
        mae_container = sorted(mae_container, key=lambda xds: xds.coords["t0"])
        mae_container = xr.concat(mae_container, dim="t0")
        fname = f"{config['output_path']}/spatial.mae.perIC.{config['model_type']}.nc"
    else:
        mae_container = root_mae
        fname = f"{config['output_path']}/spatial.mae.{config['model_type']}.nc"

    mae_container.to_netcdf(fname)

        logger.info(f" --- Done Storing Results at {config['output_path']} --- \n")
