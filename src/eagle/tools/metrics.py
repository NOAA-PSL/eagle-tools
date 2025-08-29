import logging
from math import ceil

import numpy as np
from scipy.spatial import SphericalVoronoi
import xarray as xr
import pandas as pd

import ufs2arco.utils

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_dataset, open_anemoi_inference_dataset, open_forecast_zarr_dataset

logger = logging.getLogger("eagle.tools")


def get_gridcell_area_weights(xds, model_type):

    if "global" in model_type:
        return _area_weights(xds)

    elif model_type in ("lam", "nested-lam"):
        return 1. # Assume LAM is equal area

    else:
        raise NotImplementedError


def _area_weights(xds, unit_mean=True, radius=1, center=np.array([0,0,0]), threshold=1e-12):
    """This is a nice code block copied from anemoi-graphs"""


    x = radius * np.cos(np.deg2rad(xds["latitude"])) * np.cos(np.deg2rad(xds["longitude"]))
    y = radius * np.cos(np.deg2rad(xds["latitude"])) * np.sin(np.deg2rad(xds["longitude"]))
    z = radius * np.sin(np.deg2rad(xds["latitude"]))
    sv = SphericalVoronoi(
        points=np.stack([x,y,z], -1),
        radius=radius,
        center=center,
        threshold=threshold,
    )
    area_weight = sv.calculate_areas()
    if unit_mean:
        area_weight /= area_weight.mean()

    return area_weight


def postprocess(xds):

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


def rmse(target, prediction, weights=1.):
    result = {}
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean(["cell", "ensemble"])
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def mae(target, prediction, weights=1.):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean(["cell", "ensemble"])
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


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

    rmse_container = list()
    mae_container = list()

    logger.info(f" --- Computing Error Metrics --- ")
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

        rmse_container.append(rmse(target=tds, prediction=fds, weights=latlon_weights))
        mae_container.append(mae(target=tds, prediction=fds, weights=latlon_weights))

        logger.info(f"Done with {st0}")
    logger.info(f" --- Done Computing Metrics --- \n")

    logger.info(f" --- Combining & Storing Results --- ")
    rmse_container = xr.concat(rmse_container, dim="t0")
    mae_container = xr.concat(mae_container, dim="t0")

    for varname, xda in zip(["rmse", "mae"], [rmse_container, mae_container]):
        fname = f"{config['output_path']}/{varname}.{config['model_type']}.nc"
        xda.to_netcdf(fname)
        logger.info(f"Stored result: {fname}")
    logger.info(f" --- Done Computing Error Metrics --- \n")
