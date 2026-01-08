import logging
import sys
from math import ceil

import numpy as np
import xarray as xr
import pandas as pd

import ufs2arco.utils
from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.data import open_anemoi_dataset_with_xarray, open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import get_gridcell_area_weights
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")


def postprocess(xds, keep_t0=None):

    if keep_t0:
        t0 = pd.Timestamp(xds["time"][0].values)
        xds["t0"] = xr.DataArray(t0, coords={"t0": t0})
        xds = xds.set_coords("t0")
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds["lead_time"].attrs = {} # remove any calendar details from the attributes
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
        mse = se.mean("member")
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds, keep_t0)


def mae(target, prediction, weights=1., keep_t0=False):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean("member")
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds, keep_t0)


def main(config):
    """Compute spatial maps of RMSE and MAE

    See ``eagle-tools spatial --help`` or cli.py for help
    """
    topo = config["topo"]
    if config["use_mpi"]:
        raise NotImplementedError

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    keep_t0 = config.get("keep_t0", False)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "lcc_info": config.get("lcc_info", None),
    }
    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = (target_regrid_kwargs is not None) or \
            ((forecast_regrid_kwargs is not None) and (model_type != "nested-global"))

    if model_type == "nested-global":
        forecast_regrid_kwargs["target_grid_path"] = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=forecast_regrid_kwargs,
        )

    # Verification dataset
    vds = open_anemoi_dataset_with_xarray(
        path=config["verification_dataset_path"],
        model_type=model_type,
        trim_edge=config.get("trim_edge", None),
        reshape_cell_to_2d=True,
        **subsample_kwargs,
    )

    # Area weights
    latlon_weights = get_gridcell_area_weights(
        vds,
        model_type,
        reshape_cell_to_2d=True,
        regrid_kwargs=target_regrid_kwargs,
    )

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])

    rmse_container = list() if keep_t0 else None
    mae_container = list() if keep_t0 else None

    logger.info(f"Computing Spatial Error Metrics")
    logger.info(f"Initial Conditions:\n{dates}")
    for t0 in dates:

        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")
        if config.get("from_anemoi", True):

            fds = open_anemoi_inference_dataset(
                f"{config['forecast_path']}/{st0}.{config['lead_time']}h.nc",
                model_type=model_type,
                lam_index=lam_index,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                reshape_cell_to_2d=True,
                horizontal_regrid_kwargs=forecast_regrid_kwargs if model_type == "nested-global" else None,
                **subsample_kwargs,
            )
        else:

            fds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                reshape_cell_to_2d=True,
                **subsample_kwargs,
            )

        if forecast_regrid_kwargs is not None and model_type != "nested-global":
            fds = horizontal_regrid(fds, **forecast_regrid_kwargs)

        tds = vds.sel(time=fds.time.values).load()
        if target_regrid_kwargs is not None:
            tds = horizontal_regrid(tds, **target_regrid_kwargs)

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
    logger.info(f"Done Computing Metrics")

    logger.info(f"Combining & Storing Results")
    for varname, xda in zip(["rmse", "mae"], [rmse_container, mae_container]):
        if keep_t0:
            fname = f"{config['output_path']}/spatial.{varname}.perIC.{config['model_type']}.nc"
            xda = xr.concat(xda, dim="t0")
        else:
            fname = f"{config['output_path']}/spatial.{varname}.{config['model_type']}.nc"
        xda.to_netcdf(fname)
        logger.info(f"Stored result: {fname}")
    logger.info(f"Done Storing Spatial Error Metrics")
