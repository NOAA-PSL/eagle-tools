import logging

import numpy as np
from scipy.spatial import SphericalVoronoi
import xarray as xr
import pandas as pd

import ufs2arco.utils
from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.data import open_anemoi_dataset_with_xarray, open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.reshape import flatten_to_cell
from eagle.tools.reshape import reshape_cell_to_latlon
from eagle.tools.reshape import reshape_cell_dim
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")


def get_gridcell_area_weights(xds, model_type, reshape_cell_to_2d=False, regrid_kwargs=None):

    if "global" in model_type:
        weights = _area_weights(xds, reshape_cell_to_2d=reshape_cell_to_2d)
        if regrid_kwargs is not None:
            weights = horizontal_regrid(weights.to_dataset(name="weights"), **regrid_kwargs)["weights"]

        return weights


    elif model_type in ("lam", "nested-lam"):
        return 1. # Assume LAM is equal area

    else:
        raise NotImplementedError


def _area_weights(xds, unit_mean=True, radius=1, center=np.array([0,0,0]), threshold=1e-12, reshape_cell_to_2d=False):
    """This is a nice code block copied from anemoi-graphs"""

    cds = xds.coords.to_dataset().copy()
    if "cell" not in cds["latitude"].dims:
        cds = flatten_to_cell(cds)

    x = radius * np.cos(np.deg2rad(cds["latitude"])) * np.cos(np.deg2rad(cds["longitude"]))
    y = radius * np.cos(np.deg2rad(cds["latitude"])) * np.sin(np.deg2rad(cds["longitude"]))
    z = radius * np.sin(np.deg2rad(cds["latitude"]))
    sv = SphericalVoronoi(
        points=np.stack([x,y,z], -1),
        radius=radius,
        center=center,
        threshold=threshold,
    )
    area_weight = sv.calculate_areas()
    if unit_mean:
        area_weight /= area_weight.mean()

    area_weight = xr.DataArray(area_weight, coords=cds.cell.coords)
    if reshape_cell_to_2d:
        try:
            ads = reshape_cell_to_latlon(area_weight.to_dataset(name="weights"))
            area_weight = ads["weights"]
        except:
            logger.warning("Could not reshape area weights to lat/lon")
    return area_weight


def postprocess(xds):

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


def rmse(target, prediction, weights=1., spatial_dims=("cell",)):
    result = {}
    dims = tuple(d for d in target.dims if d not in ("time", "level"))
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean(dims)
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def mae(target, prediction, weights=1., spatial_dims=("cell",)):
    result = {}
    dims = tuple(d for d in target.dims if d not in ("time", "level"))
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean(dims)
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def main(config):
    """Compute grid cell area weighted RMSE and MAE.

    See ``eagle-tools metrics --help`` or cli.py for help
    """

    topo = config["topo"]

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "lcc_info": config.get("lcc_info", None),
    }
    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = (target_regrid_kwargs is not None) or \
            ((forecast_regrid_kwargs is not None) and (model_type != "nested-global"))
    mkw = {}
    if do_any_regridding:
        mkw["spatial_dims"] = ("latitude", "longitude")

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
        **subsample_kwargs,
    )

    # Area weights
    latlon_weights = get_gridcell_area_weights(
        vds,
        model_type,
        reshape_cell_to_2d=do_any_regridding,
        regrid_kwargs=target_regrid_kwargs,
    )

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    rmse_container = list()
    mae_container = list()

    logger.info(f"Computing Error Metrics")
    logger.info(f"Initial Conditions:\n{dates}")
    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break # last batch situation

        try:
            t0 = dates[date_idx]
        except:
            logger.error(f"Error getting this date: {date_idx} / {n_dates}")
            raise

        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")
        if config.get("from_anemoi", True):

            fds = open_anemoi_inference_dataset(
                f"{config['forecast_path']}/{st0}.{config['lead_time']}h.nc",
                model_type=model_type,
                lam_index=lam_index,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                reshape_cell_to_2d=do_any_regridding,
                horizontal_regrid_kwargs=forecast_regrid_kwargs if model_type == "nested-global" else None,
                **subsample_kwargs,
            )
        else:

            fds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                reshape_cell_to_2d=do_any_regridding,
                **subsample_kwargs,
            )

        if forecast_regrid_kwargs is not None and model_type != "nested-global":
            fds = horizontal_regrid(fds, **forecast_regrid_kwargs)

        tds = vds.sel(time=fds.time.values).load()
        if do_any_regridding:
            tds = reshape_cell_dim(tds, model_type, subsample_kwargs["lcc_info"])

        if target_regrid_kwargs is not None:
            tds = horizontal_regrid(tds, **target_regrid_kwargs)

        rmse_container.append(rmse(target=tds, prediction=fds, weights=latlon_weights, **mkw))
        mae_container.append(mae(target=tds, prediction=fds, weights=latlon_weights, **mkw))

        logger.info(f"Done with {st0}")
    logger.info(f"Done Computing Metrics")

    logger.info(f"Gathering Results on Root Process")
    rmse_container = topo.gather(rmse_container)
    mae_container = topo.gather(mae_container)

    if topo.is_root:
        if config["use_mpi"]:
            rmse_container = [xds for sublist in rmse_container for xds in sublist]
            mae_container = [xds for sublist in mae_container for xds in sublist]

        # Sort before passing to xarray, potentially faster
        rmse_container = sorted(rmse_container, key=lambda xds: xds.coords["t0"])
        mae_container = sorted(mae_container, key=lambda xds: xds.coords["t0"])

        logger.info(f"Combining & Storing Results")
        rmse_container = xr.concat(rmse_container, dim="t0")
        mae_container = xr.concat(mae_container, dim="t0")

        for varname, xda in zip(["rmse", "mae"], [rmse_container, mae_container]):
            fname = f"{config['output_path']}/{varname}.{config['model_type']}.nc"
            xda.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")

        logger.info(f"Done Storing Error Metrics")
