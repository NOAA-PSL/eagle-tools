import logging

import numpy as np
from scipy.spatial import SphericalVoronoi
import xarray as xr
import pandas as pd

import ufs2arco.utils
from ufs2arco.transforms.horizontal_regrid import horizontal_regrid
from ufs2arco.mpi import MPITopology, SerialTopology

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_dataset, open_anemoi_inference_dataset, open_forecast_zarr_dataset, reshape_cell_to_latlon

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

    area_weight = xr.DataArray(area_weight, coords=xds.cell.coords)
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

    use_mpi = config.get("use_mpi", False)
    if use_mpi:
        topo = MPITopology(log_dir=config.get("log_path", "eagle-logs/metrics"))
        logger.setLevel(logging.INFO)
        logger.addHandler(topo.file_handler)

    else:
        topo = SerialTopology(log_dir=config.get("log_path", "eagle-logs/metrics"))


    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
    }
    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = target_regrid_kwargs or forecast_regrid_kwargs
    mkw = {}
    if do_any_regridding:
        mkw["spatial_dims"] = ("latitude", "longitude")

    # Verification dataset
    vds = open_anemoi_dataset(
        path=config["verification_dataset_path"],
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

    logger.info(f" --- Computing Error Metrics --- ")
    logger.info(f"Initial Conditions:\n{dates}")
    for batch_idx in range(n_dates):

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
                f"{config['forecast_path']}/{st0}.{config['lead_time']}.nc",
                model_type=model_type,
                lam_index=lam_index,
                trim_edge=config.get("trim_forecast_edge", None),
                load=True,
                reshape_cell_to_2d=do_any_regridding,
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

        if forecast_regrid_kwargs is not None:
            fds = horizontal_regrid(fds, **forecast_regrid_kwargs)

        tds = vds.sel(time=fds.time.values).load()
        if do_any_regridding:
            try:
                tds = reshape_cell_to_latlon(tds)
            except:
                logger.warning(f"Could not reshape target data to latlon")
        if target_regrid_kwargs is not None:
            tds = horizontal_regrid(tds, **target_regrid_kwargs)

        rmse_container.append(rmse(target=tds, prediction=fds, weights=latlon_weights, **mkw))
        mae_container.append(mae(target=tds, prediction=fds, weights=latlon_weights, **mkw))

        logger.info(f"Done with {st0}")
    logger.info(f" --- Done Computing Metrics --- \n")

    logger.info(f" --- Gathering Results on Root Process --- \n")
    rmse_container = topo.gather(rmse_container)
    mae_container = topo.gather(mae_container)

    if topo.is_root:
        rmse_container = [xds for sublist in rmse_container for xds in sublist]
        mae_container = [xds for sublist in mae_container for xds in sublist]

        # Sort before passing to xarray, potentially faster
        rmse_container = sorted(rmse_container, key=lambda xds: xds.coords["t0"])
        mae_container = sorted(mae_container, key=lambda xds: xds.coords["t0"])

        logger.info(f" --- Combining & Storing Results --- ")
        rmse_container = xr.concat(rmse_container, dim="t0")
        mae_container = xr.concat(mae_container, dim="t0")

        for varname, xda in zip(["rmse", "mae"], [rmse_container, mae_container]):
            fname = f"{config['output_path']}/{varname}.{config['model_type']}.nc"
            xda.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")

        logger.info(f" --- Done Storing Error Metrics --- \n")
