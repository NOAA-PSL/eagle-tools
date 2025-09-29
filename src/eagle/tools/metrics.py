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
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean(spatial_dims+("ensemble",))
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def mae(target, prediction, weights=1., spatial_dims=("cell",)):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean(spatial_dims+("ensemble",))
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def main(config):
    """Compute grid cell area weighted RMSE and MAE.

    \b
    This function processes forecast and verification datasets over a specified
    date range, computes the Root Mean Square Error (RMSE) and Mean Absolute
    Error (MAE) between them, and saves the results to NetCDF files.

    \b
    Note:
        The arguments documented here are passed via a config dictionary.

    \b
    Config Args:
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
            f"{output_path}/rmse.{model_type}.nc" and
            f"{output_path}/mae.{model_type}.nc"
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
        \b
        forecast_regrid_kwargs (dict, optional): options passed to ufs2arco.transforms.horizontal_regrid
        \b
        target_regrid_kwargs (dict, optional): options passed to ufs2arco.transforms.horizontal_regrid
        \b
        use_mpi (bool, optional): if True, use a separate MPI process per initial condition
        \b
        log_path (str, optional): if using MPI, provide a path to where the logs get saved (one per MPI process)
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
