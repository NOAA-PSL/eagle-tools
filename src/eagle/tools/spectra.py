import logging

import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from anemoi.training.diagnostics.plots import compute_spectra as compute_array_spectra, equirectangular_projection

from eagle.tools.data import open_anemoi_dataset_with_xarray, open_anemoi_inference_dataset
from eagle.tools.data import open_forecast_zarr_dataset
from eagle.tools.metrics import postprocess
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")

def get_regular_grid(xds, min_delta):
    latlons = np.stack([xds["latitude"].values, xds["longitude"].values], axis=1)
    pc_lat, pc_lon = equirectangular_projection(latlons)

    pc_lat = np.array(pc_lat)

    # Calculate delta_lat on the projected grid
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]
    min_delta_lat = np.min(abs(non_zero_delta_lat))

    if min_delta_lat < min_delta:
        min_delta_lat = min_delta

    logger.info(f"Computing spectra with min_delta_lat = {min_delta_lat}")

    # Define a regular grid for interpolation
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / min_delta_lat))
    n_pix_lon = (n_pix_lat - 1) * 2 + 1  # 2*lmax + 1
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    return {
        "lon": pc_lon,
        "lat": pc_lat,
        "mesh_lon": grid_pc_lon,
        "mesh_lat": grid_pc_lat,
    }

def compute_power_spectrum(xds, grid):

    nds = dict()
    for varname in xds.data_vars:

        varlist = []
        for time in xds.time.values:
            yp = xds[varname].sel(time=time).values.squeeze()
            if len(yp.shape) > 1:
                yp = yp.flatten()
            nan_flag = np.isnan(yp).any()

            method = "linear" if nan_flag else "cubic"
            yp_i = griddata(
                (grid["lon"], grid["lat"]),
                yp,
                (grid["mesh_lon"], grid["mesh_lat"]),
                method=method,
                fill_value=0.0,
            )

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

    topo = config["topo"]

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    min_delta = config.get("min_delta_lat", 0.0003)
    fhr_select = config.get("fhr_select", None)
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

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    pspectra = None
    grid = None
    logger.info(f"Computing Spectra")
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

        if fhr_select is not None:
            if not isinstance(fhr_select, (list, tuple)):
                fhr_select = [fhr_select]

            time_select = [
                fds["time"].values[0] + pd.Timedelta(hours=fhr)
                for fhr in fhr_select
            ]
            fds = fds.sel(time=time_select)

        if grid is None:
            grid = get_regular_grid(fds, min_delta=min_delta)
        this_pspectra = compute_power_spectrum(fds, grid)

        if pspectra is None:
            pspectra = this_pspectra / n_dates

        else:
            pspectra += this_pspectra / n_dates

        logger.info(f"Done with {st0}")

    # We can't have more ranks than initial conditions
    if pspectra is None:
        raise ValueError(f"Cannot use more MPI ranks than initial conditions, which is {n_dates}")

    logger.info(f"Summing Results on Root Process")
    result = {}
    for key in fds.data_vars:

        local_vals = pspectra[key].values
        global_vals = np.zeros_like(local_vals)
        topo.sum(local_vals, global_vals)
        result[key] = xr.DataArray(global_vals, coords=pspectra[key].coords)
        logger.info(f" ... aggregated {key}")

    logger.info(f"Storing Results")
    if topo.is_root:

        fname = f"{config['output_path']}/spectra.{config['model_type']}.nc"
        result = xr.Dataset(result, attrs=pspectra.attrs.copy())
        for key in result.data_vars:
            result[key].attrs = pspectra[key].attrs.copy()
        result.to_netcdf(fname)
        logger.info(f"Stored result: {fname}")
    logger.info(f"Done Computing Spectra")
