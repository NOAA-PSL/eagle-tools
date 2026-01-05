import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union

import xarray as xr
import numpy as np
import pandas as pd

import xesmf as xe
from ufs2arco.utils import convert_anemoi_inference_dataset

from eagle.tools.utils import open_yaml_config

logger = logging.getLogger("eagle.tools")

def open_raw_inference(path_to_raw_inference: str) -> xr.Dataset:
    """
    Open one Anemoi-Inference run.

    Args:
        path_to_raw_inference (str): Path to an Anemoi-Inference run.

    Returns:
        xr.Dataset: One initialization of an Anemoi-Inference run.
    """
    xds = xr.open_dataset(path_to_raw_inference, chunks="auto")
    return convert_anemoi_inference_dataset(xds)


def mask_values(
    area_to_return: str, ds_nested: xr.Dataset, lam_index: int
) -> xr.Dataset:
    """
    Mask dataset values based on LAM coordinates.

    Args:
        area_to_return (str): Either "lam" or "global" to specify which area to return.
        ds (xr.Dataset): Input nested dataset to mask.
        lam_index (int): Index where nested ds transitions from LAM->global.

    Returns:
        xr.Dataset: Masked dataset containing either LAM only ds or global only (lam missing) ds.
    """
    if area_to_return == "lam":
        return ds_nested.isel(cell=slice(lam_index))
    elif area_to_return == "global":
        return ds_nested.isel(cell=slice(lam_index, None))
    else:
        raise ValueError("area_to_return must be either 'lam' or 'global'")


def create_2D_grid(
    ds: xr.Dataset,
    vars_of_interest: List[str],
    lcc_info: dict = None,
) -> xr.Dataset:
    """
    Reshape dataset from 1D 'cell' dimension to 2D latitude and longitude.

    Args:
        ds (xr.Dataset): Anemoi dataset with a flattened "cell" dimension.
        vars_of_interest (List[str]): Variables to reshape.
        lcc_info (dict): Necesary info about LCC configuation.

    Returns:
        xr.Dataset: Dataset with shape (time, latitude, longitude).
    """
    ds_to_reshape = ds.copy()
    logger.info(f"ds_to_reshape:\n{ds_to_reshape}")

    if lcc_info:
        lat_length = lcc_info["lat_length"]
        lon_length = lcc_info["lon_length"]

        time_length = len(ds_to_reshape["time"].values)

        ds_to_reshape["x"] = np.arange(0, lon_length)
        ds_to_reshape["y"] = np.arange(0, lat_length)

        lats = ds_to_reshape["latitude"][:].values.reshape((lat_length, lon_length))
        lons = ds_to_reshape["longitude"][:].values.reshape((lat_length, lon_length))

        data_vars = {}
        dims = {"time": time_length, "level": len(ds_to_reshape["level"]), "y": lat_length, "x": lon_length}
        for v in vars_of_interest:

            these_dims = dims.copy()
            if "level" not in ds_to_reshape[v].dims:
                these_dims.pop("level")
            reshaped_var = ds_to_reshape[v].values.reshape(tuple(these_dims.values()))
            data_vars[v] = (list(these_dims.keys()), reshaped_var)

        reshaped = xr.Dataset(
            data_vars=data_vars, coords={"time": ds_to_reshape["time"].values}
        )
        reshaped["latitude"] = (("y", "x"), lats)
        reshaped["longitude"] = (("y", "x"), lons)
        reshaped = reshaped.set_coords(["latitude", "longitude"])

    else:
        lats = ds_to_reshape.latitude.values
        lons = ds_to_reshape.longitude.values
        sort_index = np.lexsort((lons, lats))
        ds_to_reshape = ds_to_reshape.isel(cell=sort_index)

        lat_length = len(np.unique(ds_to_reshape.latitude.values))
        lon_length = len(np.unique(ds_to_reshape.longitude.values))
        time_length = len(ds["time"].values)

        lats = ds_to_reshape["latitude"][:].values.reshape((lat_length, lon_length))
        lons = ds_to_reshape["longitude"][:].values.reshape((lat_length, lon_length))
        lat_1d = lats[:, 0]
        lon_1d = lons[0, :]

        data_vars = {}
        dims = {"time": time_length, "level": len(ds_to_reshape["level"]), "latitude": lat_length, "longitude": lon_length}
        for v in vars_of_interest:

            these_dims = dims.copy()
            if "level" not in ds_to_reshape[v].dims:
                these_dims.pop("level")
            reshaped_var = ds_to_reshape[v].values.reshape(tuple(these_dims.values()))
            data_vars[v] = (list(these_dims.keys()), reshaped_var)

        reshaped = xr.Dataset(
            data_vars=data_vars, coords={"latitude": lat_1d, "longitude": lon_1d}
        )

    return make_contiguous(reshaped)


def make_contiguous(
    reshaped,
):
    """
    xesmf was complaining about array not being in C format?
    apparently just a performance issue - but was tired of getting the warnings :)
    """
    for var in reshaped.data_vars:
        reshaped[var].data = np.ascontiguousarray(reshaped[var].values)
    return reshaped


def final_steps(ds: xr.Dataset, time: xr.DataArray) -> xr.Dataset:
    """
    Add helpful attributes and reorganize dimensions for verification pipelines.

    Args:
        ds (xr.Dataset): Input dataset.
        time (xr.DataArray): Time coordinate.

    Returns:
        xr.Dataset: Dataset with necessary attributes for verification pipelines.
    """
    ds.attrs["forecast_reference_time"] = str(ds["time"][0].values)
    if {"x", "y"}.issubset(ds.dims):
        return ds.transpose("time", "level", "y", "x")
    elif {"latitude", "longitude"}.issubset(ds.dims):
        return ds.transpose("time", "level", "latitude", "longitude")


def regrid_ds(ds_to_regrid: xr.Dataset, ds_out: xr.Dataset) -> xr.Dataset:
    """
    Regrid a dataset.

    Args:
        ds_to_regrid (xr.Dataset): Input dataset to regrid.
        ds_out (xr.Dataset): Target grid.

    Returns:
        xr.Dataset: Regridded dataset.
    """
    regridder = xe.Regridder(
        ds_to_regrid,
        ds_out,
        method="bilinear",
        unmapped_to_nan=True,  # this makes sure anything out of conus is nan instead of zero when regridding conus only
    )
    return regridder(ds_to_regrid)


def get_conus_ds_out(
    global_ds: xr.Dataset,
    conus_ds: xr.Dataset,
    global_info: dict,
) -> xr.Dataset:
    """
    Create conus dataset on global grid.
    This will then be used for regridding high-res conus to global res.
    That will then be inserted into global domain so it's all the same resolution for verification.

    Args:
        global_ds (xr.Dataset): Global dataset.
        conus_ds (xr.Dataset): CONUS dataset.
        global_info (dict): Necessary information for global grid.

    Returns:
        xr.Dataset: Output dataset with CONUS grid.
    """
    res = global_info["res"]
    lat_min = global_info["lat_min"]
    lon_min = global_info["lon_min"]
    lat_max = global_info["lat_max"]
    lon_max = global_info["lon_max"]

    return xr.Dataset(
        {
            "latitude": (
                ["latitude"],
                np.arange(lat_min, lat_max, res),
                {"units": "degrees_north"},
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_min, lon_max, res),
                {"units": "degrees_east"},
            ),
        }
    )


def flatten_grid(ds_to_flatten: xr.Dataset, vars_of_interest: List[str]) -> xr.Dataset:
    """
    Flatten a 2D lat-lon gridded dataset back to a 1D 'values' coordinate.
    This is necessary to eventually combine global and conus back together
        after high-res conus has been regridded to global res.

    Args:
        ds_to_flatten (xr.Dataset): Dataset with 2D lat/lon grid.
        vars_of_interest (List[str]): Variables to flatten.

    Returns:
        xr.Dataset: Flattened dataset with 'values' dimension.
    """
    logger.info(f"ds_to_flatten\n{ds_to_flatten}")
    ds = ds_to_flatten.stack(cell2d=("latitude", "longitude"))
    ds = ds.dropna(dim="cell2d", subset=[vars_of_interest[0]], how="any")

    ds["cell"] = xr.DataArray(
        np.arange(len(ds["cell2d"])),
        coords=ds["cell2d"].coords,
        dims=ds["cell2d"].dims,
        attrs={
            "description": f"logical index for 'cell2d', which is a flattened lon x lat array",
        },
    )
    ds = ds.swap_dims({"cell2d": "cell"})

    # For some reason, there's a failure when trying to store this multi-index
    # it's not needed in Anemoi, so no need to keep it anyway.
    ds = ds.drop_vars("cell2d")
    logger.info(f"after:\n{ds}")
    return ds


def combine_lam_w_global(
    ds_nested_w_lam_cutout: xr.Dataset, ds_lam_w_global_res: xr.Dataset
) -> xr.Dataset:
    """
    Combine LAM (regridded to global res) and global regions into a single dataset.

    Args:
        ds_nested_w_lam_cutout (xr.Dataset): Global portion of dataset.
        ds_lam_w_global_res (xr.Dataset): Regridded LAM portion of dataset.

    Returns:
        xr.Dataset: Combined dataset.
    """
    cell = len(ds_lam_w_global_res.cell) + np.arange(len(ds_nested_w_lam_cutout.cell))
    ds_nested_w_lam_cutout["cell"] = xr.DataArray(
        cell,
        coords={"cell": cell},
    )
    logger.info(f"cutout\n{ds_nested_w_lam_cutout}")
    logger.info(f"lam\n{ds_lam_w_global_res}")
    return xr.concat([ds_nested_w_lam_cutout, ds_lam_w_global_res], dim="cell")


def postprocess_lam_only(
    ds_nested: xr.Dataset,
    lam_index: int,
    vars_of_interest: List[str],
    level_variables: List[str],
    levels: List[int],
    lcc_info: bool,
) -> xr.Dataset:
    """
    Postprocess LAM-only data.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        lam_index (int): Index where nested ds transitions from LAM->global.
        vars_of_interest (List[str]): All variables to process.
        level_variables (List[str]): Variables that have levels.
        levels (List[int]): List of levels to process.
        lcc_info (bool): Flag if lcc_flag grid (e.g. HRRR).

    Returns:
        xr.Dataset: Processed LAM dataset ready for verification :)
    """
    time = ds_nested["time"]

    ds_lam = mask_values(area_to_return="lam", ds_nested=ds_nested, lam_index=lam_index)
    ds_lam = create_2D_grid(
        ds=ds_lam, vars_of_interest=vars_of_interest, lcc_info=lcc_info
    )
    logger.info(f"after 2D grid: \n {ds_lam}\n")
    ds_lam = final_steps(ds=ds_lam, time=time)
    return ds_lam


def postprocess_global(
    ds_nested: xr.Dataset,
    lam_index: int,
    vars_of_interest: List[str],
    level_variables: List[str],
    levels: List[int],
    lcc_info: dict,
    global_info: dict,
) -> xr.Dataset:
    """
    Postprocess global data.
    This will output a global ds, and the LAM region has been regridded to global res within it.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        lam_index (int): Index where nested ds transitions from LAM->global.
        vars_of_interest (List[str]): All variables to process.
        level_variables (List[str]): Variables that have levels.
        levels (List[int]): List of levels to process.
        lcc_info (dict): Necessary information for a LCC grid.
        lcc_info (dict): Necessary information for the global grid.

    Returns:
        xr.Dataset: Post-processed global dataset.
    """
    time = ds_nested["time"]

    # create lam only ds and global only ds (lam has been cut out)
    lam_ds = mask_values(area_to_return="lam", ds_nested=ds_nested, lam_index=lam_index)
    global_ds = mask_values(
        area_to_return="global", ds_nested=ds_nested, lam_index=lam_index
    )

    # take lam from 1D to 2D (values dim -> lat/lon or x/y dims)
    lam_ds = create_2D_grid(
        ds=lam_ds, vars_of_interest=vars_of_interest, lcc_info=lcc_info
    )

    # create blank grid over conus that matches global resolution
    ds_out_conus = get_conus_ds_out(global_ds, lam_ds, global_info=global_info)

    # regrid lam to match global resolution
    lam_ds_regridded = regrid_ds(ds_to_regrid=lam_ds, ds_out=ds_out_conus)

    # flatten regridded lam back to 1D (lat/lon dims -> values dim)
    # necessary to concat it back to global grid
    ds_lam_regridded_flattened = flatten_grid(
        ds_to_flatten=lam_ds_regridded, vars_of_interest=vars_of_interest
    )

    # combine global ds and regridded lam ds together
    ds_combined = combine_lam_w_global(
        ds_nested_w_lam_cutout=global_ds, ds_lam_w_global_res=ds_lam_regridded_flattened
    )

    # go back to 2D again (lots of gynmastics here!!)
    ds_combined = create_2D_grid(ds=ds_combined, vars_of_interest=vars_of_interest)

    ds_combined = final_steps(ds=ds_combined, time=time)

    return ds_combined


def run(
    initialization: pd.Timestamp,
    config,
):
    """
    Run full pipeline.

    """
    vars_of_interest = config["vars_of_interest"]
    level_variables = config["level_variables"]
    levels = config["levels"]
    lam_index = config["lam_index"]
    lcc_info = config["lcc_info"]
    global_info = config["global_info"]
    raw_inference_files_base_path = config["raw_inference_files_base_path"]

    file_date = datetime.fromisoformat(initialization).strftime("%Y-%m-%dT%H")
    file_name = f"{file_date}.240h"

    ds_nested = open_raw_inference(
        path_to_raw_inference=f"{raw_inference_files_base_path}/{file_name}.nc"
    )

    lam_ds = postprocess_lam_only(
        ds_nested=ds_nested,
        lam_index=lam_index,
        vars_of_interest=vars_of_interest,
        level_variables=level_variables,
        levels=levels,
        lcc_info=lcc_info,
    )

    lam_ds.to_netcdf(f"lam_{file_name}.nc")

    global_ds = postprocess_global(
        ds_nested=ds_nested,
        lam_index=lam_index,
        vars_of_interest=vars_of_interest,
        level_variables=level_variables,
        levels=levels,
        lcc_info=lcc_info,
        global_info=global_info,
    )

    global_ds.to_netcdf(f"global_{file_name}.nc")

    # TODO - revisit if this is how we want to be saving files out?

    return


def main(config):

    dates = pd.date_range(start=config["start_date"], end=config["end_date"], freq=config["freq"])

    logger.info(f" --- Postprocessing Inference ---")
    logger.info(f"Initial Conditions:\n{dates}")

    for i in dates:
        run(
            initialization=str(i),
            config=config,
        )
