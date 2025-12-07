import os
import logging

import numpy as np
import pandas as pd
import xarray as xr

import anemoi.datasets

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.reshape import reshape_cell_to_xy, reshape_cell_to_latlon

logger = logging.getLogger("eagle.tools")

def _rename(xds: xr.Dataset) -> xr.Dataset:
    rename = {"latitude": "lat", "longitude": "lon"}
    for key, val in rename.items():
        if key in xds:
            xds = xds.rename({key: val})
    return xds

def get_nested_plot_box(xds: xr.Dataset, lam_index: int, lcc_info: dict) -> dict:
    """
    Args:
        xds (xr.Dataset): created by ``eagle.tools.data.open_anemoi_dataset()``
        lcc_info
    """
    lam = xds.isel(cell=slice(lam_index))
    lam = lam.drop_vars(xds.data_vars)
    lam = reshape_cell_to_xy(lam, **lcc_info)

    box = {}
    for key in ["x", "y"]:
        idx = np.arange(0, len(lam[key]), 10)
        if idx[-1] != len(lam[key]) - 1:
            idx = np.append(idx, len(lam[key]) - 1)
        box[key] = idx
    box["longitude"] = lam["longitude"].values
    box["latitude"] = lam["latitude"].values
    return box


def prepare_regrid_target_mask(
    anemoi_reference_dataset_kwargs: dict,
    horizontal_regrid_kwargs: dict,
) -> str:
    """check if the target mask is there, otherwise add it and store it in a new spot"""

    target_grid_path = os.path.expandvars(horizontal_regrid_kwargs["target_grid_path"])
    regridder_kwargs = horizontal_regrid_kwargs["regridder_kwargs"]

    kw = horizontal_regrid_kwargs.get("open_target_kwargs", {})
    tds = xr.open_dataset(target_grid_path, **kw)
    tds = _rename(tds)
    if "mask" not in tds.data_vars and "mask" not in tds.coords:
        logger.info(f"Could not find mask in target dataset, but it's needed for conservative_normed regridding. Computing it now")

        # Open anemoi dataset, get the mask and reshape it
        ads = anemoi.datasets.open_dataset(**anemoi_reference_dataset_kwargs)
        mask2d = ads.global_mask.reshape( (len(tds["lat"]), len(tds["lon"])) )
        tds["mask"] = xr.DataArray(
            np.where(mask2d, 0, 1),
            dims=("lat", "lon"),
        )

        # store it
        fileparts = os.path.splitext(target_grid_path)
        target_grid_path = f"{fileparts[0]}_with_mask{fileparts[1]}"
        tds.to_netcdf(target_grid_path)
        logger.info(f"Stored {target_grid_path}")

    return target_grid_path


def regrid_nested_to_latlon(
    xds: xr.Dataset,
    lam_index: int,
    lcc_info: dict,
    horizontal_regrid_kwargs: dict,
) -> xr.Dataset:

    assert "regridder_kwargs" in horizontal_regrid_kwargs
    assert "method" in horizontal_regrid_kwargs["regridder_kwargs"]
    assert horizontal_regrid_kwargs["regridder_kwargs"]["method"] == "conservative_normed"

    # Get the LAM portion and global without lam portion
    lds = xds.isel(cell=slice(lam_index))
    lds = reshape_cell_to_xy(lds, **lcc_info)
    cds = xds.isel(cell=slice(lam_index, None))
    lat_is_ascending = cds["latitude"][0] < cds["latitude"][-1]


    # Regrid to global resolution, stack lat/lon, drop non LAM part
    lam_on_global = horizontal_regrid(lds, **horizontal_regrid_kwargs)
    lam_on_global = lam_on_global.stack(cell2d=("latitude", "longitude"))
    lam_on_global = lam_on_global.swap_dims({"cell2d": "cell"})
    lam_on_global = lam_on_global.drop_vars("cell2d")
    lam_on_global = lam_on_global.dropna("cell")

    # create a "cell" index for the two components
    # note that these are dimensions without variable values at this point
    cutout_cell = np.arange(len(cds.cell))
    lam_cell = len(cutout_cell) + np.arange(len(lam_on_global.cell))
    cds["cell"] = xr.DataArray(cutout_cell, coords={"cell": cutout_cell})
    lam_on_global["cell"] = xr.DataArray(lam_cell, coords={"cell": lam_cell})

    # concat global+lam, resort, and expand to latlon again
    result = xr.concat([cds, lam_on_global], dim="cell")
    sort_index = np.lexsort( (result["longitude"], result["latitude"]) )
    result = result.isel(cell=sort_index)
    result = reshape_cell_to_latlon(result)
    result = result.sortby("latitude", ascending=lat_is_ascending)
    return result
