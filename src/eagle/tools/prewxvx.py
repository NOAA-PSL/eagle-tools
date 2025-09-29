import os
import logging

import numpy as np
import pandas as pd
import xarray as xr

import anemoi.datasets

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset, reshape_cell_to_xy, reshape_cell_to_latlon

logger = logging.getLogger("eagle.tools")

def _rename(xds: xr.Dataset) -> xr.Dataset:
    rename = {"latitude": "lat", "longitude": "lon"}
    for key, val in rename.items():
        if key in xds:
            xds = xds.rename({key: val})
    return xds


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
    return result


def main(config):

    setup_simple_log()

    forecast_path = config["forecast_path"]
    output_path = config["output_path"]
    model_type = config["model_type"]
    from_anemoi = config.get("from_anemoi", True)
    anemoi_reference_dataset_kwargs = config.get("anemoi_reference_dataset_kwargs", None)

    open_kwargs = {
        "load": True,
        "reshape_cell_to_2d": model_type != "nested-global",
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "member": config.get("member", None),
    }

    if model_type == "nested-global":
        assert anemoi_reference_dataset_kwargs is not None, \
            f"Need to provide reference anemoi dataset to handle nested-global workflow"
        assert "lam_index" in config
        assert "lcc_info" in config
        assert "horizontal_regrid_kwargs" in config

        config["horizontal_regrid_kwargs"]["target_grid_path"] = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=config["horizontal_regrid_kwargs"],
        )

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        path_in = f"{forecast_path}/{st0}.240h.nc"
        path_out= f"{output_path}/{model_type}.{st0}.240h.nc"

        logger.info(f"Opening {path_in}")
        if from_anemoi:
            xds = open_anemoi_inference_dataset(
                path=path_in,
                model_type=model_type,
                lam_index=config.get("lam_index", None),
                lcc_info=config.get("lcc_info", None),
                **open_kwargs,
            )
        else:
            xds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                **open_kwargs,
            )

        if model_type == "nested-global":
            xds = regrid_nested_to_latlon(
                xds,
                lam_index=config["lam_index"],
                lcc_info=config["lcc_info"],
                horizontal_regrid_kwargs=config["horizontal_regrid_kwargs"],
            )

        # Clean up before storing
        for key in ["x", "y"]:
            if key in xds.coords:
                xds = xds.drop_vars(key)
        xds.attrs = {}
        if "lam" in model_type:
            xds = xds.rename({"x": "longitude", "y": "latitude"})
        xds.attrs["forecast_reference_time"] = str(xds.time.values[0])
        xds.to_netcdf(path_out)
        logger.info(f"Wrote to {path_out}")
