import os
import logging

import numpy as np
import pandas as pd
import xarray as xr

import anemoi.datasets

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")

def main(config):

    setup_simple_log()

    forecast_path = config["forecast_path"]
    output_path = config["output_path"]
    model_type = config["model_type"]
    from_anemoi = config.get("from_anemoi", True)

    open_kwargs = {
        "load": True,
        "reshape_cell_to_2d": model_type != "nested-global",
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "member": config.get("member", None),
        "lcc_info": config.get("lcc_info", None),
    }

    if model_type == "nested-global":
        config["forecast_regrid_kwargs"]["target_grid_path"] = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=config["forecast_regrid_kwargs"],
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
                horizontal_regrid_kwargs=config.get("forecast_regrid_kwargs", None),
                **open_kwargs,
            )
        else:
            xds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                **open_kwargs,
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
