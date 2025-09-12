import logging

import pandas as pd
import xarray as xr
from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset

logger = logging.getLogger("eagle.tools")

def main(config):

    setup_simple_log()

    forecast_path = config["forecast_path"]
    output_path = config["output_path"]
    model_type = config["model_type"]
    from_anemoi = config.get("from_anemoi", True)
    try:
        assert "lam" in model_type
    except:
        raise NotImplementedError

    open_kwargs = {
        "load": True,
        "reshape_cell_to_2d": True,
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
    }

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        path_in = f"{forecast_path}/{st0}.240h.nc"
        path_out= f"{output_path}/lam.{st0}.240h.nc"

        logger.info(f"Opening {path_in}")
        if from_anemoi:
            lds = open_anemoi_inference_dataset(
                path=path_in,
                model_type=model_type,
                lam_index=64220,
                lcc_info=config.get("lcc_info", None),
                **open_kwargs,
            )
        else:
            lds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                **open_kwargs,
            )
        for key in ["x", "y"]:
            if key in lds.coords:
                lds = lds.drop_vars(key)

        lds = lds.rename({"x": "longitude", "y": "latitude"})
        lds.attrs["forecast_reference_time"] = str(lds.time.values[0])
        lds.to_netcdf(path_out)
        logger.info(f"Wrote to {path_out}")
