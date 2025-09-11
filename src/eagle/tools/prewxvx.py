import logging

import pandas as pd
import xarray as xr
from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_inference_dataset

logger = logging.getLogger("eagle.tools")

def main(config):

    setup_simple_log()

    forecast_path = config["forecast_path"]
    model_type = config["model_type"]
    from_anemoi = config.get("from_anemoi", True)
    try:
        assert model_type == "nested-lam"
        assert from_anemoi
    except:
        raise NotImplementedError

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        path_in = f"{forecast_path}/{st0}.240h.nc"
        path_out= f"{forecast_path}/lam.{st0}.240h.nc"

        logger.info(f"Opening {path_in}")
        lds = open_anemoi_inference_dataset(
            path=path_in,
            model_type="nested-lam",
            lam_index=64220,
            levels=config.get("levels", None),
            vars_of_interest=config.get("vars_of_interest", None),
            load=True,
            reshape_cell_to_2d=True,
            lcc_info=config.get("lcc_info", None),
        )
        for key in ["x", "y"]:
            if key in lds.coords:
                lds = lds.drop_vars(key)

        lds = lds.rename({"x": "longitude", "y": "latitude"})
        lds.attrs["forecast_reference_time"] = str(lds.time.values[0])
        lds.to_netcdf(path_out)
        logger.info(f"Wrote to {path_out}")
