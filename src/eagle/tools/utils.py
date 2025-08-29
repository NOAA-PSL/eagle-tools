import os
import logging
import yaml

logger = logging.getLogger("eagle.tools")

def open_yaml_config(config_filename: str):
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    # expand any environment variables
    for key, val in config.items():
        if "path" in key:
            if isinstance(val, str):
                config[key] = os.path.expandvars(val)
            else:
                logger.warning(f"Not expanding environment variables in {key} in config, since it could be many different types")

    # if output_path is not created, make it here
    if "output_path" in config and not os.path.isdir(config["output_path"]):
        logger.info(f"Creating output_path: {config['output_path']}")
        os.makedirs(config["output_path"])

    return config


def drop_forcing_vars(xds):
    for key in [
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "orog",
        "orography",
        "geopotential_at_surface",
        "land_sea_mask",
        "lsm",
        "insolation",
        "cos_solar_zenith_angle",
    ]:
        if key in xds:
            xds = xds.drop_vars(key)
    return xds
