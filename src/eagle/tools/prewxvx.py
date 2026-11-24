import os
import logging
import glob
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import importlib
import yaml

import anemoi.datasets

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")
import cfgrib
import xarray.backends.api as backends_api


from pathlib import Path
from eccodes import (
    codes_index_new_from_file,
    codes_index_get,
    codes_index_select,
    codes_new_from_index,
    codes_write,
    codes_release,
    codes_index_release,
    codes_grib_new_from_samples,
    codes_set,
    codes_set_values,
)

# Fallback GRIB keys (paramId, typeOfLevel, level) for variables lacking GRIB_* attributes
_GRIB_PARAMS = {
    "u10": (165, "heightAboveGround", 10),
    "v10": (166, "heightAboveGround", 10),
    "t2m": (167, "heightAboveGround", 2),
    "d2m": (168, "heightAboveGround", 2),
    "sh2": (174096, "heightAboveGround", 2),
    "u100": (228246, "heightAboveGround", 100),
    "v100": (228247, "heightAboveGround", 100),
    "u200": (228239, "heightAboveGround", 200),
    "v200": (228240, "heightAboveGround", 200),
    "sp": (134, "surface", 0),
    "skt": (235, "surface", 0),
    "msl": (151, "meanSea", 0),
    "tp": (228, "surface", 0),
}

def write_grib(ds, path, bits_per_value=24):
    """Write a regular lat/lon dataset to GRIB2 using eccodes.

    Expects dims ``(time, [member], [level], latitude, longitude)`` per variable,
    with ``time`` holding valid times. The reference time comes from
    ``ds.attrs["forecast_reference_time"]`` (falling back to the first valid time),
    and each message's step is its valid time minus the reference time.
    GRIB keys come from a variable's ``GRIB_*`` attrs when present, otherwise from
    ``_GRIB_PARAMS``, otherwise the variable name is used as the ``shortName``.
    """
    for dim in ("latitude", "longitude"):
        if dim not in ds.dims:
            raise ValueError(f"GRIB output requires a regular lat/lon grid, missing '{dim}' dimension")
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    ds = ds.load()

    lat = ds.latitude.values
    lon = ds.longitude.values
    ref = pd.Timestamp(ds.attrs.get("forecast_reference_time", ds.time.values[0]))
    valid_times = pd.to_datetime(ds.time.values)
    missing = 9999.0

    with open(path, "wb") as f:
        for name, da in ds.data_vars.items():
            extra = [d for d in da.dims if d not in ("time", "member", "level", "latitude", "longitude")]
            if extra:
                raise ValueError(f"Cannot write {name} to GRIB, unsupported dims: {extra}")
            da = da.transpose(*[d for d in ("time", "member", "level", "latitude", "longitude") if d in da.dims])

            attrs = da.attrs
            # The level is a scalar coordinate that gets dropped on read, so it's not in the
            # attrs; take the table value first and let any GRIB_* attrs override it.
            param_id, type_of_level, level_value = _GRIB_PARAMS.get(name, (None, None, None))
            if "GRIB_paramId" in attrs:
                param_id = int(attrs["GRIB_paramId"])
            if "GRIB_typeOfLevel" in attrs and name not in _GRIB_PARAMS:
                type_of_level = attrs["GRIB_typeOfLevel"]
            if "GRIB_level" in attrs:
                level_value = attrs["GRIB_level"]
            if param_id is not None and level_value is None and "level" not in da.dims:
                logger.warning(f"No GRIB level known for {name}; add it to _GRIB_PARAMS or messages will collide")

            members = da.member.values if "member" in da.dims else [None]
            levels = da.level.values if "level" in da.dims else [None]

            for it, vt in enumerate(valid_times):
                step_hours = int((vt - ref) / pd.Timedelta(hours=1))
                for member in members:
                    for lev in levels:
                        field = da.isel(time=it)
                        if member is not None:
                            field = field.sel(member=member)
                        if lev is not None:
                            field = field.sel(level=lev)
                        values = field.values.astype("float64")
                        has_nan = bool(np.isnan(values).any())

                        gid = codes_grib_new_from_samples("regular_ll_sfc_grib2")
                        try:
                            if member is not None:
                                codes_set(gid, "productDefinitionTemplateNumber", 1)
                                codes_set(gid, "typeOfEnsembleForecast", 3)
                                codes_set(gid, "perturbationNumber", int(member))
                                codes_set(gid, "numberOfForecastsInEnsemble", int(da.member.size))
                            if param_id is not None:
                                codes_set(gid, "paramId", param_id)
                            else:
                                codes_set(gid, "shortName", name)

                            if lev is not None:
                                codes_set(gid, "typeOfLevel", "isobaricInhPa")
                                codes_set(gid, "level", int(lev))
                            elif type_of_level is not None:
                                codes_set(gid, "typeOfLevel", type_of_level)
                                codes_set(gid, "level", int(level_value or 0))

                            codes_set(gid, "dataDate", int(ref.strftime("%Y%m%d")))
                            codes_set(gid, "dataTime", int(ref.strftime("%H%M")))
                            codes_set(gid, "stepUnits", 1)
                            codes_set(gid, "step", step_hours)

                            codes_set(gid, "Ni", lon.size)
                            codes_set(gid, "Nj", lat.size)
                            codes_set(gid, "latitudeOfFirstGridPointInDegrees", float(lat[0]))
                            codes_set(gid, "latitudeOfLastGridPointInDegrees", float(lat[-1]))
                            codes_set(gid, "longitudeOfFirstGridPointInDegrees", float(lon[0]))
                            codes_set(gid, "longitudeOfLastGridPointInDegrees", float(lon[-1]))
                            codes_set(gid, "iDirectionIncrementInDegrees", float(abs(lon[1] - lon[0])))
                            codes_set(gid, "jDirectionIncrementInDegrees", float(abs(lat[1] - lat[0])))
                            codes_set(gid, "iScansNegatively", int(lon[1] < lon[0]))
                            codes_set(gid, "jScansPositively", int(lat[1] > lat[0]))

                            codes_set(gid, "bitsPerValue", bits_per_value)
                            if has_nan:
                                codes_set(gid, "bitmapPresent", 1)
                                codes_set(gid, "missingValue", missing)
                                values = np.where(np.isnan(values), missing, values)
                            codes_set_values(gid, values.ravel())
                            codes_write(gid, f)
                        finally:
                            codes_release(gid)


def write(ds, path, ext):
    """Write ``ds`` to ``path`` in the same format as the input (netcdf, zarr, or grib)."""
    if ext == "zarr":
        ds.to_zarr(path, mode="w")
    elif ext == "grib":
        write_grib(ds, path)
    else:
        ds.to_netcdf(path)
    logger.info(f"Wrote to {path}")
def _load_format_config():
    """Load file format extension configuration from format_extensions.yaml."""
    path = importlib.resources.files("eagle.tools.config") / "format_extensions.yaml"
    with path.open("r") as f:
        return yaml.safe_load(f)

def main(config):
    if isinstance(config, str):
        from eagle.tools.utils import setup
        config = setup(config, "prewxvx")

    topo = config["topo"]

    forecast_path = config["forecast_path"]
    output_path = config["output_path"]
    model_type = config["model_type"]
    from_anemoi = config.get("from_anemoi", True)
    lead_time = config["lead_time"]
    format = config.get("format", None)

    per_timestep = config.get("per_timestep", False)
    timestep_take_subset = config.get("timestep_subset", {}).get("take_subset", False)
    timestep_subset_list = config.get("timestep_subset", {}).get("subset_list", list())

    per_variable = config.get("per_variable", False)
    variable_take_subset = config.get("variable_subset", {}).get("take_subset", False)
    variable_subset_list = config.get("variable_subset", {}).get("subset_list", list())

    open_kwargs = {
        "load": True,
        "reshape_cell_to_2d": True,
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "member": config.get("member", None),
        "lcc_info": config.get("lcc_info", None),
        "rename_to_longnames": config.get("rename_to_longnames", False)
    }

    if model_type == "nested-global":
        config["forecast_regrid_kwargs"]["target_grid_path"], mask = (
            prepare_regrid_target_mask(
                anemoi_reference_dataset_kwargs=config[
                    "anemoi_reference_dataset_kwargs"
                ],
                horizontal_regrid_kwargs=config["forecast_regrid_kwargs"],
            )
        )

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))
    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break # Last batch situation

        try:
            t0 = dates[date_idx]
        except:
            logger.error(f"Error getting this date: {date_idx} / {n_dates}")
            raise

        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        format_config = _load_format_config()
        if format is None:
            logger.warning(f"No file format specified in config yaml.")
            try: 
                extensions = ("nc", "zarr", "grib")
                forecasts = []
                for ext in extensions:
                        forecasts.extend(glob.glob(f"{forecast_path}/{st0}.{lead_time}h.{ext}"))

                path_in = forecasts[0]
                _, ext = os.path.splitext(path_in)
                ext = ext.lstrip(".")

                logger.info(f"Found the following forecasts: {forecasts}")
            except:
                logger.error(f"Error getting forecast: {st0}.{lead_time}h")
                raise



            path_out = f"{output_path}/{model_type}.{st0}.{lead_time}h.{ext}"
        else:
            ext = format_config[format]
            path_in = f"{forecast_path}/{st0}.{lead_time}h.{ext}"
            path_out = f"{output_path}/{model_type}.{st0}.{lead_time}h.{ext}"
        

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
        print(f"\n\nDATASET: {xds}\n\n")
        xds.attrs = {}
        if "lam" in model_type and config.get("rename_curvilinear_coords_to_latlon", True):
            for key in ["x", "y"]:
                if key in xds.coords:
                    xds = xds.drop_vars(key)
            xds = xds.rename({"x": "longitude", "y": "latitude"})

        if model_type == "nested-global":
            xds["nest_mask"] = mask.rename({"lat": "latitude", "lon": "longitude"})
            xds["nest_mask"].attrs = {
                "long_name": "Nest mask",
                "description": "Binary mask. 1 = grid cell is in the nested region, 0 = rest of the globe",
            }
            xds = xds.set_coords("nest_mask")

        # Attributes
        xds.attrs["forecast_reference_time"] = str(xds.time.values[0])

        user_attributes = config.get("attributes", {})
        xds.attrs.update(user_attributes.get("dataset", {}))
        for varname, var_attrs in user_attributes.get("variables", {}).items():
            if varname in xds:
                xds[varname].attrs.update(var_attrs)
            else:
                logger.warning(f"Variable '{varname}' not found in dataset, skipping attributes")

        # Add descriptive meta info for diagnostic fields, which have all NaNs in the first timestamp
        for varname in xds.data_vars:
            t0 = xds[varname].isel(time=0)
            num_nans = np.isnan(t0).sum().values
            num_cell = np.prod(t0.shape)
            if num_nans == num_cell:
                logger.info(f"Found diagnostic {varname}")
                note = xds[varname].attrs.get("diagnostic_note", "")
                if len(note) > 0:
                    note += " "
                note += f"{varname} is diagnosed by the model, so the initial condition is all NaNs"
                xds[varname].attrs["diagnostic_note"] = note

        # Sort the data variables
        xds = xds[sorted(xds.data_vars)]

        # clip dataset to provided timestep subset
        if timestep_take_subset:
            timestep_subset_list = pd.to_datetime(timestep_subset_list)

            available_times = xds.time.to_index().intersection(timestep_subset_list)

            logger.info(f"Clipping output data to timestep(s): {available_times.strftime("%Y-%m-%d %H:%M").tolist()}")
            xds = xds.sel(time=available_times)

            timesteps_not_found = timestep_subset_list.difference(xds.time.to_index())
            if not timesteps_not_found.empty:
                logger.warning(f"The following timestep(s) provided in config yaml not found in dataset: {timesteps_not_found.strftime("%Y-%m-%d %H:%M").tolist()}")

        # subset variables
        if variable_take_subset:
            available_vars = [v for v in variable_subset_list if v in xds.data_vars]

            logger.info(f"Clipping output data to variable(s): {available_vars}")
            xds = xds[available_vars]

            variables_not_found = [v for v in variable_subset_list if v not in xds.data_vars]
            if len(variables_not_found) != 0 :
                logger.warning(f"The following variable(s) provided in config yaml not found in dataset: {variables_not_found}")



        if (per_timestep and per_variable):
            for timestep in xds.time.to_index():
                for variable in xds.data_vars:
                    sub_path_out = path_out.replace(f".{ext}", f".timestep_{timestep.strftime('%Y-%m-%dT%H')}.variable_{variable}.{ext}")
                    sub_xds = xds.sel(time=[timestep])[[variable]]
                    write(sub_xds, sub_path_out,ext)

        elif (per_timestep and not per_variable):
            for timestep in xds.time.to_index():
                sub_path_out = path_out.replace(f".{ext}", f".timestep_{timestep.strftime('%Y-%m-%dT%H')}.{ext}")
                sub_xds = xds.sel(time=[timestep])
                write(sub_xds, sub_path_out,ext)

        elif (per_variable and not per_timestep):
            for variable in xds.data_vars:
                sub_path_out = path_out.replace(f".{ext}", f".variable_{variable}.{ext}")
                sub_xds = xds[[variable]]
                write(sub_xds, sub_path_out,ext)

        else:
            # Chunking
            chunks = config.get("chunks", None)
            if chunks is not None:
                xds = xds.chunk(chunks)
            write(xds, path_out, ext)
    logger.info(f"Done with prewxvx workflow")
