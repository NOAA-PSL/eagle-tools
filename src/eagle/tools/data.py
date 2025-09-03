import logging
from collections.abc import Sequence
import importlib.resources
import yaml

import numpy as np
import xarray as xr
import pandas as pd

from ufs2arco.utils import expand_anemoi_dataset, convert_anemoi_inference_dataset

logger = logging.getLogger("eagle.tools")


def get_xy():
    xds = xr.open_zarr("/pscratch/sd/t/timothys/nested-eagle/v0/data/hrrr.zarr")
    return {"x": xds["x"].isel(variable=0,drop=True).load(), "y": xds["y"].isel(variable=0,drop=True).load()}


def trim_xarray_edge(xds, trim_edge):
    assert all(key in xds for key in ("x", "y"))
    xds["x"].load()
    xds["y"].load()
    condx = ( (xds["x"] > trim_edge[0]-1) & (xds["x"] < xds["x"].max().values-trim_edge[1]+1) ).compute()
    condy = ( (xds["y"] > trim_edge[2]-1) & (xds["y"] < xds["y"].max().values-trim_edge[3]+1) ).compute()
    xds = xds.where(condx & condy, drop=True)
    return xds


def open_anemoi_dataset(
    path: str,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    trim_edge: Sequence[int] = None,
    rename_to_longnames: bool = False,
) -> xr.Dataset:

    xds = xr.open_zarr(path)
    vds = expand_anemoi_dataset(xds, "data", xds.attrs["variables"])
    for key in ["x", "y"]:
        if key in xds:
            vds[key] = xds[key] if "variable" not in xds[key].dims else xds[key].isel(variable=0, drop=True)
            vds = vds.set_coords(key)

    vds = subsample(vds, levels, vars_of_interest)
    if trim_edge is not None:
        vds = trim_xarray_edge(vds, trim_edge)
    if rename_to_longnames:
        vds = rename(vds)
    return vds


def open_anemoi_inference_dataset(
    path: str,
    model_type: str,
    lam_index: int | None = None,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    trim_edge: Sequence[int] = None,
    rename_to_longnames: bool = False,
    load: bool = False,
) -> xr.Dataset:
    assert model_type in ("nested-lam", "nested-global", "global")

    ids = xr.open_dataset(path, chunks="auto")
    xds = convert_anemoi_inference_dataset(ids)
    xds = subsample(xds, levels, vars_of_interest)

    if "nested" in model_type:
        assert lam_index is not None
        if "lam" in model_type:
            xds = xds.isel(cell=slice(lam_index))

    if load:
        xds = xds.load()

    if trim_edge is not None and "lam" in model_type:
        for key in ["x", "y"]:
            if key in ids:
                xds[key] = ids[key] if "variable" not in ids[key].dims else ids[key].isel(variable=0, drop=True)
                xds = xds.set_coords(key)
            else:
                xds[key] = get_xy()[key]
                xds = xds.set_coords(key)
        xds = trim_xarray_edge(xds, trim_edge)

    if rename_to_longnames:
        xds = rename(xds)
    return xds


def open_forecast_zarr_dataset(
    path: str,
    t0: pd.Timestamp,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    trim_edge: Sequence[int] = None,
    rename_to_longnames: bool = False,
    load: bool = False,
) -> xr.Dataset:
    """This is for non-anemoi forecast datasets, for example HRRR forecast data preprocessed by ufs2arco"""

    xds = xr.open_zarr(path, decode_timedelta=True)
    xds = xds.sel(t0=t0).squeeze(drop=True)
    xds["time"] = xr.DataArray(
        [pd.Timestamp(t0) + pd.Timedelta(hours=fhr) for fhr in xds.fhr.values],
        coords=xds.fhr.coords,
    )
    xds = xds.swap_dims({"fhr": "time"}).drop_vars("fhr")
    xds = subsample(xds, levels, vars_of_interest)

    # Comparing to anemoi, it's easier to flatten than unpack anemoi
    if {"x", "y"}.issubset(xds.dims):
        xds = xds.stack(cell2d=("y", "x"))
    elif {"longitude", "latitude"}.issubset(xds.dims):
        xds = xds.stack(cell2d=("latitude", "longitude"))
    else:
        raise KeyError("Unclear on the dimensions here")

    xds["cell"] = xr.DataArray(
        np.arange(len(xds.cell2d)),
        coords=xds.cell2d.coords,
    )
    xds = xds.swap_dims({"cell2d": "cell"})
    xds = xds.drop_vars(["cell2d", "t0", "valid_time"])

    if load:
        xds = xds.load()

    if trim_edge is not None:
        xds = trim_xarray_edge(xds, trim_edge)

    if rename_to_longnames:
        xds = rename(xds)

    return xds


def subsample(xds, levels=None, vars_of_interest=None):
    """Subsample vertical levels and variables
    """

    if levels is not None:
        xds = xds.sel(level=levels)

    if vars_of_interest is not None:
        if any("wind_speed" in varname for varname in vars_of_interest):
            xds = calc_wind_speed(xds, vars_of_interest)
        xds = xds[vars_of_interest]
    else:
        xds = drop_forcing_vars(xds)

    return xds


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


def _wind_speed(u, v, long_name):
    return xr.DataArray(
        np.sqrt(u**2 + v**2),
        coords=u.coords,
        attrs={
            "long_name": long_name,
            "units": "m/s",
        },
    )

def calc_wind_speed(xds, vars_of_interest):

    if "10m_wind_speed" in vars_of_interest:
        if "ugrd10m" in xds:
            u = xds["ugrd10m"]
            v = xds["vgrd10m"]
        elif "u10" in xds:
            u = xds["u10"]
            v = xds["v10"]
        elif "10m_u_component_of_wind" in xds:
            u = xds["10m_u_component_of_wind"]
            v = xds["10m_v_component_of_wind"]
        xds["10m_wind_speed"] = _wind_speed(u, v, "10m Wind Speed")

    if "80m_wind_speed" in vars_of_interest:
        if "ugrd80m" in xds:
            u = xds["ugrd80m"]
            v = xds["vgrd80m"]
        elif "u80" in xds:
            u = xds["u80"]
            v = xds["v80"]
        elif "80m_u_component_of_wind" in xds:
            u = xds["80m_u_component_of_wind"]
            v = xds["80m_v_component_of_wind"]
        xds["80m_wind_speed"] = _wind_speed(u, v, "80m Wind Speed")

    if "100m_wind_speed" in vars_of_interest:
        if "ugrd100m" in xds:
            u = xds["ugrd100m"]
            v = xds["vgrd100m"]
        elif "u100" in xds:
            u = xds["u100"]
            v = xds["v100"]
        elif "100m_u_component_of_wind" in xds:
            u = xds["100m_u_component_of_wind"]
            v = xds["100m_v_component_of_wind"]
        xds["100m_wind_speed"] = _wind_speed(u, v, "100m Wind Speed")

    if "wind_speed" in vars_of_interest:
        if "ugrd" in xds:
            u = xds["ugrd"]
            v = xds["vgrd"]
        elif "u" in xds:
            u = xds["u"]
            v = xds["v"]
        elif "u_component_of_wind" in xds:
            u = xds["u_component_of_wind"]
            v = xds["v_component_of_wind"]
        xds["wind_speed"] = _wind_speed(u, v, "Wind Speed")
    return xds

def rename(xds):
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rdict = yaml.safe_load(f)

    for key, val in rdict.items():
        if key in xds:
            xds = xds.rename({key: val})
    return xds
