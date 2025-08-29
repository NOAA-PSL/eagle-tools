import logging

import numpy as np
import xarray as xr
import pandas as pd

from ufs2arco.utils import expand_anemoi_dataset, convert_anemoi_inference_dataset

logger = logging.getLogger("eagle.tools")

_extra_coords = get_xy()

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


def open_anemoi_dataset(path, trim_edge=None, levels=None, vars_of_interest=None):

    xds = xr.open_zarr(path)
    vds = expand_anemoi_dataset(xds, "data", xds.attrs["variables"])
    for key in ["x", "y"]:
        if key in xds:
            vds[key] = xds[key] if "variable" not in xds[key].dims else xds[key].isel(variable=0, drop=True)
            vds = vds.set_coords(key)

    vds = subsample(vds, levels, vars_of_interest)
    if trim_edge is not None:
        vds = trim_xarray_edge(vds, trim_edge)
    return vds


def open_anemoi_inference_dataset(path, model_type, lam_index=None, levels=None, vars_of_interest=None, trim_edge=None):
    assert model_type in ("nested-lam", "nested-global", "global")

    ids = xr.open_dataset(path, chunks="auto")
    xds = convert_anemoi_inference_dataset(ids)
    xds = subsample(xds, levels, vars_of_interest)

    if "nested" in model_type:
        assert lam_index is not None
        if "lam" in model_type:
            xds = xds.isel(cell=slice(lam_index))
            xds = xds.load()

        else:
            xds = xds.load()

    if trim_edge is not None and "lam" in model_type:
        for key in ["x", "y"]:
            if key in ids:
                xds[key] = ids[key] if "variable" not in ids[key].dims else ids[key].isel(variable=0, drop=True)
                xds = xds.set_coords(key)
            else:
                xds[key] = _extra_coords[key]
                xds = xds.set_coords(key)
        xds = trim_xarray_edge(xds, trim_edge)
    return xds


def open_forecast_zarr_dataset(path, t0, levels=None, vars_of_interest=None, trim_edge=None):
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
    # this is
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
    xds = xds.load()
    if trim_edge is not None:
        xds = trim_xarray_edge(xds, trim_edge)
    return xds


def subsample(xds, levels=None, vars_of_interest=None):
    """Subsample vertical levels and variables
    """

    if levels is not None:
        xds = xds.sel(level=levels)

    if vars_of_interest is not None:
        xds = xds[vars_of_interest]
    else:
        xds = drop_forcing_vars(xds)

    return xds
