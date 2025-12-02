from typing import Sequence, Any
import logging
from collections.abc import Sequence
import importlib.resources
import yaml

import numpy as np
import xarray as xr
import pandas as pd

import anemoi.datasets

from ufs2arco.utils import expand_anemoi_dataset, convert_anemoi_inference_dataset

logger = logging.getLogger("eagle.tools")


def get_xy(xds, n_x, n_y):
    """Here n_x, n_y are the untrimmed lengths"""
    x = np.arange(n_x)
    y = np.arange(n_y)
    cell = np.arange(n_x * n_y)
    xydict = {
        "x": xr.DataArray(x, coords={"x": x}),
        "y": xr.DataArray(y, coords={"y": y}),
        "xcell": xr.DataArray(np.tile(x, n_y), coords={"cell": cell}),
        "ycell": xr.DataArray(np.tile(y, (n_x, 1)).T.flatten(), coords={"cell": cell}),
    }
    if "cell" in xds.dims:
        # assume in this case we want the expanded and flattened version
        xds["x"] = xr.DataArray(np.tile(x, n_y), coords={"cell": cell})
        xds["y"] = xr.DataArray(np.tile(y, (n_x, 1)).T.flatten(), coords={"cell": cell})
    else:
        xds["x"] = xr.DataArray(x, coords={"x": x})
        xds["y"] = xr.DataArray(y, coords={"y": y})
    return xds



def trim_xarray_edge(xds, lcc_info, trim_edge):
    """lcc_info has n_x and n_y, which are the post-trimmed legnths"""

    if not {"x", "y"}.issubset(xds.dims):
        xds = get_xy(
            xds=xds,
            n_x=lcc_info["n_x"] + trim_edge[0] + trim_edge[1],
            n_y=lcc_info["n_y"] + trim_edge[2] + trim_edge[3],
        )

    condx = ( (xds["x"] > trim_edge[0]-1) & (xds["x"] < xds["x"].max().values-trim_edge[1]+1) ).compute()
    condy = ( (xds["y"] > trim_edge[2]-1) & (xds["y"] < xds["y"].max().values-trim_edge[3]+1) ).compute()
    xds = xds.where(condx & condy, drop=True)

    # reset either the cell or x & t coordinate values to be 0->len(coord)
    # incoming datasets are either flattened and so have cell as the underlying coordinate
    # or are already 2D, which is only the case for zarr forecast data
    # (i.e., from grib archives -> zarr via ufs2arco)
    if "cell" in xds.dims:
        xds["cell"] = xr.DataArray(
            np.arange(len(xds.cell)),
            dims="cell",
        )
        for key in ["x", "y"]:
            if key in xds:
                xds = xds.drop_vars(key)
    else:
        xds["x"] = xr.DataArray(
            np.arange(len(xds.x)),
            dims="x",
        )
        xds["y"] = xr.DataArray(
            np.arange(len(xds.y)),
            dims="y",
        )
        if "cell" in xds:
            xds = xds.drop_vars("cell")
    return xds


def open_anemoi_dataset(
    *args: Any,
    model_type: str,
    t0: str,
    tf: str,
    levels: Sequence[float | int] | None = None,
    vars_of_interest: Sequence[str] | None = None,
    rename_to_longnames: bool = False,
    reshape_cell_to_2d: bool = False,
    member: int | None = None,
    lcc_info: dict | None = None,
    **kwargs: Any,
) -> Any:
    """
    Wrapper for anemoi.datasets.open_dataset that applies immediate subsampling and processing.

    Note:
        This will bring the resulting dataset into memory, so use the subsampling keyword arguments to trim down the dataset. Overall this does the same thing as open_anemoi_dataset_with_xarray, except the latter does not have all of the anemoi.datasets functionality, for instance it cannot open a nested dataset.

    Parameters
    ----------
    *args, **kwargs :
        Passed directly to anemoi.datasets.open_dataset().
    model_type: str
        "global", "nested-lam", or "nested-global" for now
    t0 : str
        Starting date to select
    tf : str
        End date to select
    levels : Sequence[float | int], optional
        Select specific vertical levels.
    vars_of_interest : Sequence[str], optional
        Select specific variables/parameters.
    rename_to_longnames : bool, default False
        Renames variables to their descriptive long names.
    reshape_cell_to_2d : bool, default False
        Reshapes unstructured grid cells to 2D lat/lon.
    member : int, optional
        Selects a specific ensemble member.
    lcc_info : dict, optional
        which contains n_x and n_y, the length of the x/y dimensions after any trimming is done


    Returns
    -------
    dataset : The processed dataset object.
    """

    ads = anemoi.datasets.open_dataset(*args, **kwargs)

    # Note that we can't use the "start/end" kwargs with anemoi.datasets.open_dataset
    # because they do not work for opening a nested dataset
    start = ads.to_index(date=t0, variable=0)[0]
    end = ads.to_index(date=tf, variable=0)[0] + 1

    # Since we'll bring the array into memory, we "pre-"subsample the member dim here
    # We use a different variable here so that the subsample function used later
    # is called consistently as with other open_dataset functions
    amember = slice(None, None) if member is None else member
    if isinstance(member, int):
        amember = slice(member, member+1)

    # This next line brings the subsampled array into memory
    data = ads[start:end, :, amember, :]

    # Now we convert it to xarray to work with the rest of this package
    xda = xr.DataArray(
        data,
        coords={
            "time": np.arange(end-start),
            "variable": np.arange(ads.shape[1]),
            "ensemble": np.arange(ads.shape[2]),
            "cell": np.arange(ads.shape[3]),
        },
        dims=("time", "variable", "ensemble", "cell"),
    )
    xds = xda.to_dataset(name="data")
    xds["latitudes"] = xr.DataArray(ads.latitudes, coords=xds["cell"].coords)
    xds["longitudes"] = xr.DataArray(ads.longitudes, coords=xds["cell"].coords)
    xds["dates"] = xr.DataArray(ads.dates[start:end], dims="time")
    xds = xds.set_coords(["latitudes", "longitudes", "dates"])
    xds = expand_anemoi_dataset(xds, "data", ads.variables)

    xds = xds.rename({"ensemble": "member"})

    xds = subsample(xds, levels, vars_of_interest, member=member)
    if rename_to_longnames:
        xds = rename(xds)

    if reshape_cell_to_2d:
        xds = reshape_cell_dim(xds, model_type, lcc_info)

    return xds


def open_anemoi_dataset_with_xarray(
    path: str,
    model_type: str,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    trim_edge: Sequence[int] = None,
    rename_to_longnames: bool = False,
    reshape_cell_to_2d: bool = False,
    member: int | None = None,
    lcc_info: dict | None = None,
) -> xr.Dataset:
    """
    Note that the result of this and `open_anemoi_dataset` are the same,
    except that this does not load the data into memory.
    """

    ads = xr.open_zarr(path)
    xds = expand_anemoi_dataset(ads, "data", ads.attrs["variables"])

    xds = xds.rename({"ensemble": "member"})
    xds = subsample(xds, levels, vars_of_interest, member=member)
    if trim_edge is not None and "lam" in model_type:
        xds = trim_xarray_edge(xds, lcc_info, trim_edge)

    if rename_to_longnames:
        xds = rename(xds)

    if reshape_cell_to_2d:
        xds = reshape_cell_dim(xds, model_type, lcc_info)

    return xds


def open_anemoi_inference_dataset(
    path: str,
    model_type: str,
    lam_index: int | None = None,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    trim_edge: Sequence[int] = None,
    rename_to_longnames: bool = False,
    load: bool = False,
    reshape_cell_to_2d: bool = False,
    lcc_info: dict | None = None,
    member: int | None = None,
) -> xr.Dataset:
    """Note that the result from anemoi inference has been trimmed, as far as the LAM is concerned.
    So if trim_edge is set to True, this will trim the result even more.
    """

    assert model_type in ("nested-lam", "nested-global", "global")

    ids = xr.open_dataset(path, chunks="auto")
    xds = convert_anemoi_inference_dataset(ids)
    # TODO: add this next line to ufs2arco, if keeping the convert function in that repo
    xds["cell"] = xr.DataArray(
        np.arange(len(xds.cell)),
        dims=("cell",),
    )
    xds = subsample(xds, levels, vars_of_interest, member=member)
    if "ensemble" in xds.dims:
        raise NotImplementedError(f"note to future self from eagle.tools.data: open_anemoi_dataset_with_xarray renames ensemble-> member, need to do this here")

    if model_type == "nested-lam":
        assert lam_index is not None
        if "lam" in model_type:
            xds = xds.isel(cell=slice(lam_index))

    if load:
        xds = xds.load()

    if trim_edge is not None and "lam" in model_type:
        xds = trim_xarray_edge(xds, lcc_info, trim_edge)

    if rename_to_longnames:
        xds = rename(xds)

    if reshape_cell_to_2d:
        xds = reshape_cell_dim(xds, model_type, lcc_info)

    return xds


def open_forecast_zarr_dataset(
    path: str,
    t0: pd.Timestamp,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    trim_edge: Sequence[int] = None,
    rename_to_longnames: bool = False,
    load: bool = False,
    reshape_cell_to_2d: bool = False,
    member: int | None = None,
    lcc_info: dict | None = None,
) -> xr.Dataset:
    """This is for non-anemoi forecast datasets, for example HRRR forecast data preprocessed by ufs2arco"""

    xds = xr.open_zarr(path, decode_timedelta=True)
    xds = xds.sel(t0=t0).squeeze(drop=True)
    xds["time"] = xr.DataArray(
        [pd.Timestamp(t0) + pd.Timedelta(hours=fhr) for fhr in xds.fhr.values],
        coords=xds.fhr.coords,
    )
    xds = xds.swap_dims({"fhr": "time"}).drop_vars("fhr")
    xds = subsample(xds, levels, vars_of_interest, member=member)

    # Comparing to anemoi, it's sometimes easier to flatten than unpack anemoi
    if not reshape_cell_to_2d:
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
        for key in ["x", "y", "cell2d"]:
            if key in xds:
                xds = xds.drop_vars(key)
    xds = xds.drop_vars(["t0", "valid_time"])

    if load:
        xds = xds.load()

    if trim_edge is not None:
        xds = trim_xarray_edge(xds, lcc_info, trim_edge)

    if rename_to_longnames:
        xds = rename(xds)
    return xds


def subsample(xds, levels=None, vars_of_interest=None, member=None):
    """Subsample vertical levels, ensemble member(s), and variables
    """

    if levels is not None:
        xds = xds.sel(level=levels)

    if member is not None:
        xds = xds.sel(member=member)

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

def reshape_cell_dim(xds, model_type, lcc_info=None):
    if "global" in model_type:
        try:
            xds = reshape_cell_to_latlon(xds)
        except:
            logger.warning("reshape_cell_to_2d: could not reshape cell -> (latitude, longitude), skipping...")

    elif "lam" in model_type:
        assert isinstance(lcc_info, dict), "Need lcc_info={'n_x': ..., 'n_y': ...} for LAM model type"
        xds = reshape_cell_to_xy(xds, **lcc_info)
        #except:
        #    logger.warning("reshape_cell_to_2d: could not reshape cell -> (y, x), skipping...")
    return xds

def reshape_cell_to_latlon(xds):

    lon = np.unique(xds["longitude"])
    lat = np.unique(xds["latitude"])
    if xds["latitude"][0] > xds["latitude"][-1]:
        lat = lat[::-1]

    nds = xr.Dataset()
    nds["longitude"] = xr.DataArray(
        lon,
        coords={"longitude": lon},
    )
    nds["latitude"] = xr.DataArray(
        lat,
        coords={"latitude": lat},
    )
    for key in xds.dims:
        if key != "cell":
            nds[key] = xds[key].copy()

    for key in xds.data_vars:
        dims = tuple(d for d in xds[key].dims if d != "cell")
        dims += ("latitude", "longitude")
        shape = tuple(len(nds[d]) for d in dims)
        nds[key] = xr.DataArray(
            xds[key].data.reshape(shape),
            dims=dims,
            attrs=xds[key].attrs.copy(),
        )
    return nds

def reshape_cell_to_xy(xds, n_x, n_y):
    """n_x and n_y are the lengths after trimming, the final lengths of the data
    """
    x = np.arange(n_x)
    y = np.arange(n_y)

    nds = xr.Dataset()
    nds["x"] = xr.DataArray(
        x,
        coords={"x": x},
    )
    nds["y"] = xr.DataArray(
        y,
        coords={"y": y},
    )
    for key in xds.dims:
        if key != "cell":
            nds[key] = xds[key].copy()

    coords = [x for x in list(xds.coords) if x not in xds.dims]
    for key in list(xds.data_vars) + coords:
        if "cell" in xds[key].dims:
            dims = tuple(d for d in xds[key].dims if d != "cell")
            dims += ("y", "x")
            shape = tuple(len(nds[d]) for d in dims)
            nds[key] = xr.DataArray(
                xds[key].data.reshape(shape),
                dims=dims,
                attrs=xds[key].attrs.copy(),
            )
        else:
            nds[key] = xds[key].copy()

    nds = nds.set_coords(coords)
    return nds
