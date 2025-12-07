import logging

import numpy as np
import xarray as xr

logger = logging.getLogger("eagle.tools")

def flatten_to_cell(xds: xr.Dataset):

    if {"x", "y"}.issubset(xds.dims):
        xds = xds.stack(cell2d=("y", "x"))

    elif {"longitude", "latitude"}.issubset(xds.dims):
        xds = xds.stack(cell2d=("latitude", "longitude"))

    else:
        raise KeyError("Unclear on the dimensions here")

    xds["cell"] = xr.DataArray(
        np.arange(len(xds["cell2d"])),
        coords=xds["cell2d"].coords,
    )
    xds = xds.swap_dims({"cell2d": "cell"})
    for key in ["x", "y", "cell2d"]:
        if key in xds:
            xds = xds.drop_vars(key)
    return xds


def reshape_cell_dim(xds: xr.Dataset, model_type: str, lcc_info: dict = None) -> xr.Dataset:
    """
    Reshapes the 'cell' dimension into standard 2D spatial dimensions.

    Args:
        xds (xr.Dataset): Input dataset with a 'cell' dimension.
        model_type (str): "global" (maps to lat/lon) or "lam" (maps to y/x).
        lcc_info (dict, optional): Dictionary containing Lambert Conformal Conic (LCC) projection details.
            Must contain entries ``{"n_x": length of LAM dataset in x direction, "n_y": length of LAM dataset in y direction}``.
            Required if model_type contains "lam".
            Note these lengths are after any trimming.

    Returns:
        xr.Dataset: Reshaped dataset.
    """
    if "global" in model_type:
        try:
            xds = reshape_cell_to_latlon(xds)
        except:
            logger.warning("reshape_cell_to_2d: could not reshape cell -> (latitude, longitude), skipping...")

    elif "lam" in model_type:
        assert isinstance(lcc_info, dict), "Need lcc_info={'n_x': ..., 'n_y': ...} for LAM model type"
        try:
            xds = reshape_cell_to_xy(xds, **lcc_info)
        except:
            logger.warning("reshape_cell_to_2d: could not reshape cell -> (y, x), skipping...")
    return xds

def reshape_cell_to_latlon(xds: xr.Dataset) -> xr.Dataset:
    """
    Reshapes a dataset with a 'cell' dimension into 'latitude' and 'longitude' dimensions.

    Args:
        xds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Reshaped dataset.
    """

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

def reshape_cell_to_xy(xds: xr.Dataset, n_x: int, n_y: int) -> xr.Dataset:
    """
    Reshapes a dataset with a 'cell' dimension into 'y' and 'x' dimensions.

    Args:
        xds (xr.Dataset): Input dataset.
        n_x (int): The final length of the data in the x direction (after any trimming).
        n_y (int): The final length of the data in the y direction (after any trimming).

    Returns:
        xr.Dataset: Reshaped dataset.
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
