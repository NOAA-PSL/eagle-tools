from typing import Sequence, Any
import logging
import importlib.resources
import yaml

import numpy as np
import xarray as xr
import pandas as pd

import anemoi.datasets

from ufs2arco.utils import expand_anemoi_dataset, convert_anemoi_inference_dataset

from eagle.tools.nested import regrid_nested_to_latlon
from eagle.tools.reshape import flatten_to_cell
from eagle.tools.reshape import reshape_cell_dim

logger = logging.getLogger("eagle.tools")

def _get_xy(xds: xr.Dataset, n_x: int, n_y: int) -> xr.Dataset:
    """
    Generates x and y coordinates for a LAM dataset.

    Args:
        xds (xr.Dataset): The input dataset.
        n_x (int): The untrimmed length of the dataset in the x direction.
        n_y (int): The untrimmed length of the dataset in the y direction.

    Returns:
        xr.Dataset: The dataset with 'x' and 'y' added.
    """
    x = np.arange(n_x)
    y = np.arange(n_y)
    cell = np.arange(n_x * n_y)

    if "cell" in xds.dims:
        # assume in this case we want the expanded and flattened version
        xds["x"] = xr.DataArray(np.tile(x, n_y), coords={"cell": cell})
        xds["y"] = xr.DataArray(np.tile(y, (n_x, 1)).T.flatten(), coords={"cell": cell})
    else:
        xds["x"] = xr.DataArray(x, coords={"x": x})
        xds["y"] = xr.DataArray(y, coords={"y": y})
    return xds

def trim_xarray_edge(
    xds: xr.Dataset,
    lcc_info: dict,
    trim_edge: Sequence[int],
    stack_order: Sequence[str] | None = None,
) -> xr.Dataset:
    """
    Trim the boundary of a Limited Area Model (LAM) dataset using xarray.

    Args:
        xds (xr.Dataset): The input dataset to trim.
        lcc_info (dict): Dictionary containing Lambert Conformal Conic (LCC) projection details.
            Must contain entries ``{"n_x": length of LAM dataset in x direction, "n_y": length of LAM dataset in y direction}``
            representing the **post-trimmed** lengths.
        trim_edge (Sequence[int]): A sequence (e.g., [left, right, bottom, top]) defining
            how many grid points to trim from the edges.
        stack_order (Sequenc[str], optional): This would be e.g. ["latitude", "longitude"] or ["y", x"] depending on the original dimension order of your dataset

    Returns:
        xr.Dataset: The trimmed dataset with updated coordinates.
    """
    if stack_order is None:
        logger.warning(f"eagle.tools.data.trim_xarray_edge: stack_order not provided for trimming, I'm assuming ['y', 'x'] not ['x', 'y']")
        stack_order  = ["y", "x"]

    idx = stack_order.index("x")*2
    idy = stack_order.index("y")*2
    if not {"x", "y"}.issubset(xds.dims):
        # Note: assuming _get_xy is meant here (was get_xy in original snippet)
        xtrim = trim_edge[0]
        xds = _get_xy(
            xds=xds,
            n_x=lcc_info["n_x"] + trim_edge[idx] + trim_edge[idx+1] ,
            n_y=lcc_info["n_y"] + trim_edge[idy] + trim_edge[idy+1],
        )

    condx = ( (xds["x"] > trim_edge[idx]-1) & (xds["x"] < xds["x"].max().values-trim_edge[idx+1]+1) ).compute()
    condy = ( (xds["y"] > trim_edge[idy]-1) & (xds["y"] < xds["y"].max().values-trim_edge[idy+1]+1) ).compute()
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
) -> xr.Dataset:
    """
    Wrapper for ``anemoi.datasets.open_dataset`` that applies immediate subsampling and processing.

    This function opens a dataset and immediately processes it based on the provided
    subsampling arguments. It provides similar functionality to ``open_anemoi_dataset_with_xarray``
    but retains specific ``anemoi.datasets`` features, such as the ability to open nested datasets.

    .. warning::
        This function loads the resulting dataset into memory. You must use the subsampling
        keyword arguments (e.g., ``t0``, ``tf``, ``levels``) to trim the dataset size
        before loading to avoid OutOfMemory errors.

    Args:
        *args: Passed directly to ``anemoi.datasets.open_dataset()``.
        model_type (str): The specific model configuration. Options include: ``"global"``, ``"lam"``, ``"nested-lam"``, or ``"nested-global"``.
        t0 (str): The starting date/timestamp for selection (inclusive).
        tf (str): The ending date/timestamp for selection (inclusive).
        levels (Sequence[float | int], optional): Specific vertical levels to select from the dataset.
        vars_of_interest (Sequence[str], optional): A list of specific variable or parameter names to keep.
        rename_to_longnames (bool, optional): If True, renames variables to their descriptive long names. Defaults to False.
        reshape_cell_to_2d (bool, optional): If True, reshapes unstructured grid cells into a 2D grid. Defaults to False.
        member (int, optional): The specific ensemble member ID to select.
        lcc_info (dict, optional): Dictionary containing Lambert Conformal Conic (LCC) projection details.
            Must contain entries ``{"n_x": length of LAM dataset in x direction, "n_y": length of LAM dataset in y direction}``.
            Note these lengths are after any trimming.
        **kwargs: Passed directly to ``anemoi.datasets.open_dataset()``.

    Returns:
        xr.Dataset: The subsampled and processed dataset.
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
    Opens an Anemoi dataset using xarray directly (lazy loading).

    Note that the result of this and ``open_anemoi_dataset`` are the same,
    except that this does not load the data into memory immediately.

    Args:
        path (str): Path to the Zarr store.
        model_type (str): The specific model configuration. Options include: ``"global"``, ``"lam"``, ``"nested-lam"``, or ``"nested-global"``.
        levels (Sequence[float | int], optional): Vertical levels to select.
        vars_of_interest (Sequence[str], optional): specific variables to keep.
        trim_edge (Sequence[int], optional): Edge trimming parameters for LAM datasets.
        rename_to_longnames (bool, optional): Rename vars to long names. Defaults to False.
        reshape_cell_to_2d (bool, optional): Reshape cells to 2D grid. Defaults to False.
        member (int, optional): Specific ensemble member to select.
        lcc_info (dict, optional): Dictionary containing Lambert Conformal Conic (LCC) projection details.
            Must contain entries ``{"n_x": length of LAM dataset in x direction, "n_y": length of LAM dataset in y direction}``.
            Note these lengths are after any trimming.

    Returns:
        xr.Dataset: The lazily loaded dataset.
    """

    ads = xr.open_zarr(path)
    stack_order = ads.attrs.get("stack_order", None)
    xds = expand_anemoi_dataset(ads, "data", ads.attrs["variables"])

    xds = xds.rename({"ensemble": "member"})
    xds = subsample(xds, levels, vars_of_interest, member=member)
    if trim_edge is not None and "lam" in model_type:
        xds = trim_xarray_edge(xds, lcc_info, trim_edge, stack_order)

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
    horizontal_regrid_kwargs: dict | None = None,
    stack_order: Sequence[str] | None = None,
) -> xr.Dataset:
    """
    Opens an inference dataset from Anemoi.

    Note that the result from anemoi inference has often been trimmed already,
    as far as the LAM is concerned. If ``trim_edge`` is provided, this will trim
    the result further.

    Note:
        When ``model_type = "nested-global"``, the dataset is regridded to a common global resolution.

    Args:
        path (str): Path to the dataset.
        model_type (str): "nested-lam", "nested-global", or "global".
        lam_index (int, optional): Index for LAM selection in nested models.
        levels (Sequence[float | int], optional): Vertical levels to select.
        vars_of_interest (Sequence[str], optional): specific variables to keep.
        trim_edge (Sequence[int], optional): Additional edge trimming.
        rename_to_longnames (bool, optional): Rename vars to long names. Defaults to False.
        load (bool, optional): If True, load data into memory. Defaults to False.
        reshape_cell_to_2d (bool, optional): Reshape cells to 2D grid. Defaults to False.
        lcc_info (dict, optional): Dictionary containing Lambert Conformal Conic (LCC) projection details.
            Must contain entries ``{"n_x": length of LAM dataset in x direction, "n_y": length of LAM dataset in y direction}``.
            Note these lengths are after any trimming.
        member (int, optional): Specific ensemble member to select.
        horizontal_regrid_kwargs (dict, optional): only used if ``model_type = "nested-global"``
            in order to regrid to common global resolution.
            Options are passed directly to ``ufs2arco.transforms.horizontal_regrid.horizontal_regrid()``
        stack_order (Sequenc[str], optional): Only used if trimming further. This would be e.g. ["latitude", "longitude"] or ["y", x"] depending on the original dimension order of your dataset

    Returns:
        xr.Dataset: The inference dataset.
    """

    assert model_type in ("nested", "nested-lam", "nested-global", "global")

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

    if "nested" in model_type:
        assert lam_index is not None
        if "lam" in model_type:
            xds = xds.isel(cell=slice(lam_index))
        elif "global" in model_type:
            xds = regrid_nested_to_latlon(
                xds,
                lam_index=lam_index,
                lcc_info=lcc_info,
                horizontal_regrid_kwargs=horizontal_regrid_kwargs,
            )

            if not reshape_cell_to_2d:
                xds = flatten_to_cell(xds)

    if load:
        xds = xds.load()

    if trim_edge is not None and "lam" in model_type:
        xds = trim_xarray_edge(xds, lcc_info, trim_edge, stack_order)

    if rename_to_longnames:
        xds = rename(xds)

    if reshape_cell_to_2d and model_type != "nested-global":
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
    """
    Opens non-anemoi forecast datasets (e.g., HRRR forecast data preprocessed by ufs2arco).

    Args:
        path (str): Path to the Zarr dataset.
        t0 (pd.Timestamp): The initialization time to select.
        levels (Sequence[float | int], optional): Vertical levels to select.
        vars_of_interest (Sequence[str], optional): specific variables to keep.
        trim_edge (Sequence[int], optional): Edge trimming parameters.
        rename_to_longnames (bool, optional): Rename vars to long names. Defaults to False.
        load (bool, optional): If True, load data into memory. Defaults to False.
        reshape_cell_to_2d (bool, optional): Reshape cells to 2D grid. Defaults to False.
        member (int, optional): Specific ensemble member.
        lcc_info (dict, optional): Dictionary containing Lambert Conformal Conic (LCC) projection details.
            Must contain entries ``{"n_x": length of LAM dataset in x direction, "n_y": length of LAM dataset in y direction}``.
            Note these lengths are after any trimming.

    Returns:
        xr.Dataset: The forecast dataset.
    """

    xds = xr.open_zarr(path, decode_timedelta=True)
    stack_order = list(x for x in xds.dims if x not in ["t0", "fhr", "level", "member"])
    xds = xds.sel(t0=t0).squeeze(drop=True)
    xds["time"] = xr.DataArray(
        [pd.Timestamp(t0) + pd.Timedelta(hours=fhr) for fhr in xds.fhr.values],
        coords=xds.fhr.coords,
    )
    xds = xds.swap_dims({"fhr": "time"}).drop_vars("fhr")
    xds = subsample(xds, levels, vars_of_interest, member=member)

    # Comparing to anemoi, it's sometimes easier to flatten than unpack anemoi
    if not reshape_cell_to_2d:
        xds = flatten_to_cell(xds)

    xds = xds.drop_vars(["t0", "valid_time"])

    if load:
        xds = xds.load()

    if trim_edge is not None:
        xds = trim_xarray_edge(xds, lcc_info, trim_edge, stack_order)

    if rename_to_longnames:
        xds = rename(xds)
    return xds

def subsample(
    xds: xr.Dataset,
    levels: Sequence[float | int] = None,
    vars_of_interest: Sequence[str] = None,
    member: int | None = None
) -> xr.Dataset:
    """
    Subsample vertical levels, ensemble member(s), and variables.

    Also calculates wind speed derived variables if requested in ``vars_of_interest``.

    Args:
        xds (xr.Dataset): Input dataset.
        levels (Sequence[float | int], optional): Vertical levels to keep.
        vars_of_interest (Sequence[str], optional): Variables to keep. If None, only forces drop of forcing vars.
        member (int, optional): Ensemble member to keep.

    Returns:
        xr.Dataset: The subsampled dataset.
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

def drop_forcing_vars(xds: xr.Dataset) -> xr.Dataset:
    """
    Drops standard forcing variables (e.g., Julian day, solar angles, orography) from the dataset.

    Args:
        xds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Dataset with forcing variables removed.
    """
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

def _wind_speed(u: xr.DataArray, v: xr.DataArray, long_name: str) -> xr.DataArray:
    """
    Calculates wind speed magnitude from U and V components.
    """
    return xr.DataArray(
        np.sqrt(u**2 + v**2),
        coords=u.coords,
        attrs={
            "long_name": long_name,
            "units": "m/s",
        },
    )

def calc_wind_speed(xds: xr.Dataset, vars_of_interest: Sequence[str]) -> xr.Dataset:
    """
    Calculates and adds wind speed variables (10m, 80m, 100m, or bulk) if requested.

    Args:
        xds (xr.Dataset): Input dataset containing U/V components.
        vars_of_interest (Sequence[str]): List of variables desired (checks for specific wind speed keys).

    Returns:
        xr.Dataset: Dataset with calculated wind speed variables added.
    """

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

def rename(xds: xr.Dataset) -> xr.Dataset:
    """
    Renames dataset variables based on an external YAML configuration file.

    Args:
        xds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Dataset with renamed variables.
    """
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rdict = yaml.safe_load(f)

    for key, val in rdict.items():
        if key in xds:
            xds = xds.rename({key: val})
    return xds
