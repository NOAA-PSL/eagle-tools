import numpy as np
import pytest
import xarray as xr

from eagle.tools.reshape import (
    flatten_to_cell,
    reshape_cell_dim,
    reshape_cell_to_latlon,
    reshape_cell_to_xy,
)

from synthetic import LAT, LON, N_CELL, make_cell_ds


def xy_ds(n_x=4, n_y=3):
    data = np.arange(2 * n_y * n_x, dtype=float).reshape(2, n_y, n_x)
    return xr.Dataset(
        {"t2m": (("time", "y", "x"), data)},
        coords={"time": [0, 1], "x": np.arange(n_x), "y": np.arange(n_y)},
    )


def test_flatten_xy_is_row_major():
    xds = xy_ds()

    result = flatten_to_cell(xds)

    assert result["t2m"].dims == ("time", "cell")
    np.testing.assert_array_equal(result["cell"], np.arange(12))
    np.testing.assert_array_equal(result["t2m"], xds["t2m"].values.reshape(2, 12))
    assert "x" not in result and "y" not in result


def test_flatten_and_reshape_latlon_round_trip():
    xds = make_cell_ds(n_time=2, varnames=("t2m", "sp"))
    xds["t2m"].attrs = {"units": "K"}

    gridded = reshape_cell_to_latlon(xds)

    assert gridded["t2m"].dims == ("time", "latitude", "longitude")
    np.testing.assert_array_equal(gridded["latitude"], LAT)  # stays descending
    np.testing.assert_array_equal(gridded["longitude"], LON)
    assert gridded["t2m"].attrs == {"units": "K"}

    flat = flatten_to_cell(gridded)
    np.testing.assert_array_equal(flat["t2m"], xds["t2m"])


def test_reshape_to_latlon_with_ascending_latitude():
    xds = make_cell_ds(n_time=1).isel(cell=slice(None, None, -1))

    gridded = reshape_cell_to_latlon(xds)

    np.testing.assert_array_equal(gridded["latitude"], np.sort(LAT))


def test_flatten_raises_on_unknown_dims():
    xds = xr.Dataset({"t2m": (("time", "z"), np.zeros((2, 3)))})

    with pytest.raises(KeyError, match="Unclear on the dimensions"):
        flatten_to_cell(xds)


def test_reshape_to_xy_reshapes_cell_coords_too():
    xds = flatten_to_cell(xy_ds()).assign_coords(lat2d=("cell", np.arange(12.0)))

    result = reshape_cell_to_xy(xds, n_x=4, n_y=3)

    assert result["t2m"].dims == ("time", "y", "x")
    assert result["lat2d"].dims == ("y", "x")
    assert "lat2d" in result.coords
    np.testing.assert_array_equal(result["t2m"], xy_ds()["t2m"])


def test_reshape_to_xy_copies_variables_without_cell():
    xds = flatten_to_cell(xy_ds())
    xds["scale"] = ("time", [1.0, 2.0])

    result = reshape_cell_to_xy(xds, n_x=4, n_y=3)

    xr.testing.assert_equal(result["scale"], xds["scale"])


@pytest.mark.parametrize(
    ("model_type", "lcc_info", "dims"),
    [
        ("global", None, ("time", "latitude", "longitude")),
        ("nested-global", None, ("time", "latitude", "longitude")),
        ("lam", {"n_x": 4, "n_y": 4}, ("time", "y", "x")),
        ("nested-lam", {"n_x": 4, "n_y": 4}, ("time", "y", "x")),
    ],
)
def test_reshape_cell_dim(model_type, lcc_info, dims):
    result = reshape_cell_dim(make_cell_ds(), model_type, lcc_info)

    assert result["t2m"].dims == dims


def test_reshape_cell_dim_lam_requires_lcc_info():
    with pytest.raises(AssertionError, match="Need lcc_info"):
        reshape_cell_dim(make_cell_ds(), "lam", lcc_info=None)


@pytest.mark.parametrize(
    ("model_type", "lcc_info", "message"),
    [
        ("global", None, "(latitude, longitude)"),
        ("lam", {"n_x": 5, "n_y": 5}, "(y, x)"),
    ],
)
def test_reshape_cell_dim_warns_and_skips_on_bad_shape(model_type, lcc_info, message, caplog):
    xds = make_cell_ds().isel(cell=slice(0, N_CELL - 1))  # no longer a full grid

    result = reshape_cell_dim(xds, model_type, lcc_info)

    assert result is xds
    assert f"could not reshape cell -> {message}" in caplog.text
