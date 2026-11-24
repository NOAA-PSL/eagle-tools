from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eagle.tools import data
from eagle.tools.reshape import reshape_cell_to_latlon

from synthetic import CELL_LAT, DATES, LAT, N_CELL, VARIABLES, make_cell_ds, truth, truth_at

# --- _get_xy / trim_xarray_edge ---------------------------------------------------


def test_get_xy_on_flattened_data_is_row_major():
    xds = xr.Dataset({"t2m": ("cell", np.zeros(6))})

    result = data._get_xy(xds, n_x=3, n_y=2)

    np.testing.assert_array_equal(result["x"], [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(result["y"], [0, 0, 0, 1, 1, 1])


def test_get_xy_without_cell_adds_1d_coords():
    result = data._get_xy(xr.Dataset(), n_x=3, n_y=2)

    assert result["x"].dims == ("x",) and len(result["x"]) == 3
    assert result["y"].dims == ("y",) and len(result["y"]) == 2


# Trim 1 point from the start of y, 2 from the end of y, and 1 from the start of x.
# With stack_order ["y", "x"], trim_edge is ordered [y_start, y_end, x_start, x_end].
TRIM_EDGE = [1, 2, 1, 0]
N_Y, N_X = 6, 5
TRIMMED = {"n_y": N_Y - 3, "n_x": N_X - 1}


def test_trim_2d_dataset():
    values = np.arange(N_Y * N_X).reshape(N_Y, N_X)
    xds = xr.Dataset(
        {"t2m": (("y", "x"), values)},
        coords={"y": np.arange(N_Y), "x": np.arange(N_X)},
    )

    result = data.trim_xarray_edge(xds, TRIMMED, TRIM_EDGE, stack_order=["y", "x"])

    np.testing.assert_array_equal(result["t2m"], values[1:-2, 1:])
    np.testing.assert_array_equal(result["x"], np.arange(TRIMMED["n_x"]))
    np.testing.assert_array_equal(result["y"], np.arange(TRIMMED["n_y"]))


def test_trim_flattened_dataset():
    values = np.arange(N_Y * N_X)
    xds = xr.Dataset({"t2m": ("cell", values)})

    result = data.trim_xarray_edge(xds, TRIMMED, TRIM_EDGE, stack_order=["y", "x"])

    expected = values.reshape(N_Y, N_X)[1:-2, 1:].flatten()
    np.testing.assert_array_equal(result["t2m"], expected)
    np.testing.assert_array_equal(result["cell"], np.arange(expected.size))
    assert "x" not in result and "y" not in result


def test_trim_warns_and_assumes_y_x_without_stack_order(caplog):
    xds = xr.Dataset({"t2m": ("cell", np.arange(N_Y * N_X))})

    result = data.trim_xarray_edge(xds, TRIMMED, TRIM_EDGE)

    assert "stack_order not provided" in caplog.text
    assert result.sizes["cell"] == TRIMMED["n_y"] * TRIMMED["n_x"]


def test_trim_2d_dataset_drops_stale_cell():
    xds = xr.Dataset(
        {"t2m": (("y", "x"), np.zeros((N_Y, N_X)))},
        coords={"y": np.arange(N_Y), "x": np.arange(N_X), "cell": 0},
    )

    result = data.trim_xarray_edge(xds, TRIMMED, TRIM_EDGE, stack_order=["y", "x"])

    assert "cell" not in result


# --- subsample / drop_forcing_vars / wind speed / rename -----------------------------


def test_drop_forcing_vars():
    xds = make_cell_ds(varnames=("t2m", "cos_julian_day", "lsm", "insolation"))

    assert list(data.drop_forcing_vars(xds)) == ["t2m"]


def test_subsample_levels_member_and_vars():
    xds = make_cell_ds(member=3, varnames=("t", "q")).expand_dims(level=[250, 500, 850], axis=2)

    result = data.subsample(xds, levels=[500], vars_of_interest=["t"], member=1)

    assert list(result) == ["t"]
    assert result["t"].dims == ("time", "level", "cell")
    xr.testing.assert_equal(result["t"], xds["t"].sel(level=[500], member=1))


def test_subsample_without_vars_drops_forcing():
    xds = make_cell_ds(varnames=("t2m", "cos_julian_day"))

    assert list(data.subsample(xds)) == ["t2m"]


@pytest.mark.parametrize(
    ("speed", "u", "v"),
    [
        ("10m_wind_speed", "ugrd10m", "vgrd10m"),
        ("10m_wind_speed", "u10", "v10"),
        ("10m_wind_speed", "10m_u_component_of_wind", "10m_v_component_of_wind"),
        ("80m_wind_speed", "ugrd80m", "vgrd80m"),
        ("80m_wind_speed", "u80", "v80"),
        ("80m_wind_speed", "80m_u_component_of_wind", "80m_v_component_of_wind"),
        ("100m_wind_speed", "ugrd100m", "vgrd100m"),
        ("100m_wind_speed", "u100", "v100"),
        ("100m_wind_speed", "100m_u_component_of_wind", "100m_v_component_of_wind"),
        ("wind_speed", "ugrd", "vgrd"),
        ("wind_speed", "u", "v"),
        ("wind_speed", "u_component_of_wind", "v_component_of_wind"),
    ],
)
def test_subsample_computes_wind_speed(speed, u, v):
    xds = make_cell_ds(value=3.0, varnames=(u,))
    xds[v] = xds[u] + 1.0  # a 3-4-5 triangle

    result = data.subsample(xds, vars_of_interest=[speed])

    assert list(result) == [speed]
    np.testing.assert_allclose(result[speed], 5.0)
    assert result[speed].attrs["units"] == "m/s"


@pytest.mark.parametrize("speed", ["10m_wind_speed", "80m_wind_speed", "100m_wind_speed", "wind_speed"])
def test_wind_speed_raises_on_unrecognized_components(speed):
    xds = make_cell_ds(varnames=("eastward_wind", "northward_wind"))

    with pytest.raises(KeyError, match=f"could not compute '{speed}'"):
        data.subsample(xds, vars_of_interest=[speed])


def test_wind_speed_does_not_reuse_another_heights_components():
    # Only 10m winds are present, so 80m must fail, not silently reuse the 10m components
    xds = make_cell_ds(varnames=("u10", "v10"))

    with pytest.raises(KeyError, match="could not compute '80m_wind_speed'"):
        data.subsample(xds, vars_of_interest=["10m_wind_speed", "80m_wind_speed"])


def test_rename_uses_long_names():
    xds = make_cell_ds(varnames=("t2m", "t", "not_in_rename_yaml"))

    result = data.rename(xds)

    assert set(result) == {"2m_temperature", "temperature", "not_in_rename_yaml"}


# --- open_anemoi_dataset_with_xarray ---------------------------------------------


def test_open_anemoi_dataset_with_xarray(anemoi_zarr):
    result = data.open_anemoi_dataset_with_xarray(anemoi_zarr, model_type="global", levels=[850])

    assert set(result) == {"t", "t2m"}  # forcing dropped, levels stacked
    assert result["t"].dims == ("time", "member", "level", "cell")
    np.testing.assert_array_equal(result["time"], DATES)
    np.testing.assert_array_equal(result["latitude"], CELL_LAT)
    np.testing.assert_array_equal(result["t"].isel(member=0, level=0), truth(len(DATES))[:, 3])


def test_open_anemoi_dataset_with_xarray_reshape_and_rename(anemoi_zarr):
    result = data.open_anemoi_dataset_with_xarray(
        anemoi_zarr,
        model_type="global",
        vars_of_interest=["t2m"],
        member=0,
        rename_to_longnames=True,
        reshape_cell_to_2d=True,
    )

    assert list(result) == ["2m_temperature"]
    assert result["2m_temperature"].dims == ("time", "latitude", "longitude")


def test_open_anemoi_dataset_with_xarray_trims_lam(anemoi_zarr):
    # Treat the 4 x 4 grid as a LAM and trim the first and last rows
    result = data.open_anemoi_dataset_with_xarray(
        anemoi_zarr,
        model_type="lam",
        vars_of_interest=["t2m"],
        member=0,
        trim_edge=[1, 1, 0, 0],
        lcc_info={"n_x": 4, "n_y": 2},
        reshape_cell_to_2d=True,
    )

    expected = truth(len(DATES))[:, 1].reshape(-1, 4, 4)[:, 1:3, :]
    assert result["t2m"].dims == ("time", "y", "x")
    np.testing.assert_array_equal(result["t2m"], expected)


# --- open_anemoi_dataset ---------------------------------------------------------


class FakeAnemoiDataset:
    """Just enough of anemoi.datasets' interface for open_anemoi_dataset."""

    dates = DATES.values
    variables = VARIABLES
    latitudes = CELL_LAT
    longitudes = np.tile([0.0, 90.0, 180.0, 270.0], 4)

    def __init__(self):
        self._data = truth(len(DATES))[:, :, None, :]
        self.shape = self._data.shape

    def to_index(self, date, variable):
        return (list(DATES).index(pd.Timestamp(date)), variable)

    def __getitem__(self, key):
        return self._data[key]


@pytest.mark.parametrize("member", [None, 0])
def test_open_anemoi_dataset(member):
    with mock.patch("anemoi.datasets.open_dataset", return_value=FakeAnemoiDataset()) as open_dataset:
        result = data.open_anemoi_dataset(
            "some.zarr",
            model_type="global",
            t0=str(DATES[1]),
            tf=str(DATES[3]),
            vars_of_interest=["t2m"],
            member=member,
            rename_to_longnames=True,
            reshape_cell_to_2d=True,
        )

    open_dataset.assert_called_once_with("some.zarr")
    expected = truth(len(DATES))[1:4, 1].reshape(3, 4, 4)
    xda = result["2m_temperature"]
    if member is None:
        xda = xda.squeeze("member")
    assert xda.dims == ("time", "latitude", "longitude")
    np.testing.assert_array_equal(xda, expected)
    np.testing.assert_array_equal(result["time"], DATES[1:4])


# --- open_anemoi_inference_dataset ------------------------------------------------


def test_open_anemoi_inference_dataset_global(write_inference):
    path = write_inference(DATES[0], lead_time=12)

    result = data.open_anemoi_inference_dataset(path, model_type="global", load=True)

    expected = truth_at(DATES[0], 12)
    assert set(result) == {"t", "t2m"}
    assert result["t"].dims == ("time", "level", "cell")
    np.testing.assert_array_equal(result["t2m"], expected["t2m"])
    np.testing.assert_array_equal(result["t"].sel(level=500), expected["t_500"])


def test_open_anemoi_inference_dataset_nested_lam(write_inference):
    path = write_inference(DATES[0])

    result = data.open_anemoi_inference_dataset(
        path,
        model_type="nested-lam",
        lam_index=12,  # the first three rows are the LAM
        vars_of_interest=["t2m"],
        trim_edge=[0, 1, 0, 0],
        stack_order=["y", "x"],
        lcc_info={"n_x": 4, "n_y": 2},
        reshape_cell_to_2d=True,
        rename_to_longnames=True,
    )

    expected = truth_at(DATES[0], 12)["t2m"].values.reshape(-1, 4, 4)[:, :2, :]
    assert result["2m_temperature"].dims == ("time", "y", "x")
    np.testing.assert_array_equal(result["2m_temperature"], expected)


@pytest.mark.parametrize("reshape_cell_to_2d", [True, False])
def test_open_anemoi_inference_dataset_nested_global_regrids(write_inference, reshape_cell_to_2d):
    path = write_inference(DATES[0])
    regrid_kwargs = {"target_grid_path": "grid.nc"}

    with mock.patch.object(data, "regrid_nested_to_latlon", side_effect=lambda xds, **kw: reshape_cell_to_latlon(xds)) as regrid:
        result = data.open_anemoi_inference_dataset(
            path,
            model_type="nested-global",
            lam_index=8,
            vars_of_interest=["t2m"],
            reshape_cell_to_2d=reshape_cell_to_2d,
            horizontal_regrid_kwargs=regrid_kwargs,
        )

    regrid.assert_called_once()
    assert regrid.call_args.kwargs == {"lam_index": 8, "lcc_info": None, "horizontal_regrid_kwargs": regrid_kwargs}
    dims = ("time", "latitude", "longitude") if reshape_cell_to_2d else ("time", "cell")
    assert result["t2m"].dims == dims


def test_open_anemoi_inference_dataset_rejects_ensemble_dim(tmp_path):
    ids = truth_at(DATES[0], 6).expand_dims(ensemble=2, axis=1)
    path = tmp_path / "ens.nc"
    ids.to_netcdf(path)

    with pytest.raises(NotImplementedError, match="ensemble"):
        data.open_anemoi_inference_dataset(str(path), model_type="global")


# --- open_forecast_zarr_dataset ---------------------------------------------------


def test_open_forecast_zarr_dataset_flattens(write_forecast_zarr):
    path = write_forecast_zarr("latlon")

    result = data.open_forecast_zarr_dataset(path, t0=DATES[1], member=1, levels=[850], load=True)

    raw = xr.open_zarr(path).sel(t0=DATES[1], member=1)
    assert set(result) == {"t", "t2m"}
    assert result["t"].dims == ("time", "level", "cell")
    np.testing.assert_array_equal(result["time"], DATES[1:4])
    np.testing.assert_array_equal(result["t2m"], raw["t2m"].values.reshape(3, N_CELL))
    assert "t0" not in result and "valid_time" not in result


def test_open_forecast_zarr_dataset_2d_with_rename(write_forecast_zarr):
    path = write_forecast_zarr("latlon")

    result = data.open_forecast_zarr_dataset(
        path, t0=DATES[0], member=0, reshape_cell_to_2d=True, rename_to_longnames=True
    )

    assert result["2m_temperature"].dims == ("time", "latitude", "longitude")
    np.testing.assert_array_equal(result["latitude"], LAT)


def test_open_forecast_zarr_dataset_trims_lam(write_forecast_zarr):
    path = write_forecast_zarr("xy")

    result = data.open_forecast_zarr_dataset(
        path,
        t0=DATES[0],
        member=0,
        vars_of_interest=["t2m"],
        trim_edge=[1, 0, 0, 1],
        lcc_info={"n_x": 3, "n_y": 3},
    )

    raw = xr.open_zarr(path)["t2m"].sel(t0=DATES[0], member=0).values
    np.testing.assert_array_equal(result["t2m"], raw[:, 1:, :3].reshape(3, 9))
