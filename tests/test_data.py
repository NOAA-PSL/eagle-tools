import pytest
import numpy as np
import xarray as xr
import pandas as pd
from eagle.tools.data import (
    subsample,
    drop_forcing_vars,
    calc_wind_speed,
    _wind_speed,
    _get_xy,
    trim_xarray_edge,
)


@pytest.mark.unit
class TestGetXY:
    """Tests for the _get_xy function."""

    def test_get_xy_with_cell_dim(self):
        """Test _get_xy with a dataset that has cell dimension."""
        n_x = 10
        n_y = 8

        ds = xr.Dataset(
            {"temperature": (["cell"], np.random.randn(n_x * n_y))},
            coords={"cell": np.arange(n_x * n_y)},
        )

        result = _get_xy(ds, n_x=n_x, n_y=n_y)

        assert "x" in result
        assert "y" in result
        assert len(result.x) == n_x * n_y
        assert len(result.y) == n_x * n_y

        # Check that x and y are properly tiled
        expected_x = np.tile(np.arange(n_x), n_y)
        np.testing.assert_array_equal(result.x.values, expected_x)

    def test_get_xy_without_cell_dim(self):
        """Test _get_xy with a dataset without cell dimension."""
        n_x = 10
        n_y = 8

        ds = xr.Dataset(
            {"temperature": (["time"], np.random.randn(5))},
            coords={"time": np.arange(5)},
        )

        result = _get_xy(ds, n_x=n_x, n_y=n_y)

        assert "x" in result
        assert "y" in result
        assert result.x.dims == ("x",)
        assert result.y.dims == ("y",)
        assert len(result.x) == n_x
        assert len(result.y) == n_y


@pytest.mark.unit
class TestTrimXarrayEdge:
    """Tests for the trim_xarray_edge function."""

    def test_trim_xarray_edge_with_xy_dims(self):
        """Test trimming a dataset with x and y dimensions."""
        n_x = 20
        n_y = 15

        ds = xr.Dataset(
            {"temperature": (["time", "y", "x"], np.random.randn(5, n_y, n_x))},
            coords={
                "time": np.arange(5),
                "x": np.arange(n_x),
                "y": np.arange(n_y),
            },
        )

        # Trim 2 from left, 3 from right, 1 from bottom, 2 from top
        trim_edge = [2, 3, 1, 2]
        lcc_info = {"n_x": n_x - 5, "n_y": n_y - 3}

        result = trim_xarray_edge(ds, lcc_info, trim_edge)

        # Expected size after trimming
        expected_x = n_x - 2 - 3
        expected_y = n_y - 1 - 2

        assert len(result.x) == expected_x
        assert len(result.y) == expected_y

        # Check that coordinates are reset to 0->len
        np.testing.assert_array_equal(result.x.values, np.arange(expected_x))
        np.testing.assert_array_equal(result.y.values, np.arange(expected_y))

    def test_trim_xarray_edge_with_cell_dim(self):
        """Test trimming a dataset with cell dimension."""
        n_x = 20
        n_y = 15

        ds = xr.Dataset(
            {"temperature": (["cell"], np.random.randn(n_x * n_y))},
            coords={"cell": np.arange(n_x * n_y)},
        )

        trim_edge = [2, 3, 1, 2]
        lcc_info = {"n_x": n_x - 5, "n_y": n_y - 3}

        result = trim_xarray_edge(ds, lcc_info, trim_edge)

        # Cell dimension should be reset
        assert "cell" in result.dims
        assert result.cell.values[0] == 0
        assert len(result.cell) == (n_x - 5) * (n_y - 3)


@pytest.mark.unit
class TestDropForcingVars:
    """Tests for the drop_forcing_vars function."""

    def test_drop_forcing_vars_removes_standard_forcing(self):
        """Test that standard forcing variables are removed."""
        ds = xr.Dataset(
            {
                "temperature": (["time"], np.random.randn(10)),
                "pressure": (["time"], np.random.randn(10)),
                "cos_julian_day": (["time"], np.random.randn(10)),
                "sin_julian_day": (["time"], np.random.randn(10)),
                "orog": (["cell"], np.random.randn(100)),
                "lsm": (["cell"], np.random.randn(100)),
            },
            coords={"time": np.arange(10), "cell": np.arange(100)},
        )

        result = drop_forcing_vars(ds)

        assert "temperature" in result.data_vars
        assert "pressure" in result.data_vars
        assert "cos_julian_day" not in result.data_vars
        assert "sin_julian_day" not in result.data_vars
        assert "orog" not in result.data_vars
        assert "lsm" not in result.data_vars

    def test_drop_forcing_vars_handles_missing_vars(self):
        """Test that function handles datasets without forcing vars."""
        ds = xr.Dataset(
            {
                "temperature": (["time"], np.random.randn(10)),
                "pressure": (["time"], np.random.randn(10)),
            },
            coords={"time": np.arange(10)},
        )

        # Should not raise an error
        result = drop_forcing_vars(ds)

        assert "temperature" in result.data_vars
        assert "pressure" in result.data_vars


@pytest.mark.unit
class TestWindSpeed:
    """Tests for wind speed calculation functions."""

    def test_wind_speed_calculation(self):
        """Test basic wind speed calculation."""
        u = xr.DataArray([3.0, 4.0, 0.0], coords={"x": [0, 1, 2]})
        v = xr.DataArray([4.0, 3.0, 5.0], coords={"x": [0, 1, 2]})

        result = _wind_speed(u, v, "Test Wind Speed")

        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_allclose(result.values, expected)
        assert result.attrs["long_name"] == "Test Wind Speed"
        assert result.attrs["units"] == "m/s"

    def test_calc_wind_speed_10m(self, sample_dataset_with_wind):
        """Test calculation of 10m wind speed."""
        vars_of_interest = ["10m_wind_speed", "ugrd10m", "vgrd10m"]

        result = calc_wind_speed(sample_dataset_with_wind, vars_of_interest)

        assert "10m_wind_speed" in result.data_vars
        assert result["10m_wind_speed"].shape == sample_dataset_with_wind["ugrd10m"].shape

        # Verify calculation
        expected = np.sqrt(
            sample_dataset_with_wind["ugrd10m"] ** 2
            + sample_dataset_with_wind["vgrd10m"] ** 2
        )
        np.testing.assert_allclose(result["10m_wind_speed"].values, expected.values)

    def test_calc_wind_speed_100m(self, sample_dataset_with_wind):
        """Test calculation of 100m wind speed."""
        vars_of_interest = ["100m_wind_speed", "u100", "v100"]

        result = calc_wind_speed(sample_dataset_with_wind, vars_of_interest)

        assert "100m_wind_speed" in result.data_vars

        # Verify calculation
        expected = np.sqrt(
            sample_dataset_with_wind["u100"] ** 2 + sample_dataset_with_wind["v100"] ** 2
        )
        np.testing.assert_allclose(result["100m_wind_speed"].values, expected.values)

    def test_calc_wind_speed_multiple_heights(self, sample_dataset_with_wind):
        """Test calculation of wind speed at multiple heights."""
        vars_of_interest = ["10m_wind_speed", "100m_wind_speed"]

        result = calc_wind_speed(sample_dataset_with_wind, vars_of_interest)

        assert "10m_wind_speed" in result.data_vars
        assert "100m_wind_speed" in result.data_vars


@pytest.mark.unit
class TestSubsample:
    """Tests for the subsample function."""

    def test_subsample_levels(self, sample_dataset_with_levels):
        """Test subsampling specific vertical levels."""
        levels = [850, 500]

        result = subsample(sample_dataset_with_levels, levels=levels, vars_of_interest=None)

        assert len(result.level) == 2
        np.testing.assert_array_equal(result.level.values, levels)

    def test_subsample_vars_of_interest(self, sample_dataset_with_levels):
        """Test subsampling specific variables."""
        vars_of_interest = ["temperature"]

        result = subsample(
            sample_dataset_with_levels, levels=None, vars_of_interest=vars_of_interest
        )

        assert "temperature" in result.data_vars
        assert "u" not in result.data_vars
        assert "v" not in result.data_vars

    def test_subsample_member(self, sample_dataset_with_member):
        """Test subsampling specific ensemble member."""
        member = 1

        result = subsample(sample_dataset_with_member, member=member)

        assert "member" not in result.dims
        assert result.temperature.shape == (5, 50)  # (time, cell) - member squeezed

    def test_subsample_with_wind_speed(self):
        """Test that subsample calculates wind speed when requested."""
        ds = xr.Dataset(
            {
                "ugrd": (["time", "cell"], np.random.randn(5, 50)),
                "vgrd": (["time", "cell"], np.random.randn(5, 50)),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=5, freq="6h"),
                "cell": np.arange(50),
            },
        )

        vars_of_interest = ["wind_speed", "ugrd", "vgrd"]

        result = subsample(ds, vars_of_interest=vars_of_interest)

        assert "wind_speed" in result.data_vars
        assert "ugrd" in result.data_vars
        assert "vgrd" in result.data_vars

    def test_subsample_drops_forcing_when_no_vars_specified(self):
        """Test that forcing vars are dropped when vars_of_interest is None."""
        ds = xr.Dataset(
            {
                "temperature": (["time"], np.random.randn(10)),
                "cos_julian_day": (["time"], np.random.randn(10)),
            },
            coords={"time": np.arange(10)},
        )

        result = subsample(ds, levels=None, vars_of_interest=None)

        assert "temperature" in result.data_vars
        assert "cos_julian_day" not in result.data_vars

    def test_subsample_combined(self, sample_dataset_with_levels):
        """Test combining multiple subsample operations."""
        levels = [850]
        vars_of_interest = ["temperature"]

        result = subsample(
            sample_dataset_with_levels, levels=levels, vars_of_interest=vars_of_interest
        )

        assert len(result.level) == 1
        assert result.level.values[0] == 850
        assert "temperature" in result.data_vars
        assert "u" not in result.data_vars
