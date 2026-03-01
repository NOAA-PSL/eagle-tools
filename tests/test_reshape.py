import pytest
import numpy as np
import xarray as xr
from eagle.tools.reshape import (
    flatten_to_cell,
    reshape_cell_dim,
    reshape_cell_to_latlon,
    reshape_cell_to_xy,
)


@pytest.mark.unit
class TestFlattenToCell:
    """Tests for the flatten_to_cell function."""

    def test_flatten_to_cell_from_xy(self, sample_dataset_2d):
        """Test flattening a dataset with x and y dimensions."""
        result = flatten_to_cell(sample_dataset_2d)

        assert "cell" in result.dims
        assert "x" not in result.dims
        assert "y" not in result.dims
        assert len(result.cell) == 100  # 10 * 10

        # Check that data is preserved
        assert "temperature" in result.data_vars
        assert result.temperature.shape == (10, 100)  # (time, cell)

    def test_flatten_to_cell_from_latlon(self, sample_latlon_dataset):
        """Test flattening a dataset with latitude and longitude dimensions."""
        result = flatten_to_cell(sample_latlon_dataset)

        assert "cell" in result.dims
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims
        assert len(result.cell) == 200  # 10 * 20

        # Check that data is preserved
        assert "temperature" in result.data_vars
        assert result.temperature.shape == (5, 200)  # (time, cell)

    def test_flatten_to_cell_raises_on_invalid_dims(self):
        """Test that flatten_to_cell raises an error for invalid dimensions."""
        # Create dataset with neither (x,y) nor (lat,lon)
        ds = xr.Dataset(
            {"temp": (["time", "z"], np.random.randn(5, 10))},
            coords={"time": np.arange(5), "z": np.arange(10)},
        )

        with pytest.raises(KeyError, match="Unclear on the dimensions"):
            flatten_to_cell(ds)


@pytest.mark.unit
class TestReshapeCellToLatlon:
    """Tests for the reshape_cell_to_latlon function."""

    def test_reshape_cell_to_latlon(self):
        """Test reshaping from cell to lat/lon grid."""
        # Create a dataset with cell dimension that can be reshaped
        n_lat = 5
        n_lon = 10
        n_time = 3

        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)

        # Create flattened coordinates
        lat_flat = np.repeat(lats, n_lon)
        lon_flat = np.tile(lons, n_lat)

        ds = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.random.randn(n_time, n_lat * n_lon)),
            },
            coords={
                "time": np.arange(n_time),
                "cell": np.arange(n_lat * n_lon),
                "latitude": ("cell", lat_flat),
                "longitude": ("cell", lon_flat),
            },
        )

        result = reshape_cell_to_latlon(ds)

        assert "latitude" in result.dims
        assert "longitude" in result.dims
        assert "cell" not in result.dims
        assert len(result.latitude) == n_lat
        assert len(result.longitude) == n_lon
        assert result.temperature.shape == (n_time, n_lat, n_lon)

    def test_reshape_cell_to_latlon_preserves_attributes(self):
        """Test that attributes are preserved during reshaping."""
        n_lat = 3
        n_lon = 4
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)

        lat_flat = np.repeat(lats, n_lon)
        lon_flat = np.tile(lons, n_lat)

        ds = xr.Dataset(
            {
                "temperature": (
                    ["cell"],
                    np.random.randn(n_lat * n_lon),
                    {"units": "K", "long_name": "Temperature"},
                ),
            },
            coords={
                "cell": np.arange(n_lat * n_lon),
                "latitude": ("cell", lat_flat),
                "longitude": ("cell", lon_flat),
            },
        )

        result = reshape_cell_to_latlon(ds)

        assert result.temperature.attrs["units"] == "K"
        assert result.temperature.attrs["long_name"] == "Temperature"


@pytest.mark.unit
class TestReshapeCellToXY:
    """Tests for the reshape_cell_to_xy function."""

    def test_reshape_cell_to_xy(self):
        """Test reshaping from cell to x/y grid."""
        n_x = 10
        n_y = 8
        n_time = 3

        ds = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.random.randn(n_time, n_x * n_y)),
            },
            coords={
                "time": np.arange(n_time),
                "cell": np.arange(n_x * n_y),
            },
        )

        result = reshape_cell_to_xy(ds, n_x=n_x, n_y=n_y)

        assert "x" in result.dims
        assert "y" in result.dims
        assert "cell" not in result.dims
        assert len(result.x) == n_x
        assert len(result.y) == n_y
        assert result.temperature.shape == (n_time, n_y, n_x)

    def test_reshape_cell_to_xy_with_coords(self):
        """Test that coordinates are handled properly."""
        n_x = 5
        n_y = 4

        ds = xr.Dataset(
            {
                "temperature": (["cell"], np.random.randn(n_x * n_y)),
            },
            coords={
                "cell": np.arange(n_x * n_y),
                "latitude": ("cell", np.random.randn(n_x * n_y)),
                "longitude": ("cell", np.random.randn(n_x * n_y)),
            },
        )

        result = reshape_cell_to_xy(ds, n_x=n_x, n_y=n_y)

        # Coordinates with cell dimension should be reshaped
        assert "latitude" in result
        assert result.latitude.dims == ("y", "x")
        assert result.longitude.dims == ("y", "x")


@pytest.mark.unit
class TestReshapeCellDim:
    """Tests for the reshape_cell_dim function."""

    def test_reshape_cell_dim_global(self):
        """Test reshape_cell_dim for global model type."""
        n_lat = 5
        n_lon = 10

        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)

        lat_flat = np.repeat(lats, n_lon)
        lon_flat = np.tile(lons, n_lat)

        ds = xr.Dataset(
            {
                "temperature": (["cell"], np.random.randn(n_lat * n_lon)),
            },
            coords={
                "cell": np.arange(n_lat * n_lon),
                "latitude": ("cell", lat_flat),
                "longitude": ("cell", lon_flat),
            },
        )

        result = reshape_cell_dim(ds, model_type="global", lcc_info=None)

        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_reshape_cell_dim_lam(self):
        """Test reshape_cell_dim for LAM model type."""
        n_x = 10
        n_y = 8

        ds = xr.Dataset(
            {
                "temperature": (["cell"], np.random.randn(n_x * n_y)),
            },
            coords={
                "cell": np.arange(n_x * n_y),
            },
        )

        lcc_info = {"n_x": n_x, "n_y": n_y}
        result = reshape_cell_dim(ds, model_type="lam", lcc_info=lcc_info)

        assert "x" in result.dims
        assert "y" in result.dims

    def test_reshape_cell_dim_lam_requires_lcc_info(self):
        """Test that LAM model type requires lcc_info."""
        ds = xr.Dataset(
            {
                "temperature": (["cell"], np.random.randn(100)),
            },
            coords={
                "cell": np.arange(100),
            },
        )

        with pytest.raises(AssertionError, match="Need lcc_info"):
            reshape_cell_dim(ds, model_type="lam", lcc_info=None)

    def test_reshape_cell_dim_nested_lam(self):
        """Test reshape_cell_dim for nested-lam model type."""
        n_x = 10
        n_y = 8

        ds = xr.Dataset(
            {
                "temperature": (["cell"], np.random.randn(n_x * n_y)),
            },
            coords={
                "cell": np.arange(n_x * n_y),
            },
        )

        lcc_info = {"n_x": n_x, "n_y": n_y}
        result = reshape_cell_dim(ds, model_type="nested-lam", lcc_info=lcc_info)

        assert "x" in result.dims
        assert "y" in result.dims
