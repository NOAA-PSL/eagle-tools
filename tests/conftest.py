import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def sample_dataset():
    """Create a simple xarray dataset for testing."""
    np.random.seed(42)
    n_time = 10
    n_cell = 100

    ds = xr.Dataset(
        {
            "temperature": (["time", "cell"], np.random.randn(n_time, n_cell) * 10 + 273.15),
            "pressure": (["time", "cell"], np.random.randn(n_time, n_cell) * 100 + 1013.25),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
            "cell": np.arange(n_cell),
            "latitude": ("cell", np.random.uniform(-90, 90, n_cell)),
            "longitude": ("cell", np.random.uniform(-180, 180, n_cell)),
        },
    )
    return ds


@pytest.fixture
def sample_dataset_2d():
    """Create a 2D gridded xarray dataset for testing."""
    np.random.seed(42)
    n_time = 10
    n_x = 10
    n_y = 10

    ds = xr.Dataset(
        {
            "temperature": (["time", "y", "x"], np.random.randn(n_time, n_y, n_x) * 10 + 273.15),
            "pressure": (["time", "y", "x"], np.random.randn(n_time, n_y, n_x) * 100 + 1013.25),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
            "x": np.arange(n_x),
            "y": np.arange(n_y),
        },
    )
    return ds


@pytest.fixture
def sample_dataset_with_levels():
    """Create a dataset with vertical levels for testing."""
    np.random.seed(42)
    n_time = 5
    n_cell = 50
    n_level = 3

    ds = xr.Dataset(
        {
            "temperature": (["time", "level", "cell"], np.random.randn(n_time, n_level, n_cell) * 10 + 273.15),
            "u": (["time", "level", "cell"], np.random.randn(n_time, n_level, n_cell) * 5),
            "v": (["time", "level", "cell"], np.random.randn(n_time, n_level, n_cell) * 5),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
            "level": [850, 500, 250],
            "cell": np.arange(n_cell),
            "latitude": ("cell", np.random.uniform(-90, 90, n_cell)),
            "longitude": ("cell", np.random.uniform(-180, 180, n_cell)),
        },
    )
    return ds


@pytest.fixture
def sample_dataset_with_wind():
    """Create a dataset with wind components for testing wind speed calculations."""
    np.random.seed(42)
    n_time = 5
    n_cell = 50

    ds = xr.Dataset(
        {
            "ugrd10m": (["time", "cell"], np.random.randn(n_time, n_cell) * 5),
            "vgrd10m": (["time", "cell"], np.random.randn(n_time, n_cell) * 5),
            "u100": (["time", "cell"], np.random.randn(n_time, n_cell) * 10),
            "v100": (["time", "cell"], np.random.randn(n_time, n_cell) * 10),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
            "cell": np.arange(n_cell),
            "latitude": ("cell", np.random.uniform(-90, 90, n_cell)),
            "longitude": ("cell", np.random.uniform(-180, 180, n_cell)),
        },
    )
    return ds


@pytest.fixture
def sample_dataset_with_member():
    """Create a dataset with ensemble member dimension."""
    np.random.seed(42)
    n_time = 5
    n_cell = 50
    n_member = 3

    ds = xr.Dataset(
        {
            "temperature": (["time", "member", "cell"], np.random.randn(n_time, n_member, n_cell) * 10 + 273.15),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
            "member": np.arange(n_member),
            "cell": np.arange(n_cell),
        },
    )
    return ds


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML config file for testing."""
    import yaml

    config = {
        "model_type": "global",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
        "freq": "6h",
        "lead_time": "240h",
        "output_path": str(temp_dir),
        "levels": [850, 500, 250],
        "vars_of_interest": ["temperature", "pressure"],
    }

    config_file = temp_dir / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return str(config_file)


@pytest.fixture
def sample_latlon_dataset():
    """Create a dataset with regular lat/lon grid for testing."""
    np.random.seed(42)
    n_time = 5
    n_lat = 10
    n_lon = 20

    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(-180, 180, n_lon)

    ds = xr.Dataset(
        {
            "temperature": (["time", "latitude", "longitude"],
                          np.random.randn(n_time, n_lat, n_lon) * 10 + 273.15),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds
