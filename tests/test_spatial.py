import pytest
import numpy as np
import xarray as xr
import pandas as pd
from eagle.tools.spatial import postprocess, rmse, mae


@pytest.mark.unit
class TestSpatialPostprocess:
    """Tests for the spatial postprocess function."""

    def test_postprocess_without_keep_t0(self, sample_dataset):
        """Test postprocess when keep_t0 is False."""
        result = postprocess(sample_dataset, keep_t0=False)

        assert "t0" not in result.coords
        assert "fhr" in result.coords
        assert "lead_time" in result.coords

    def test_postprocess_with_keep_t0(self, sample_dataset):
        """Test postprocess when keep_t0 is True."""
        result = postprocess(sample_dataset, keep_t0=True)

        assert "t0" in result.coords
        assert "fhr" in result.coords
        assert "lead_time" in result.coords

        # t0 should be the first timestamp
        expected_t0 = pd.Timestamp(sample_dataset["time"][0].values)
        assert result.coords["t0"].values == expected_t0

    def test_postprocess_creates_lead_time(self, sample_dataset):
        """Test that postprocess creates lead_time coordinate."""
        result = postprocess(sample_dataset, keep_t0=False)

        assert "lead_time" in result.coords

        expected_lead_time = sample_dataset["time"] - sample_dataset["time"][0]
        np.testing.assert_array_equal(
            result.coords["lead_time"].values, expected_lead_time.values
        )

    def test_postprocess_creates_fhr(self, sample_dataset):
        """Test that postprocess creates fhr variable."""
        result = postprocess(sample_dataset, keep_t0=False)

        assert "fhr" in result.coords

        lead_time = sample_dataset["time"] - sample_dataset["time"][0]
        expected_fhr = lead_time.values.astype("timedelta64[h]").astype(int)
        np.testing.assert_array_equal(result.coords["fhr"].values, expected_fhr)


@pytest.mark.unit
class TestSpatialRMSE:
    """Tests for the spatial rmse function."""

    def test_rmse_basic_calculation(self, sample_dataset_with_member):
        """Test basic RMSE calculation for spatial module."""
        # Create target by adding constant error
        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 5.0  # Add error of 5

        result = rmse(target, prediction, keep_t0=False)

        # RMSE should be 5.0 everywhere
        assert "temperature" in result.data_vars
        np.testing.assert_allclose(result["temperature"].values, 5.0)

    def test_rmse_averages_over_member(self, sample_dataset_with_member):
        """Test that RMSE averages over member dimension."""
        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 3.0

        result = rmse(target, prediction, keep_t0=False)

        # Member dimension should be gone
        assert "member" not in result.dims

        # Spatial dimensions should be preserved
        assert "cell" in result.dims

    def test_rmse_with_keep_t0(self, sample_dataset_with_member):
        """Test RMSE with keep_t0=True."""
        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 2.0

        result = rmse(target, prediction, keep_t0=True)

        assert "t0" in result.coords

    def test_rmse_with_weights(self, sample_dataset_with_member):
        """Test RMSE with weights."""
        n_cell = len(sample_dataset_with_member.cell)
        weights = xr.DataArray(np.ones(n_cell), coords={"cell": np.arange(n_cell)})

        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 4.0

        result = rmse(target, prediction, weights=weights, keep_t0=False)

        np.testing.assert_allclose(result["temperature"].values, 4.0)


@pytest.mark.unit
class TestSpatialMAE:
    """Tests for the spatial mae function."""

    def test_mae_basic_calculation(self, sample_dataset_with_member):
        """Test basic MAE calculation for spatial module."""
        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 5.0

        result = mae(target, prediction, keep_t0=False)

        assert "temperature" in result.data_vars
        np.testing.assert_allclose(result["temperature"].values, 5.0)

    def test_mae_averages_over_member(self, sample_dataset_with_member):
        """Test that MAE averages over member dimension."""
        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 3.0

        result = mae(target, prediction, keep_t0=False)

        assert "member" not in result.dims
        assert "cell" in result.dims

    def test_mae_handles_positive_and_negative_errors(self):
        """Test that MAE handles mixed errors correctly."""
        n_time = 3
        n_member = 2
        n_cell = 10

        target = xr.Dataset(
            {
                "temperature": (
                    ["time", "member", "cell"],
                    np.ones((n_time, n_member, n_cell)) * 10,
                ),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "member": np.arange(n_member),
                "cell": np.arange(n_cell),
            },
        )

        # Create errors: +5 for all members
        prediction = target.copy()
        prediction["temperature"] = prediction["temperature"] + 5.0

        result = mae(target, prediction, keep_t0=False)

        # MAE should be 5.0 (absolute value)
        np.testing.assert_allclose(result["temperature"].values, 5.0)

    def test_mae_with_weights(self, sample_dataset_with_member):
        """Test MAE with weights."""
        n_cell = len(sample_dataset_with_member.cell)
        weights = xr.DataArray(np.ones(n_cell), coords={"cell": np.arange(n_cell)})

        target = sample_dataset_with_member.copy()
        prediction = sample_dataset_with_member.copy()
        prediction["temperature"] = prediction["temperature"] + 3.0

        result = mae(target, prediction, weights=weights, keep_t0=False)

        np.testing.assert_allclose(result["temperature"].values, 3.0)
