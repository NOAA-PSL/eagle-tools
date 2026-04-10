import pytest
import numpy as np
import xarray as xr
import pandas as pd
from eagle.tools.metrics import rmse, mae, postprocess, get_gridcell_area_weights


@pytest.mark.unit
class TestPostprocess:
    """Tests for the postprocess function."""

    def test_postprocess_creates_t0(self, sample_dataset):
        """Test that postprocess creates t0 coordinate."""
        result = postprocess(sample_dataset)

        assert "t0" in result.coords

        # t0 should be the first timestamp (as numpy datetime64 or pandas Timestamp)
        expected_t0 = pd.Timestamp(sample_dataset["time"][0].values)
        result_t0 = pd.Timestamp(result.coords["t0"].values)
        assert result_t0 == expected_t0

    def test_postprocess_creates_lead_time(self, sample_dataset):
        """Test that postprocess creates lead_time coordinate."""
        result = postprocess(sample_dataset)

        assert "lead_time" in result.coords

        # lead_time should be relative to first time
        expected_lead_time = sample_dataset["time"] - sample_dataset["time"][0]
        np.testing.assert_array_equal(
            result.coords["lead_time"].values, expected_lead_time.values
        )

    def test_postprocess_creates_fhr(self, sample_dataset):
        """Test that postprocess creates fhr (forecast hour) variable."""
        result = postprocess(sample_dataset)

        assert "fhr" in result.coords

        # fhr should be in integer hours
        lead_time = sample_dataset["time"] - sample_dataset["time"][0]
        expected_fhr = lead_time.values.astype("timedelta64[h]").astype(int)
        np.testing.assert_array_equal(result.coords["fhr"].values, expected_fhr)

    def test_postprocess_swaps_dims(self, sample_dataset):
        """Test that postprocess swaps time dimension with fhr."""
        result = postprocess(sample_dataset)

        assert "fhr" in result.dims
        assert "time" not in result.dims

    def test_postprocess_drops_time(self, sample_dataset):
        """Test that time variable is dropped after swap."""
        result = postprocess(sample_dataset)

        assert "time" not in result.data_vars
        assert "time" not in result.coords


@pytest.mark.unit
class TestRMSE:
    """Tests for the rmse function."""

    def test_rmse_basic_calculation(self):
        """Test basic RMSE calculation."""
        n_time = 5
        n_cell = 100

        # Create target and prediction datasets
        target = xr.Dataset(
            {
                "temperature": (
                    ["time", "cell"],
                    np.ones((n_time, n_cell)) * 10,
                ),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = xr.Dataset(
            {
                "temperature": (
                    ["time", "cell"],
                    np.ones((n_time, n_cell)) * 13,  # Error of 3
                ),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        result = rmse(target, prediction)

        # RMSE should be 3.0 for all times
        assert "temperature" in result.data_vars
        np.testing.assert_allclose(result["temperature"].values, 3.0)

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect prediction (should be 0)."""
        n_time = 5
        n_cell = 100

        target = xr.Dataset(
            {
                "temperature": (
                    ["time", "cell"],
                    np.random.randn(n_time, n_cell),
                ),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = target.copy()

        result = rmse(target, prediction)

        np.testing.assert_allclose(result["temperature"].values, 0.0, atol=1e-10)

    def test_rmse_with_weights(self):
        """Test RMSE calculation with area weights."""
        n_time = 3
        n_cell = 10

        target = xr.Dataset(
            {
                "temperature": (
                    ["time", "cell"],
                    np.ones((n_time, n_cell)) * 10,
                ),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = xr.Dataset(
            {
                "temperature": (
                    ["time", "cell"],
                    np.ones((n_time, n_cell)) * 12,  # Error of 2
                ),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        # Create simple weights
        weights = xr.DataArray(np.ones(n_cell), coords={"cell": np.arange(n_cell)})

        result = rmse(target, prediction, weights=weights)

        # With uniform weights, RMSE should still be 2.0
        np.testing.assert_allclose(result["temperature"].values, 2.0)

    def test_rmse_multiple_variables(self):
        """Test RMSE calculation with multiple variables."""
        n_time = 5
        n_cell = 100

        target = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 10),
                "pressure": (["time", "cell"], np.ones((n_time, n_cell)) * 1000),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 13),
                "pressure": (["time", "cell"], np.ones((n_time, n_cell)) * 1005),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        result = rmse(target, prediction)

        assert "temperature" in result.data_vars
        assert "pressure" in result.data_vars
        np.testing.assert_allclose(result["temperature"].values, 3.0)
        np.testing.assert_allclose(result["pressure"].values, 5.0)

    def test_rmse_postprocessing(self):
        """Test that rmse applies postprocessing."""
        n_time = 5
        n_cell = 100

        target = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 10),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 13),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        result = rmse(target, prediction)

        # Check postprocessing was applied
        assert "t0" in result.coords
        assert "fhr" in result.coords
        assert "lead_time" in result.coords


@pytest.mark.unit
class TestMAE:
    """Tests for the mae function."""

    def test_mae_basic_calculation(self):
        """Test basic MAE calculation."""
        n_time = 5
        n_cell = 100

        target = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 10),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 13),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        result = mae(target, prediction)

        # MAE should be 3.0 for all times
        assert "temperature" in result.data_vars
        np.testing.assert_allclose(result["temperature"].values, 3.0)

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect prediction (should be 0)."""
        n_time = 5
        n_cell = 100

        target = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.random.randn(n_time, n_cell)),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = target.copy()

        result = mae(target, prediction)

        np.testing.assert_allclose(result["temperature"].values, 0.0, atol=1e-10)

    def test_mae_handles_positive_and_negative_errors(self):
        """Test that MAE handles positive and negative errors correctly."""
        n_time = 2
        n_cell = 2

        target = xr.Dataset(
            {
                "temperature": (["time", "cell"], [[10, 10], [10, 10]]),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        # One cell has +3 error, one has -3 error
        prediction = xr.Dataset(
            {
                "temperature": (["time", "cell"], [[13, 7], [13, 7]]),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        result = mae(target, prediction)

        # MAE should be 3.0 (absolute value of errors)
        np.testing.assert_allclose(result["temperature"].values, 3.0)

    def test_mae_with_weights(self):
        """Test MAE calculation with area weights."""
        n_time = 3
        n_cell = 10

        target = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 10),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        prediction = xr.Dataset(
            {
                "temperature": (["time", "cell"], np.ones((n_time, n_cell)) * 12),
            },
            coords={
                "time": pd.date_range("2024-01-01", periods=n_time, freq="6h"),
                "cell": np.arange(n_cell),
            },
        )

        weights = xr.DataArray(np.ones(n_cell), coords={"cell": np.arange(n_cell)})

        result = mae(target, prediction, weights=weights)

        np.testing.assert_allclose(result["temperature"].values, 2.0)


@pytest.mark.unit
class TestGetGridcellAreaWeights:
    """Tests for the get_gridcell_area_weights function."""

    def test_get_gridcell_area_weights_lam(self):
        """Test that LAM model type returns 1.0 (assumes equal area)."""
        ds = xr.Dataset()  # Empty dataset is fine for LAM

        result = get_gridcell_area_weights(ds, model_type="lam")

        assert result == 1.0

    def test_get_gridcell_area_weights_nested_lam(self):
        """Test that nested-lam model type returns 1.0."""
        ds = xr.Dataset()

        result = get_gridcell_area_weights(ds, model_type="nested-lam")

        assert result == 1.0

    def test_get_gridcell_area_weights_invalid_model_type(self):
        """Test that invalid model type raises NotImplementedError."""
        ds = xr.Dataset()

        with pytest.raises(NotImplementedError):
            get_gridcell_area_weights(ds, model_type="invalid")
