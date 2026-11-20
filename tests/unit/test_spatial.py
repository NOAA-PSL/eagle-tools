from unittest import mock

import numpy as np
import pytest
import xarray as xr
import yaml

from eagle.tools import data, spatial
from eagle.tools.metrics import get_gridcell_area_weights
from eagle.tools.reshape import reshape_cell_to_latlon

from synthetic import DATES, make_cell_ds


@pytest.fixture
def target():
    return make_cell_ds(value=0.0)


@pytest.fixture
def prediction():
    """Two members, with errors of -1 and +3 relative to the target."""
    return make_cell_ds(member=2, value=0.0) + xr.DataArray([1.0, -3.0], dims="member")


@pytest.mark.parametrize("keep_t0", [False, True])
def test_postprocess(keep_t0):
    result = spatial.postprocess(make_cell_ds(n_time=3), keep_t0=keep_t0)

    assert result["t2m"].dims == ("fhr", "cell")
    np.testing.assert_array_equal(result["fhr"], [0, 6, 12])
    assert ("t0" in result.coords) is keep_t0


def test_rmse_is_per_cell_member_mean(target, prediction):
    result = spatial.rmse(target, prediction, weights=2.0)

    assert result["t2m"].dims == ("fhr", "cell")
    np.testing.assert_allclose(result["t2m"], np.sqrt(2.0 * (1 + 9) / 2))


def test_mae_is_per_cell_member_mean(target, prediction):
    np.testing.assert_allclose(spatial.mae(target, prediction, weights=2.0)["t2m"], 2.0 * (1 + 3) / 2)


def test_signed_difference_is_target_minus_prediction(target, prediction):
    result = spatial.signed_difference(target, prediction, keep_t0=True)

    np.testing.assert_allclose(result["t2m"], (-1 + 3) / 2)
    assert "t0" in result.coords


# --- main -------------------------------------------------------------------------


@pytest.fixture
def config(serial_config, anemoi_zarr, write_inference):
    for t0 in DATES[:2]:
        write_inference(t0, offset=1.0)
    return {
        **serial_config,
        "model_type": "global",
        "verification_dataset_path": anemoi_zarr,
        "forecast_path": write_inference.path,
        "start_date": str(DATES[0]),
        "end_date": str(DATES[1]),
        "freq": "6h",
        "lead_time": 12,
        "vars_of_interest": ["t2m"],
    }


@pytest.fixture
def weights(anemoi_zarr):
    vds = data.open_anemoi_dataset_with_xarray(anemoi_zarr, "global", reshape_cell_to_2d=True)
    return get_gridcell_area_weights(vds, "global", reshape_cell_to_2d=True)


def open_result(config, name):
    return xr.open_dataset(f"{config['output_path']}/spatial.{name}.global.nc")


def test_main_averages_over_initial_conditions(config, weights, tmp_path):
    path = tmp_path / "spatial.yaml"
    path.write_text(yaml.dump({k: v for k, v in config.items() if k not in ("topo", "use_mpi")}))

    spatial.main(str(path))

    # A uniform error of 1, weighted by area per gridcell
    rmse = open_result(config, "rmse")
    assert set(rmse["t2m"].dims) == {"fhr", "latitude", "longitude"}
    np.testing.assert_allclose(rmse["t2m"], np.sqrt(weights).broadcast_like(rmse["t2m"]))
    np.testing.assert_allclose(open_result(config, "mae")["t2m"], weights.broadcast_like(rmse["t2m"]))


def test_main_keep_t0_stores_each_initial_condition(config, weights):
    config["keep_t0"] = True

    spatial.main(config)

    for name in ("rmse", "mae"):
        result = xr.open_dataset(f"{config['output_path']}/spatial.{name}.perIC.global.nc")
        assert set(result["t2m"].dims) == {"t0", "fhr", "latitude", "longitude"}
        np.testing.assert_array_equal(result["t0"], DATES[:2])


def test_main_from_forecast_zarr(config, write_forecast_zarr):
    config.update(forecast_path=write_forecast_zarr("latlon"), from_anemoi=False)

    spatial.main(config)

    result = open_result(config, "rmse")
    assert set(result["t2m"].dims) == {"fhr", "latitude", "longitude"}
    assert np.isfinite(result["t2m"]).all()


def test_main_regrids_forecast_and_target(config):
    config.update(target_regrid_kwargs={"method": "target"}, forecast_regrid_kwargs={"method": "forecast"})

    with (
        mock.patch.object(spatial, "horizontal_regrid", side_effect=lambda xds, **kw: xds) as regrid,
        mock.patch("eagle.tools.metrics.horizontal_regrid", side_effect=lambda xds, **kw: xds),
    ):
        spatial.main(config)

    methods = [c.kwargs["method"] for c in regrid.call_args_list]
    assert methods == ["forecast", "target"] * 2
    assert set(open_result(config, "rmse")["t2m"].dims) == {"fhr", "latitude", "longitude"}


def test_main_nested_global(config):
    config.update(
        model_type="nested-global",
        lam_index=8,
        anemoi_reference_dataset_kwargs={},
        forecast_regrid_kwargs={},
    )

    with (
        mock.patch.object(spatial, "prepare_regrid_target_mask", return_value=("grid.nc", None)),
        mock.patch.object(data, "regrid_nested_to_latlon", side_effect=lambda xds, **kw: reshape_cell_to_latlon(xds)),
    ):
        spatial.main(config)

    result = xr.open_dataset(f"{config['output_path']}/spatial.rmse.nested-global.nc")
    assert set(result["t2m"].dims) == {"fhr", "latitude", "longitude"}


def test_main_does_not_support_mpi(config):
    config["use_mpi"] = True

    with pytest.raises(NotImplementedError):
        spatial.main(config)
