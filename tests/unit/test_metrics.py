from unittest import mock

import numpy as np
import pytest
import xarray as xr
import yaml
from ufs2arco.mpi import SerialTopology

from eagle.tools import data, metrics
from eagle.tools.reshape import reshape_cell_to_latlon

from synthetic import CELL_LAT, DATES, N_CELL, make_cell_ds

# Error of 2 in the northern half, 0 in the southern half, and area weights
# (unit mean) that count the north 3x as much as the south.
HALF = N_CELL // 2
ERROR = np.r_[np.full(HALF, 2.0), np.zeros(HALF)]
WEIGHTS = xr.DataArray(np.r_[np.full(HALF, 1.5), np.full(HALF, 0.5)], dims="cell")


@pytest.fixture
def target():
    return make_cell_ds(value=0.0)


@pytest.fixture
def prediction(target):
    return target + xr.DataArray(ERROR, dims="cell")


def identity_regrid(xds, **kwargs):
    return xds


# --- postprocess ------------------------------------------------------------------


def test_postprocess_indexes_by_forecast_hour():
    result = metrics.postprocess(make_cell_ds(n_time=3))

    assert "time" not in result.variables
    assert result["t2m"].dims == ("fhr", "cell")
    np.testing.assert_array_equal(result["fhr"], [0, 6, 12])
    np.testing.assert_array_equal(result["lead_time"], np.array([0, 6, 12], dtype="timedelta64[h]"))
    assert result["t0"].values == DATES[0]


# --- rmse / mae / spread / fcrps --------------------------------------------------


def test_rmse(target, prediction):
    np.testing.assert_allclose(metrics.rmse(target, prediction)["t2m"], np.sqrt(2.0))
    np.testing.assert_allclose(metrics.rmse(target, prediction, weights=WEIGHTS)["t2m"], np.sqrt(3.0))


def test_mae(target, prediction):
    np.testing.assert_allclose(metrics.mae(target, prediction)["t2m"], 1.0)
    np.testing.assert_allclose(metrics.mae(target, prediction, weights=WEIGHTS)["t2m"], 1.5)


def test_rmse_keeps_level_and_averages_over_member():
    target = make_cell_ds(member=2, value=0.0).expand_dims(level=[500, 850])
    prediction = target + xr.DataArray([1.0, 3.0], dims="member")

    result = metrics.rmse(target, prediction)

    assert result["t2m"].dims == ("level", "fhr")
    np.testing.assert_allclose(result["t2m"], np.sqrt(5.0))


@pytest.fixture
def ensemble():
    """Three members at -1, 0, +1 around a target of 0."""
    return make_cell_ds(member=3, value=0.0) + xr.DataArray([-1.0, 0.0, 1.0], dims="member")


def test_spread(ensemble):
    result = metrics.spread(ensemble)

    np.testing.assert_allclose(result["t2m"], np.sqrt(2.0 / 3.0))
    assert "fhr" in result.dims


def test_fcrps(ensemble):
    target = make_cell_ds(member=1, value=0.0)

    result = metrics.fcrps(target, ensemble)

    # mean |x_e - y| = 2/3, mean |x_e - x_i| = 8/9, fCRPS = 2/3 - (8/9) / (2 * 2)
    np.testing.assert_allclose(result["t2m"], 4.0 / 9.0)


# --- area weights -----------------------------------------------------------------


def test_area_weights_have_unit_mean_and_favor_low_latitudes():
    weights = metrics._area_weights(make_cell_ds())

    np.testing.assert_allclose(weights.mean(), 1.0)
    np.testing.assert_allclose(weights.where(CELL_LAT == 60).dropna("cell"), weights.where(CELL_LAT == -60).dropna("cell"))
    assert (weights.where(abs(CELL_LAT) == 20).min() > weights.where(abs(CELL_LAT) == 60).max()).item()


def test_area_weights_from_2d_grid_and_back():
    gridded = reshape_cell_to_latlon(make_cell_ds())

    weights = metrics._area_weights(gridded, reshape_cell_to_2d=True)

    assert weights.dims == ("latitude", "longitude")
    np.testing.assert_allclose(weights.mean(), 1.0)


def test_area_weights_warn_when_not_a_grid(caplog):
    xds = make_cell_ds().isel(cell=slice(0, N_CELL - 1))

    weights = metrics._area_weights(xds, reshape_cell_to_2d=True)

    assert weights.dims == ("cell",)
    assert "Could not reshape area weights" in caplog.text


def test_gridcell_area_weights_by_model_type():
    xds = make_cell_ds()

    assert metrics.get_gridcell_area_weights(xds, "lam") == 1.0
    assert metrics.get_gridcell_area_weights(xds, "nested-lam") == 1.0
    assert metrics.get_gridcell_area_weights(xds, "nested-global").dims == ("cell",)
    with pytest.raises(NotImplementedError):
        metrics.get_gridcell_area_weights(xds, "unknown")


def test_gridcell_area_weights_regrid():
    with mock.patch.object(metrics, "horizontal_regrid", side_effect=identity_regrid) as regrid:
        weights = metrics.get_gridcell_area_weights(make_cell_ds(), "global", regrid_kwargs={"method": "x"})

    assert regrid.call_args.kwargs == {"method": "x"}
    assert weights.name == "weights"


# --- subregions -------------------------------------------------------------------


def test_parse_subregions_fills_defaults_and_wraps_longitude():
    config = {
        "subregions": {
            "conus": {"latitude": [20, 50], "longitude": [-130, -60]},
            "tropics": {"latitude": [-20, 20]},
            "dateline": {"longitude": [170, -170]},
        }
    }

    assert metrics._parse_subregions(config) == {
        "conus": {"latitude": (20, 50), "longitude": (230, 300)},
        "tropics": {"latitude": (-20, 20), "longitude": (0, 359.99)},
        "dateline": {"latitude": (-90, 90), "longitude": (170, 190)},
    }
    assert metrics._parse_subregions({}) == {}


def test_parse_subregions_requires_bounds():
    with pytest.raises(ValueError, match="'empty' must have at least"):
        metrics._parse_subregions({"subregions": {"empty": {}}})


@pytest.mark.parametrize(
    ("longitude", "expected_lons"),
    [
        ((80, 190), {90, 180}),
        ((260, 10), {0, 270}),  # wraps through 0
    ],
)
def test_subregion_mask(longitude, expected_lons):
    xds = make_cell_ds()

    mask = metrics._subregion_mask(xds["latitude"], xds["longitude"], {"latitude": (0, 90), "longitude": longitude})

    selected = xds.where(mask, drop=True)
    assert set(selected["latitude"].values) == {60, 20}
    assert set(selected["longitude"].values) == expected_lons


def test_subregion_weights_renormalize_over_region():
    xds = make_cell_ds()
    weights = metrics._area_weights(xds)

    result = metrics._subregion_weights(weights, xds, {"latitude": (0, 90), "longitude": (0, 359.99)})

    assert int(result.notnull().sum()) == HALF
    np.testing.assert_allclose(result.mean(), 1.0)


def test_get_latlon_falls_back_to_dataset_for_scalar_weights():
    weights = metrics._area_weights(make_cell_ds())
    shifted = make_cell_ds().assign_coords(latitude=("cell", CELL_LAT + 1))

    np.testing.assert_array_equal(metrics._get_latlon(weights, shifted)[0], CELL_LAT)
    np.testing.assert_array_equal(metrics._get_latlon(1.0, shifted)[0], CELL_LAT + 1)


# --- main -------------------------------------------------------------------------


@pytest.fixture
def config(serial_config, anemoi_zarr, write_inference):
    return {
        **serial_config,
        "model_type": "global",
        "verification_dataset_path": anemoi_zarr,
        "forecast_path": write_inference.path,
        "start_date": str(DATES[0]),
        "end_date": str(DATES[1]),
        "freq": "6h",
        "lead_time": 12,
    }


def open_result(config, name):
    return xr.open_dataset(f"{config['output_path']}/{name}.nc")


def test_main_deterministic_from_yaml(config, write_inference, tmp_path):
    for t0 in DATES[:2]:
        write_inference(t0, offset=1.0)
    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.dump({k: v for k, v in config.items() if k not in ("topo", "use_mpi")}))

    metrics.main(str(path))

    for name in ("rmse", "mae"):
        result = open_result(config, f"{name}.global")
        assert set(result) == {"t", "t2m"}
        assert result["t"].dims == ("t0", "fhr", "level")
        np.testing.assert_array_equal(result["t0"], DATES[:2])
        np.testing.assert_allclose(result["t2m"], 1.0)


def test_main_ensemble_with_subregion(config, write_inference):
    for t0 in DATES[:2]:
        write_inference(t0, offset=1.0, member=0)
        write_inference(t0, offset=-1.0, member=1)
    config.update(n_members=2, vars_of_interest=["t2m"], subregions={"north": {"latitude": [0, 90]}})

    metrics.main(config)

    expected = {"rmse": 1.0, "mae": 1.0, "spread": 1.0, "fcrps": 0.5, "rmse_ensmean": 0.0, "mae_ensmean": 0.0}
    for suffix in ("", ".north"):
        for name, value in expected.items():
            result = open_result(config, f"{name}.global{suffix}")
            np.testing.assert_allclose(result["t2m"], value, atol=1e-12, err_msg=f"{name}{suffix}")
    assert open_result(config, "rmse.global")["t2m"].dims == ("t0", "member", "fhr")

    masks = open_result(config, "subregions.global")
    assert int(masks["north"].notnull().sum()) == HALF


def test_main_from_forecast_zarr(config, write_forecast_zarr):
    config.update(forecast_path=write_forecast_zarr("latlon"), from_anemoi=False, n_members=2)

    metrics.main(config)

    result = open_result(config, "fcrps.global")
    assert result["t2m"].dims == ("t0", "fhr")
    assert np.isfinite(result["t2m"]).all()


def test_main_regrids_forecast_and_target(config, write_inference):
    for t0 in DATES[:2]:
        write_inference(t0, offset=1.0)
    config.update(target_regrid_kwargs={"method": "target"}, forecast_regrid_kwargs={"method": "forecast"})

    with mock.patch.object(metrics, "horizontal_regrid", side_effect=identity_regrid) as regrid:
        metrics.main(config)

    methods = [c.kwargs["method"] for c in regrid.call_args_list]
    assert methods.count("forecast") == 2  # once per t0
    assert methods.count("target") == 3  # weights, then once per t0
    np.testing.assert_allclose(open_result(config, "rmse.global")["t2m"], 1.0)


def test_main_nested_global(config, write_inference):
    for t0 in DATES[:2]:
        write_inference(t0, offset=1.0)
    config.update(
        model_type="nested-global",
        lam_index=8,
        anemoi_reference_dataset_kwargs={"dataset": "ref.zarr"},
        forecast_regrid_kwargs={},
    )

    with (
        mock.patch.object(metrics, "prepare_regrid_target_mask", return_value=("grid.nc", None)) as prepare,
        mock.patch.object(data, "regrid_nested_to_latlon", side_effect=lambda xds, **kw: reshape_cell_to_latlon(xds)) as regrid,
    ):
        metrics.main(config)

    prepare.assert_called_once()
    assert regrid.call_args.kwargs["horizontal_regrid_kwargs"] == {"target_grid_path": "grid.nc"}
    np.testing.assert_allclose(open_result(config, "rmse.nested-global")["t2m"], 1.0)


class GatheringTopology(SerialTopology):
    """Serial, but gather returns one list per rank like MPITopology does."""

    def gather(self, local_array):
        return [local_array]


def test_main_flattens_gathered_results_with_mpi(config, write_inference):
    for t0 in DATES[:2]:
        write_inference(t0, offset=1.0)
    config.update(topo=GatheringTopology(), use_mpi=True)

    metrics.main(config)

    result = open_result(config, "rmse.global")
    np.testing.assert_array_equal(result["t0"], DATES[:2])
