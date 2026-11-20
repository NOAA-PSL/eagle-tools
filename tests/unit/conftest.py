"""Fixtures that write the synthetic data from ``synthetic.py`` to files eagle-tools reads."""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ufs2arco.mpi import SerialTopology

from synthetic import CELL_LAT, CELL_LON, DATES, LAT, LON, VARIABLES, truth, truth_at


@pytest.fixture
def anemoi_zarr(tmp_path):
    """Path to an anemoi-format zarr store holding the truth over ``DATES``."""
    data = truth(len(DATES))[:, :, None, :]  # add the ensemble dim
    ads = xr.Dataset(
        {
            "data": (("time", "variable", "ensemble", "cell"), data),
            "dates": ("time", DATES.values),
            "latitudes": ("cell", CELL_LAT),
            "longitudes": ("cell", CELL_LON),
        },
        # ufs2arco writes integer index coordinates for each dimension
        coords={dim: np.arange(n) for dim, n in zip(("time", "variable", "ensemble", "cell"), data.shape)},
        attrs={"variables": VARIABLES},
    )
    path = tmp_path / "verification.zarr"
    ads.to_zarr(path)
    return str(path)


@pytest.fixture
def write_inference(tmp_path):
    """Write anemoi-inference netcdf output, the truth plus ``offset``, and return its path."""
    forecast_path = tmp_path / "forecasts"
    forecast_path.mkdir()

    def write(t0, lead_time=12, offset=0.0, member=None):
        ids = truth_at(t0, lead_time)
        for key in VARIABLES:
            ids[key] = ids[key] + offset
        st0 = pd.Timestamp(t0).strftime("%Y-%m-%dT%H")
        suffix = "" if member is None else f".member{member:03d}"
        fname = forecast_path / f"{st0}.{lead_time}h{suffix}.nc"
        ids.to_netcdf(fname)
        return str(fname)

    write.path = str(forecast_path)
    return write


@pytest.fixture
def write_forecast_zarr(tmp_path):
    """Write a ufs2arco-style forecast zarr, dims (t0, fhr, member, [level], <grid>), and return its path.

    ``grid="latlon"`` uses the shared global grid, ``grid="xy"`` the same 4 x 4 values as a LAM grid.
    """

    def write(grid="latlon"):
        t0 = DATES[:2]
        fhr = np.array([0, 6, 12])
        member = np.arange(2)
        level = [500, 850]
        hdims = ("latitude", "longitude") if grid == "latlon" else ("y", "x")
        hcoords = {"latitude": LAT, "longitude": LON} if grid == "latlon" else {"y": np.arange(4), "x": np.arange(4)}
        shape = (len(t0), len(fhr), len(member), 4, 4)
        rng = np.random.default_rng(1)
        xds = xr.Dataset(
            {
                "t2m": (("t0", "fhr", "member", *hdims), rng.normal(size=shape)),
                "t": (
                    ("t0", "fhr", "member", "level", *hdims),
                    rng.normal(size=shape[:3] + (len(level),) + shape[3:]),
                ),
            },
            coords={
                "t0": t0,
                "fhr": fhr,
                "member": member,
                "level": level,
                **hcoords,
                "valid_time": (("t0", "fhr"), t0.values[:, None] + fhr * np.timedelta64(1, "h")),
            },
        )
        path = tmp_path / f"forecast.{grid}.zarr"
        xds.to_zarr(path)
        return str(path)

    return write


@pytest.fixture
def serial_config(tmp_path):
    """Config as returned by ``eagle.tools.utils.setup``, for a serial run."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return {
        "topo": SerialTopology(),
        "use_mpi": False,
        "output_path": str(output_path),
    }


@pytest.fixture(autouse=True)
def restore_loggers():
    """Undo the handlers/levels that setup_simple_log and init_topo attach."""
    loggers = [logging.getLogger(name) for name in ("eagle.tools", "ufs2arco")]
    saved = [(lg, list(lg.handlers), lg.level, lg.propagate) for lg in loggers]
    yield
    for lg, handlers, level, propagate in saved:
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate
