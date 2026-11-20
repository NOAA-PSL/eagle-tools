"""Tiny synthetic datasets shaped like the data eagle-tools reads.

Everything lives on the same 4 x 4 global grid, with latitude descending as in
anemoi datasets, so truth and forecasts line up cell for cell.
"""

import numpy as np
import pandas as pd
import xarray as xr

LAT = np.array([60.0, 20.0, -20.0, -60.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])
CELL_LAT = np.repeat(LAT, len(LON))
CELL_LON = np.tile(LON, len(LAT))
N_CELL = len(CELL_LAT)

# Verification period; forecasts below must fall inside it
DATES = pd.date_range("2024-01-01T00", "2024-01-02T00", freq="6h")

# anemoi-style flattened names: "t_500" -> variable "t" at level 500
VARIABLES = ["cos_julian_day", "t2m", "t_500", "t_850"]


def truth(n_time, n_var=len(VARIABLES)):
    """Deterministic, cell- and time-varying values."""
    return np.random.default_rng(0).normal(280.0, 5.0, size=(n_time, n_var, N_CELL))


def make_cell_ds(n_time=3, member=None, varnames=("t2m",), value=None):
    """In-memory ``[time, (member), cell]`` dataset on the shared grid."""
    dims = ("time", "member", "cell") if member else ("time", "cell")
    shape = (n_time, member, N_CELL) if member else (n_time, N_CELL)
    rng = np.random.default_rng(0)
    data_vars = {
        name: (dims, np.full(shape, value) if value is not None else rng.normal(size=shape))
        for name in varnames
    }
    coords = {
        "time": DATES[:n_time],
        "cell": np.arange(N_CELL),
        "latitude": ("cell", CELL_LAT),
        "longitude": ("cell", CELL_LON),
    }
    if member:
        coords["member"] = np.arange(member)
    return xr.Dataset(data_vars, coords=coords)


def truth_at(t0, lead_time):
    """Truth from ``anemoi_zarr`` for t0 -> t0 + lead_time, as anemoi-inference names it."""
    times = pd.date_range(t0, pd.Timestamp(t0) + pd.Timedelta(hours=lead_time), freq="6h")
    itime = DATES.get_indexer(times)
    data = truth(len(DATES))[itime]
    return xr.Dataset(
        {name: (("time", "values"), data[:, i]) for i, name in enumerate(VARIABLES)},
        coords={
            "time": times,
            "latitude": ("values", CELL_LAT),
            "longitude": ("values", CELL_LON),
        },
    )
