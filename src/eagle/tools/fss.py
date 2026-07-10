import logging
import os

import numpy as np
from scipy import ndimage
import xarray as xr
import pandas as pd

from eagle.tools.data import (
    open_anemoi_inference_dataset,
    open_forecast_zarr_dataset,
    trim_xarray_edge,
)

logger = logging.getLogger("eagle.tools")

# Common name given to the precip variable in both datasets so they can be
# compared directly regardless of their original variable names.
PRECIP = "precip"


def _uniform_filter_lastdims(arr, size):
    """Apply a square uniform (box) filter over the trailing two axes only.

    ``scipy.ndimage.uniform_filter`` is separable and runs in O(N) regardless
    of window size, which is what makes FSS cheap for large neighborhoods. We
    filter the trailing ``(y, x)`` axes and leave any leading broadcast axes
    (fhr, threshold) untouched via a size of 1 on those axes.
    """
    fsize = (1,) * (arr.ndim - 2) + (size, size)
    return ndimage.uniform_filter(arr, size=fsize, mode="constant", cval=0.0)


def _box_mean(da, size):
    """Neighborhood (box) mean over the (y, x) core dims, keeping other dims."""
    return xr.apply_ufunc(
        _uniform_filter_lastdims,
        da,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        kwargs={"size": size},
    )


def _neighborhood_fraction(binary, valid, size):
    """Mask-aware fraction of valid points exceeding the threshold in each window.

    ``uniform_filter`` normalizes by the full window count, so dividing the
    filtered ``binary*valid`` by the filtered ``valid`` recovers
    (exceedances / valid points) within each window. This shrinks the window at
    the domain edge and around any NaNs, matching Roberts & Lean's in-domain
    fraction. Windows with no valid points become NaN and are skipped later.

    ``den`` is a fraction in [0, 1]; the smallest *legitimate* nonzero value is
    ``1 / size**2`` (a single valid cell in the window). ``uniform_filter``
    leaves ~1e-15 roundoff where the true value is zero, so we threshold at half
    the smallest real fraction. A plain ``den > 0`` guard would instead let tiny
    positive roundoff through and divide near-zero ``num`` by it, yielding wild
    (even negative) "fractions" that corrupt the score.
    """
    num = _box_mean(binary * valid, size)
    den = _box_mean(valid, size)
    eps = 0.5 / (size * size)
    return num / den.where(den > eps)


def fractions_skill_score(fcst, obs, thresholds, radii_gp):
    """Deterministic Fractions Skill Score (Roberts & Lean, 2008).

    Args:
        fcst, obs (xr.DataArray): forecast and verification fields with dims
            ``(fhr, y, x)``.
        thresholds (Sequence[float]): exceedance thresholds (same units as the fields).
        radii_gp (Sequence[tuple[float, int]]): ``(radius_km, window_size)`` pairs,
            where ``window_size = 2*n+1`` grid points.

    Returns:
        xr.Dataset: ``fss``, ``mse``, ``mse_ref`` with dims ``(fhr, threshold, radius)``.
    """
    txda = xr.DataArray(
        list(thresholds), dims="threshold", coords={"threshold": list(thresholds)}
    )
    valid = (np.isfinite(fcst) & np.isfinite(obs)).astype(float)
    fbin = ((fcst >= txda) & (valid > 0)).astype(float)
    obin = ((obs >= txda) & (valid > 0)).astype(float)

    per_radius = []
    for radius_km, size in radii_gp:
        M = _neighborhood_fraction(fbin, valid, size)
        O = _neighborhood_fraction(obin, valid, size)

        mse = ((M - O) ** 2).mean(("y", "x"))
        mse_ref = (M**2 + O**2).mean(("y", "x"))
        fss = 1 - mse / mse_ref.where(mse_ref > 0)

        ds = xr.Dataset({"fss": fss, "mse": mse, "mse_ref": mse_ref})
        ds = ds.expand_dims(radius=[radius_km])
        per_radius.append(ds)

    return xr.concat(per_radius, dim="radius")


#: Order in which the per-axis ``trim_edge`` dict is flattened for
#: ``trim_xarray_edge``. AORC/mask arrays are stored ``(..., y, x)``.
STACK_ORDER = ("y", "x")


def _open_verification(path, precip_varname, lcc_info, trim_edge=None):
    """Open the gridded verification dataset (e.g. AORC), select & rename precip.

    AORC is a plain gridded zarr with ``(time, y, x)`` dims (not an anemoi
    dataset), so it is opened directly and trimmed via ``trim_xarray_edge``.
    """
    xds = xr.open_zarr(path)
    if precip_varname is None:
        data_vars = list(xds.data_vars)
        assert len(data_vars) == 1, (
            f"verification precip_varname not set and {path} has multiple "
            f"data_vars: {data_vars}"
        )
        precip_varname = data_vars[0]
    xds = xds[[precip_varname]].rename({precip_varname: PRECIP})
    if trim_edge is not None:
        xds = trim_xarray_edge(
            xds,
            lcc_info=lcc_info,
            trim_edge=trim_edge,
            stack_order=list(STACK_ORDER),
        )
    return xds


def _open_mask(path, varname, lcc_info, trim_edge=None):
    """Open the verification validity mask, trim it, and binarize.

    The mask is a static ``(y, x)`` field flagging where the verification
    dataset is valid. It is stored with float/bool values; ``mask > 0`` gives
    the boolean "valid" field. It lives on the untrimmed verification grid, so
    the same ``trim_edge`` is applied to co-register it (and reset its coords to
    start at 0) to match the trimmed forecast/verification fields.
    """
    xds = xr.open_dataset(path)[[varname]]
    if trim_edge is not None:
        xds = trim_xarray_edge(
            xds,
            lcc_info=lcc_info,
            trim_edge=trim_edge,
            stack_order=list(STACK_ORDER),
        )
    return xds[varname] > 0


def _radii_in_gridpoints(radius, grid_spacing_km):
    """Convert (half-width) radii in km to ``(radius_km, window_size)`` pairs.

    ``radius`` may be a scalar or a list. ``n = round(radius / dx)`` and the
    square window side is ``2*n + 1`` grid points.
    """
    radii = radius if isinstance(radius, (list, tuple)) else [radius]
    out = []
    for r in radii:
        n = int(round(r / grid_spacing_km))
        out.append((r, 2 * n + 1))
    return out


def main(config):
    """Compute the deterministic Fractions Skill Score.

    See ``eagle-tools fss --help`` or cli.py for help
    """
    if isinstance(config, str):
        from eagle.tools.utils import setup

        config = setup(config, "fss")

    topo = config["topo"]

    vcfg = config["verification_dataset"]
    fcfg = config["forecast_dataset"]

    model_type = fcfg["model_type"]
    fcst_precip = fcfg["precip_varname"]
    lam_index = fcfg.get("lam_index", None)
    lcc_info = fcfg.get("lcc_info", None)
    from_anemoi = fcfg.get("from_anemoi", True)
    trim_forecast_edge = fcfg.get("trim_edge", None)

    forecast_hours = config["forecast_hours"]
    thresholds = config["thresholds"]
    grid_spacing_km = config["grid_spacing_km"]
    radii_gp = _radii_in_gridpoints(config["radius"], grid_spacing_km)

    logger.info(
        f"FSS setup: thresholds={thresholds}, "
        f"radii (km, window)={radii_gp}, forecast_hours={forecast_hours}"
    )

    # Verification dataset opened once, lazily
    vds = _open_verification(
        path=vcfg["path"],
        precip_varname=vcfg.get("precip_varname", None),
        lcc_info=lcc_info,
        trim_edge=vcfg.get("trim_edge", None),
    )

    # Optional validity mask (where the verification dataset is valid). Applied
    # to the forecast so invalid points are ignored in every comparison.
    mask = None
    if vcfg.get("mask", None) is not None:
        mcfg = vcfg["mask"]
        mask = _open_mask(
            path=mcfg["path"],
            varname=mcfg["varname"],
            lcc_info=lcc_info,
            trim_edge=vcfg.get("trim_edge", None),
        ).load()
        logger.info(
            f"Loaded verification mask (valid fraction={float(mask.mean()):.3f})"
        )

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    container = []

    logger.info("Computing Fractions Skill Score")
    logger.info(f"Initial Conditions:\n{dates}")
    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break  # last batch situation

        t0 = dates[date_idx]
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        # Load forecast, reshaped to (time, y, x). Either an anemoi inference
        # NetCDF (one file per initial condition) or a forecast zarr store
        # holding all initial conditions along a t0 dim.
        if from_anemoi:
            fname = f"{config['forecast_path']}/{st0}.{config['lead_time']}h.nc"
            fds = open_anemoi_inference_dataset(
                fname,
                model_type=model_type,
                lam_index=lam_index,
                lcc_info=lcc_info,
                trim_edge=trim_forecast_edge,
                vars_of_interest=[fcst_precip],
                reshape_cell_to_2d=True,
                load=True,
            )
        else:
            # A single forecast zarr, or (when lead_time is a list) one store
            # per lead time whose paths come from a ``{fhr:02d}`` template and
            # are concatenated along fhr.
            lead_time = config["lead_time"]
            if isinstance(lead_time, (list, tuple)):
                fpath = [
                    os.path.expandvars(config["forecast_path"].format(fhr=h))
                    for h in lead_time
                ]
            else:
                fpath = config["forecast_path"]
            fds = open_forecast_zarr_dataset(
                fpath,
                t0=t0,
                vars_of_interest=[fcst_precip],
                trim_edge=trim_forecast_edge,
                reshape_cell_to_2d=True,
                lcc_info=lcc_info,
                load=True,
            )
        fds = fds.rename({fcst_precip: PRECIP})

        # Select the requested lead times and matching verification valid times
        target_times = [t0 + pd.Timedelta(hours=h) for h in forecast_hours]
        fcst = fds[PRECIP].sel(time=target_times)
        obs = vds[PRECIP].sel(time=target_times).load()

        assert fcst.sizes["y"] == obs.sizes["y"] and fcst.sizes["x"] == obs.sizes["x"], (
            f"Forecast (y,x)=({fcst.sizes['y']},{fcst.sizes['x']}) and verification "
            f"(y,x)=({obs.sizes['y']},{obs.sizes['x']}) grids do not match"
        )

        # Mask out points where the verification is invalid, so they are
        # excluded from the neighborhood fractions (via the finite check).
        if mask is not None:
            assert mask.sizes["y"] == fcst.sizes["y"] and mask.sizes["x"] == fcst.sizes["x"], (
                f"Mask (y,x)=({mask.sizes['y']},{mask.sizes['x']}) and forecast "
                f"(y,x)=({fcst.sizes['y']},{fcst.sizes['x']}) grids do not match"
            )
            fcst = fcst.where(mask)

        # Replace the datetime time dim with integer forecast hour
        fhr = np.array(forecast_hours, dtype=int)
        fcst = fcst.rename({"time": "fhr"}).assign_coords(fhr=fhr)
        obs = obs.rename({"time": "fhr"}).assign_coords(fhr=fhr)

        result = fractions_skill_score(fcst, obs, thresholds, radii_gp)

        # Per-IC coords: keep t0 so the full (t0, fhr, threshold, radius) sample
        # is retained for aggregation and significance testing downstream.
        result = result.expand_dims(t0=[t0])
        result = result.assign_coords(
            lead_time=("fhr", (fhr * np.timedelta64(1, "h")).astype("timedelta64[ns]"))
        )
        container.append(result)

        logger.info(f"Done with {st0}")
    logger.info("Done Computing FSS")

    logger.info("Gathering Results on Root Process")
    container = topo.gather(container)

    if topo.is_root:
        logger.info("Combining & Storing Results")
        if config["use_mpi"]:
            container = [xds for sublist in container for xds in sublist]
        container = sorted(container, key=lambda xds: xds.coords["t0"].values)
        result = xr.concat(container, dim="t0")
        fname = f"{config['output_path']}/fss.{model_type}.nc"
        result.to_netcdf(fname)
        logger.info(f"Stored result: {fname}")
        logger.info("Done Storing FSS")
