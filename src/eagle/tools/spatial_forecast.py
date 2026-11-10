import logging
from math import ceil

import numpy as np
import xarray as xr
import pandas as pd

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import get_gridcell_area_weights
from eagle.tools.nested import prepare_regrid_target_mask
from eagle.tools.spatial import rmse, mae, signed_difference

logger = logging.getLogger("eagle.tools")


def _sum_dataset_across_ranks(topo, ds):
    """Sum a Dataset across MPI ranks with buffer-based Reduce (no pickling).

    ``comm.gather`` pickles the whole object into a single message, which overflows
    MPI's size limit for full spatial maps. Reducing one variable at a time over raw
    numpy buffers keeps each message's element count well under the limit and avoids
    pickling entirely.

    Returns the summed Dataset on root and None elsewhere. Rank 0 always holds a valid
    Dataset (it processes the first date); ranks that drew no dates may pass None and
    contribute zeros.
    """
    from mpi4py import MPI

    comm = topo.comm
    # Broadcast per-variable specs from root so any data-less rank can zero-fill.
    specs = None
    if topo.is_root:
        specs = [(v, ds[v].dims, ds[v].shape, ds[v].values.dtype.str) for v in ds.data_vars]
    specs = comm.bcast(specs, root=topo.root)

    out = {}
    for name, dims, shape, dtype in specs:
        if ds is not None:
            local = np.ascontiguousarray(ds[name].values, dtype=dtype)
        else:
            local = np.zeros(shape, dtype=dtype)
        recv = np.empty(shape, dtype=dtype) if topo.is_root else None
        comm.Reduce(local, recv, op=MPI.SUM, root=topo.root)
        if topo.is_root:
            out[name] = (dims, recv)

    return xr.Dataset(out, coords=ds.coords) if topo.is_root else None


def _setup_nested_global(fc_config):
    """Resolve the regrid target grid path (with mask) for nested-global forecasts."""
    if fc_config.get("model_type") == "nested-global":
        fc_config["horizontal_regrid_kwargs"]["target_grid_path"], _ = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=fc_config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=fc_config["horizontal_regrid_kwargs"],
        )


def _truncate_lead_time(fds, max_fhr):
    """Keep only forecast steps within the first ``max_fhr`` hours of lead time."""
    if max_fhr is None:
        return fds
    t_start = fds["time"].values[0]
    t_stop = t_start + np.timedelta64(int(max_fhr), "h")
    return fds.sel(time=slice(t_start, t_stop))


def _init_diff_zarr_template(zarr_path, d_example, all_t0):
    """Write a lazy zarr template with a leading ``t0`` dimension.

    ``d_example`` is one IC's signed-difference Dataset (dims ``fhr, level, y, x``),
    which fixes the per-IC structure (vars, shapes, dtypes, and the shared
    ``fhr``/``level``/``latitude``/``longitude``/``lead_time`` coords). Each rank
    later fills its own ``t0`` slices via region writes, so nothing is gathered.
    """
    import dask.array as da

    nt = len(all_t0)
    template_vars = {}
    for name, davar in d_example.data_vars.items():
        shape = (nt,) + davar.shape
        chunks = (1,) + davar.shape
        template_vars[name] = (
            ("t0",) + davar.dims,
            da.zeros(shape, chunks=chunks, dtype=davar.dtype),
        )
    template = xr.Dataset(template_vars, coords=dict(d_example.coords))
    template = template.assign_coords(t0=("t0", np.asarray(all_t0, dtype="datetime64[ns]")))
    template.to_zarr(zarr_path, mode="w", compute=False)
    logger.info(f"Initialized per-IC difference zarr template: {zarr_path} (t0={nt})")


def _write_diff_region(zarr_path, d, t0, date_idx):
    """Write one IC's signed-difference map into its ``t0`` slice of the zarr."""
    d_ic = d.expand_dims(t0=[np.datetime64(t0, "ns")])
    # Region writes must not re-write coords that lack the region dim.
    drop = [c for c in d_ic.coords if "t0" not in d_ic[c].dims]
    d_ic = d_ic.drop_vars(drop)
    d_ic.to_zarr(zarr_path, region={"t0": slice(date_idx, date_idx + 1)})


def decompose_from_zarr(zarr_path, groupby=None):
    """Bias/variance/RMSE decomposition from a per-IC signed-difference zarr.

    Parameters
    ----------
    zarr_path : str
        Path written by the ``decompose`` path of :func:`main`.
    groupby : None or str
        How to bin the initial conditions before decomposing:

        - ``None``      : decompose over all ICs (dims ``fhr, level, y, x``).
        - ``"init_hour"``: bin by the UTC hour of ``t0`` (keeps ``fhr``). With a
          fixed ``fhr`` this is equivalent to binning by valid hour, so this is
          the clean way to see a diurnal cycle without mixing lead times.
        - ``"valid_hour"``: bin by the UTC hour of the valid time (``t0`` +
          ``lead_time``), collapsing ``fhr`` into the diurnal bin.
        - ``"season"`` / ``"month"``: bin by the season/month of ``t0``.

    Returns
    -------
    xarray.Dataset with ``*_bias``, ``*_variance`` and ``*_rmse`` for each var.
    """
    ds = xr.open_zarr(zarr_path)

    if groupby in (None, "init_hour"):
        d = ds
        sample_dim = "t0"
        if groupby == "init_hour":
            d = d.assign_coords(init_hour=ds["t0"].dt.hour)
            d = d.groupby("init_hour")
    elif groupby == "valid_hour":
        d = ds.stack(sample=("t0", "fhr"))
        valid = d["t0"] + d["lead_time"]
        # Materialize the labels: xarray cannot groupby a chunked coordinate.
        valid_hour = np.asarray(valid.dt.hour.values)
        d = d.assign_coords(valid_hour=("sample", valid_hour))
        d = d.groupby("valid_hour")
        sample_dim = "sample"
    elif groupby in ("season", "month"):
        d = ds.assign_coords(**{groupby: getattr(ds["t0"].dt, groupby)})
        d = d.groupby(groupby)
        sample_dim = "t0"
    else:
        raise ValueError(f"Unknown groupby={groupby!r}")

    bias = d.mean(sample_dim)
    var = d.var(sample_dim)
    out = {}
    for name in (bias.data_vars if hasattr(bias, "data_vars") else ds.data_vars):
        out[f"{name}_bias"] = bias[name]
        out[f"{name}_variance"] = var[name]
        out[f"{name}_rmse"] = np.sqrt(bias[name] ** 2 + var[name])
    return xr.Dataset(out)


def _open_forecast(fc_config, st0, t0, lead_time, levels, vars_of_interest):
    model_type = fc_config["model_type"]
    subsample_kwargs = {
        "levels": levels,
        "vars_of_interest": vars_of_interest,
        "lcc_info": fc_config.get("lcc_info", None),
    }
    if fc_config.get("from_anemoi", True):
        return open_anemoi_inference_dataset(
            f"{fc_config['path']}/{st0}.{lead_time}h.nc",
            model_type=model_type,
            lam_index=fc_config.get("lam_index", None),
            trim_edge=fc_config.get("trim_edge", None),
            load=True,
            reshape_cell_to_2d=True,
            horizontal_regrid_kwargs=fc_config.get("horizontal_regrid_kwargs", None),
            **subsample_kwargs,
        )
    else:
        return open_forecast_zarr_dataset(
            fc_config["path"],
            t0=t0,
            trim_edge=fc_config.get("trim_edge", None),
            load=True,
            reshape_cell_to_2d=True,
            **subsample_kwargs,
        )


def main(config):
    """Compute spatial maps of RMSE and MAE between two forecast datasets.

    See ``eagle-tools spatial-forecast --help`` or cli.py for help
    """
    if isinstance(config, str):
        from eagle.tools.utils import setup
        config = setup(config, "spatial_forecast")

    topo = config["topo"]
    use_mpi = config["use_mpi"]
    keep_t0 = config.get("keep_t0", False)
    levels = config.get("levels", None)
    vars_of_interest = config.get("vars_of_interest", None)
    lead_time = config["lead_time"]

    if keep_t0 and use_mpi:
        raise NotImplementedError(
            "keep_t0=True is not supported with MPI: the per-IC spatial maps are too "
            "large to gather across ranks. Run serially (use_mpi=False) for keep_t0=True."
        )

    # Optional bias/variance decomposition (see decompose_from_zarr for analysis).
    # Stores a per-IC signed-difference zarr (truncated to the first `max_fhr` of
    # lead time) plus global bias/variance/rmse maps. Region-based zarr writes
    # keep per-rank memory low and are MPI-safe (no cross-rank gather).
    decompose_config = config.get("decompose", None)
    decompose = bool(decompose_config) and decompose_config.get("enabled", True)
    if decompose:
        decompose_max_fhr = decompose_config.get("max_fhr", 48)
        decompose_zarr = decompose_config.get(
            "zarr_path", f"{config['output_path']}/diff.perIC.fc1vfc2.zarr"
        )

    fc1_config = config["forecast1"]
    fc2_config = config["forecast2"]

    _setup_nested_global(fc1_config)
    _setup_nested_global(fc2_config)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = ceil(n_dates / topo.size)

    # Pre-pass: root opens the first IC to fix the per-IC structure and writes the
    # lazy zarr template, then all ranks proceed to fill their own t0 regions.
    if decompose:
        if topo.is_root:
            _st0 = dates[0].strftime("%Y-%m-%dT%H")
            _f1 = _open_forecast(fc1_config, _st0, dates[0], lead_time, levels, vars_of_interest)
            _f2 = _open_forecast(fc2_config, _st0, dates[0], lead_time, levels, vars_of_interest)
            _f2 = _f2.sel(time=_f1.time.values)
            _f1 = _truncate_lead_time(_f1, decompose_max_fhr)
            _f2 = _truncate_lead_time(_f2, decompose_max_fhr)
            if "member" not in _f1.dims:
                _f1 = _f1.expand_dims("member")
            if "member" not in _f2.dims:
                _f2 = _f2.expand_dims("member")
            _d = signed_difference(target=_f1, prediction=_f2)
            _init_diff_zarr_template(decompose_zarr, _d, dates)
            del _f1, _f2, _d
        if use_mpi:
            topo.comm.barrier()

    # keep_t0=True: list of per-IC maps (serial only).
    # keep_t0=False: running partial sum, pre-divided by the global n_dates so that
    #   summing each rank's contribution across ranks yields the full average. The
    #   cross-rank sum uses buffer-based MPI Reduce (see _sum_dataset_across_ranks).
    if keep_t0:
        rmse_container = []
        mae_container = []
    else:
        rmse_container = None
        mae_container = None
    latlon_weights = None

    # Running moment sums for the global decomposition: S1 = sum_t0[d],
    # S2 = sum_t0[d**2]. Summed across ranks at the end (buffer-based Reduce).
    diff_s1 = None
    diff_s2 = None

    logger.info("Computing Spatial Error Metrics between two forecasts")
    logger.info(f"Initial Conditions:\n{dates}")

    for batch_idx in range(n_batches):
        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break

        t0 = dates[date_idx]
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        fds1 = _open_forecast(fc1_config, st0, t0, lead_time, levels, vars_of_interest)
        fds2 = _open_forecast(fc2_config, st0, t0, lead_time, levels, vars_of_interest)
        fds2 = fds2.sel(time=fds1.time.values)

        if "member" not in fds1.dims:
            fds1 = fds1.expand_dims("member")
        if "member" not in fds2.dims:
            fds2 = fds2.expand_dims("member")

        if latlon_weights is None:
            latlon_weights = get_gridcell_area_weights(
                fds1, fc1_config["model_type"], reshape_cell_to_2d=True,
            )

        if decompose:
            d1 = _truncate_lead_time(fds1, decompose_max_fhr)
            d2 = _truncate_lead_time(fds2, decompose_max_fhr)
            this_diff = signed_difference(target=d1, prediction=d2)
            _write_diff_region(decompose_zarr, this_diff, t0, date_idx)
            this_diff2 = this_diff**2
            if diff_s1 is None:
                diff_s1 = this_diff
                diff_s2 = this_diff2
            else:
                diff_s1 += this_diff
                diff_s2 += this_diff2

        this_rmse = rmse(target=fds1, prediction=fds2, weights=latlon_weights, keep_t0=keep_t0)
        this_mae = mae(target=fds1, prediction=fds2, weights=latlon_weights, keep_t0=keep_t0)

        if keep_t0:
            rmse_container.append(this_rmse)
            mae_container.append(this_mae)
        else:
            if rmse_container is None:
                rmse_container = this_rmse / n_dates
                mae_container = this_mae / n_dates
            else:
                rmse_container += this_rmse / n_dates
                mae_container += this_mae / n_dates

        logger.info(f"Done with {st0}")

    logger.info("Done Computing Metrics")
    logger.info("Combining & Storing Results")
    model_label = f"{fc1_config['model_type']}v{fc2_config['model_type']}"

    if keep_t0:
        # Serial only (guarded above). Concatenate the per-IC maps along t0.
        for varname, container in zip(["rmse", "mae"], [rmse_container, mae_container]):
            container = sorted(container, key=lambda xds: xds.coords["t0"])
            xda = xr.concat(container, dim="t0")
            fname = f"{config['output_path']}/spatial.{varname}.perIC.{model_label}.fc1vfc2.nc"
            xda.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")

    else:
        # Each rank holds a partial sum (pre-divided by n_dates). Sum across ranks
        # via buffer-based Reduce; serial runs already hold the full average.
        if use_mpi:
            xda_rmse = _sum_dataset_across_ranks(topo, rmse_container)
            xda_mae = _sum_dataset_across_ranks(topo, mae_container)
        else:
            xda_rmse = rmse_container
            xda_mae = mae_container

        if topo.is_root:
            for varname, xda in zip(["rmse", "mae"], [xda_rmse, xda_mae]):
                fname = f"{config['output_path']}/spatial.{varname}.{model_label}.fc1vfc2.nc"
                xda.to_netcdf(fname)
                logger.info(f"Stored result: {fname}")

    if decompose:
        # Sum the moment partials across ranks, then form global bias/variance/rmse.
        if use_mpi:
            s1 = _sum_dataset_across_ranks(topo, diff_s1)
            s2 = _sum_dataset_across_ranks(topo, diff_s2)
        else:
            s1, s2 = diff_s1, diff_s2

        if topo.is_root:
            bias = s1 / n_dates
            mse = s2 / n_dates
            var = (mse - bias**2).clip(min=0.0)  # guard tiny negatives from roundoff
            for varname, xda in zip(["bias", "variance", "rmse"], [bias, var, np.sqrt(mse)]):
                fname = f"{config['output_path']}/spatial.diff.{varname}.{model_label}.fc1vfc2.nc"
                xda.to_netcdf(fname)
                logger.info(f"Stored result: {fname}")
            logger.info(f"Per-IC signed differences stored: {decompose_zarr}")

    logger.info("Done Storing Spatial Error Metrics")
