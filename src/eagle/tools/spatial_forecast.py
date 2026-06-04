import logging
from math import ceil

import numpy as np
import xarray as xr
import pandas as pd

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import get_gridcell_area_weights
from eagle.tools.nested import prepare_regrid_target_mask
from eagle.tools.spatial import rmse, mae

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

    fc1_config = config["forecast1"]
    fc2_config = config["forecast2"]

    _setup_nested_global(fc1_config)
    _setup_nested_global(fc2_config)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = ceil(n_dates / topo.size)

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

    logger.info("Done Storing Spatial Error Metrics")
