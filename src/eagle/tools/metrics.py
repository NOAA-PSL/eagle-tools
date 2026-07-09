import logging

import numpy as np
from scipy.spatial import SphericalVoronoi
import xarray as xr
import pandas as pd

import ufs2arco.utils
from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

from eagle.tools.data import open_anemoi_dataset_with_xarray, open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.reshape import flatten_to_cell
from eagle.tools.reshape import reshape_cell_to_latlon
from eagle.tools.reshape import reshape_cell_dim
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")


def get_gridcell_area_weights(xds, model_type, reshape_cell_to_2d=False, regrid_kwargs=None):

    if "global" in model_type:
        weights = _area_weights(xds, reshape_cell_to_2d=reshape_cell_to_2d)
        if regrid_kwargs is not None:
            weights = horizontal_regrid(weights.to_dataset(name="weights"), **regrid_kwargs)["weights"]

        return weights


    elif model_type in ("lam", "nested-lam"):
        return 1. # Assume LAM is equal area

    else:
        raise NotImplementedError


def _area_weights(xds, unit_mean=True, radius=1, center=np.array([0,0,0]), threshold=1e-12, reshape_cell_to_2d=False):
    """This is a nice code block copied from anemoi-graphs"""

    cds = xds.coords.to_dataset().copy()
    if "cell" not in cds["latitude"].dims:
        cds = flatten_to_cell(cds)

    x = radius * np.cos(np.deg2rad(cds["latitude"])) * np.cos(np.deg2rad(cds["longitude"]))
    y = radius * np.cos(np.deg2rad(cds["latitude"])) * np.sin(np.deg2rad(cds["longitude"]))
    z = radius * np.sin(np.deg2rad(cds["latitude"]))
    sv = SphericalVoronoi(
        points=np.stack([x,y,z], -1),
        radius=radius,
        center=center,
        threshold=threshold,
    )
    area_weight = sv.calculate_areas()
    if unit_mean:
        area_weight /= area_weight.mean()

    area_weight = xr.DataArray(area_weight, coords=cds.cell.coords)
    if reshape_cell_to_2d:
        try:
            ads = reshape_cell_to_latlon(area_weight.to_dataset(name="weights"))
            area_weight = ads["weights"]
        except:
            logger.warning("Could not reshape area weights to lat/lon")
    return area_weight


def _parse_subregions(config):
    """Parse subregion definitions from config.

    Returns dict of {name: {"latitude": (min, max), "longitude": (min, max)}}.
    Longitude bounds are accepted in [-180, 180] and converted to [0, 360] to
    match the forecast/verification grid convention.
    """
    raw = config.get("subregions", {})
    subregions = {}
    for name, bounds in raw.items():
        if "latitude" not in bounds and "longitude" not in bounds:
            raise ValueError(f"Subregion '{name}' must have at least 'latitude' or 'longitude'")
        lat = tuple(bounds["latitude"]) if "latitude" in bounds else (-90, 90)
        lon = bounds.get("longitude", [0, 359.99])
        lon = tuple(ll % 360 for ll in lon)
        subregions[name] = {"latitude": lat, "longitude": lon}
    return subregions


def _subregion_mask(latitude, longitude, bounds):
    """Boolean DataArray selecting grid cells within geographic bounds.

    Handles longitude wrapping around 0/360.
    """
    lat_min, lat_max = bounds["latitude"]
    lon_min, lon_max = bounds["longitude"]
    lat_mask = (latitude >= lat_min) & (latitude <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (longitude >= lon_min) & (longitude <= lon_max)
    else:
        lon_mask = (longitude >= lon_min) | (longitude <= lon_max)
    return lat_mask & lon_mask


def _get_latlon(weights, fallback_ds):
    """Get latitude/longitude coords from the weights array, else the dataset.

    For equal-area (LAM) grids ``weights`` is a scalar, so fall back to the
    dataset's coordinates.
    """
    if isinstance(weights, xr.DataArray) and "latitude" in weights.coords:
        return weights["latitude"], weights["longitude"]
    return fallback_ds["latitude"], fallback_ds["longitude"]


def _subregion_weights(weights, fallback_ds, bounds):
    """Mask the area weights to a subregion and renormalize to unit mean.

    Renormalizing so the masked weights have unit mean *over the region* lets
    the existing metric functions' ``.mean()`` produce the correct
    area-weighted mean over the subregion (NaNs outside are skipped).
    """
    latitude, longitude = _get_latlon(weights, fallback_ds)
    mask = _subregion_mask(latitude, longitude, bounds)
    w_sub = xr.where(mask, weights, np.nan)
    return w_sub / w_sub.mean()


def postprocess(xds):

    t0 = pd.Timestamp(xds["time"][0].values)
    xds["t0"] = xr.DataArray(t0, coords={"t0": t0})
    xds = xds.set_coords("t0")
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds["lead_time"].attrs = {} # remove any calendar details from the attributes
    xds["fhr"] = xr.DataArray(
        xds["lead_time"].values.astype("timedelta64[h]").astype(int),
        coords=xds.time.coords,
        attrs={"description": "forecast hour, aka lead time in hours"},
    )
    xds = xds.swap_dims({"time": "fhr"}).drop_vars("time")
    xds = xds.set_coords("lead_time")
    return xds


def rmse(target, prediction, weights=1., spatial_dims=("cell",)):
    result = {}
    dims = tuple(d for d in target.dims if d not in ("time", "level"))
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean(dims)
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def mae(target, prediction, weights=1., spatial_dims=("cell",)):
    result = {}
    dims = tuple(d for d in target.dims if d not in ("time", "level"))
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean(dims)
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def spread(ensemble, weights=1.):
    """Area-weighted ensemble spread (std dev over member dim)."""
    result = {}
    dims = tuple(d for d in ensemble.dims if d not in ("time", "level", "member"))
    for key in ensemble.data_vars:
        std = ensemble[key].std("member")
        weighted_std = weights * std
        result[key] = weighted_std.mean(dims).compute()
    xds = xr.Dataset(result)
    return postprocess(xds)


def fcrps(target, ensemble, weights=1.):
    """Area-weighted fair Continuous Ranked Probability Score.

    fCRPS = (1/N) * Σ|u_e - u*| - 1/(2*N*(N-1)) * ΣΣ|u_e - u_i|
    """
    result = {}
    n_members = ensemble.sizes["member"]
    dims = tuple(d for d in ensemble.dims if d not in ("time", "level", "member"))
    for key in ensemble.data_vars:
        abs_err = np.abs(ensemble[key] - target[key].squeeze("member")).mean("member")
        pairwise = np.abs(
            ensemble[key] - ensemble[key].rename({"member": "_member"})
        ).mean(("member", "_member"))
        fair_crps = weights * (abs_err - pairwise / (2 * (n_members - 1)))
        result[key] = fair_crps.mean(dims).compute()
    xds = xr.Dataset(result)
    return postprocess(xds)


def main(config):
    """Compute grid cell area weighted RMSE and MAE.

    See ``eagle-tools metrics --help`` or cli.py for help
    """
    if isinstance(config, str):
        from eagle.tools.utils import setup
        config = setup(config, "metrics")

    topo = config["topo"]

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "lcc_info": config.get("lcc_info", None),
    }
    n_members = config.get("n_members", 1)
    is_ensemble = n_members > 1

    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = (target_regrid_kwargs is not None) or \
            ((forecast_regrid_kwargs is not None) and (model_type != "nested-global"))
    mkw = {}
    if do_any_regridding:
        mkw["spatial_dims"] = ("latitude", "longitude")

    if model_type == "nested-global":
        forecast_regrid_kwargs["target_grid_path"], _ = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=forecast_regrid_kwargs,
        )

    # Verification dataset
    vds = open_anemoi_dataset_with_xarray(
        path=config["verification_dataset_path"],
        model_type=model_type,
        trim_edge=config.get("trim_edge", None),
        **subsample_kwargs,
    )

    # Area weights
    latlon_weights = get_gridcell_area_weights(
        vds,
        model_type,
        reshape_cell_to_2d=do_any_regridding,
        regrid_kwargs=target_regrid_kwargs,
    )

    # Subregions: "global" (full field, no suffix) plus any user-defined regions
    subregions = _parse_subregions(config)
    region_names = ["global"] + list(subregions.keys())
    region_weights = {"global": latlon_weights}
    for sr_name, sr_bounds in subregions.items():
        region_weights[sr_name] = _subregion_weights(latlon_weights, vds, sr_bounds)

    if subregions:
        logger.info(f"Subregions: {list(subregions.keys())}")
        if topo.is_root:
            latitude, longitude = _get_latlon(latlon_weights, vds)
            srds = xr.Dataset({
                sr_name: xr.where(_subregion_mask(latitude, longitude, sr_bounds), 1.0, np.nan)
                for sr_name, sr_bounds in subregions.items()
            })
            fname = f"{config['output_path']}/subregions.{model_type}.nc"
            srds.to_netcdf(fname)
            logger.info(f"Stored subregion masks at {fname}")

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    metric_names = ["rmse", "mae"]
    if is_ensemble:
        metric_names += ["spread", "fcrps", "rmse_ensmean", "mae_ensmean"]

    # Containers nested by region: {region: {metric: [per-IC datasets]}}
    containers = {rn: {m: [] for m in metric_names} for rn in region_names}

    logger.info(f"Computing Error Metrics")
    logger.info(f"Initial Conditions:\n{dates}")
    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break # last batch situation

        try:
            t0 = dates[date_idx]
        except:
            logger.error(f"Error getting this date: {date_idx} / {n_dates}")
            raise

        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        # Load forecast member(s)
        member_fds_list = []
        for member in range(n_members):
            if config.get("from_anemoi", True):
                fname = f"{config['forecast_path']}/{st0}.{config['lead_time']}h.nc"
                if is_ensemble:
                    fname = fname.replace(".nc", f".member{member:03d}.nc")
                fds = open_anemoi_inference_dataset(
                    fname,
                    model_type=model_type,
                    lam_index=lam_index,
                    trim_edge=config.get("trim_forecast_edge", None),
                    load=True,
                    reshape_cell_to_2d=do_any_regridding,
                    horizontal_regrid_kwargs=forecast_regrid_kwargs if model_type == "nested-global" else None,
                    **subsample_kwargs,
                )
            else:
                fds = open_forecast_zarr_dataset(
                    config["forecast_path"],
                    t0=t0,
                    member=member if is_ensemble else None,
                    trim_edge=config.get("trim_forecast_edge", None),
                    load=True,
                    reshape_cell_to_2d=do_any_regridding,
                    **subsample_kwargs,
                )

            if forecast_regrid_kwargs is not None and model_type != "nested-global":
                fds = horizontal_regrid(fds, **forecast_regrid_kwargs)

            member_fds_list.append(fds)

        # Load target data once (using time coords from first member)
        tds = vds.sel(time=member_fds_list[0].time.values).load()
        if do_any_regridding:
            tds = reshape_cell_dim(tds, model_type, subsample_kwargs["lcc_info"])
        if target_regrid_kwargs is not None:
            tds = horizontal_regrid(tds, **target_regrid_kwargs)

        if is_ensemble:
            ensemble_fds = xr.concat(member_fds_list, dim="member")
            ensmean = ensemble_fds.mean("member")

        # Compute metrics for the full field ("global") and each subregion, by
        # re-using the same metric functions with region-masked area weights.
        for rn in region_names:
            weights = region_weights[rn]

            member_rmse_list = []
            member_mae_list = []
            for member in range(n_members):
                member_rmse_list.append(rmse(target=tds, prediction=member_fds_list[member], weights=weights, **mkw))
                member_mae_list.append(mae(target=tds, prediction=member_fds_list[member], weights=weights, **mkw))

            if is_ensemble:
                containers[rn]["rmse"].append(xr.concat(member_rmse_list, dim="member"))
                containers[rn]["mae"].append(xr.concat(member_mae_list, dim="member"))
            else:
                containers[rn]["rmse"].append(member_rmse_list[0])
                containers[rn]["mae"].append(member_mae_list[0])

            # Ensemble-only metrics
            if is_ensemble:
                containers[rn]["spread"].append(spread(ensemble_fds, weights=weights))
                containers[rn]["fcrps"].append(fcrps(target=tds, ensemble=ensemble_fds, weights=weights))
                containers[rn]["rmse_ensmean"].append(rmse(target=tds, prediction=ensmean, weights=weights, **mkw))
                containers[rn]["mae_ensmean"].append(mae(target=tds, prediction=ensmean, weights=weights, **mkw))

        logger.info(f"Done with {st0}")
    logger.info(f"Done Computing Metrics")

    logger.info(f"Gathering Results on Root Process")
    for rn in region_names:
        for name in metric_names:
            containers[rn][name] = topo.gather(containers[rn][name])

    if topo.is_root:
        logger.info("Combining & Storing Results")
        for rn in region_names:
            suffix = "" if rn == "global" else f".{rn}"
            for name in metric_names:
                c = containers[rn][name]
                if config["use_mpi"]:
                    c = [xds for sublist in c for xds in sublist]
                c = sorted(c, key=lambda xds: xds.coords["t0"])
                c = xr.concat(c, dim="t0")
                fname = f"{config['output_path']}/{name}.{model_type}{suffix}.nc"
                c.to_netcdf(fname)
                logger.info(f"Stored result: {fname}")

        logger.info("Done Storing Error Metrics")
