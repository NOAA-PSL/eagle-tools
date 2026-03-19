"""
Compute metrics against observations.
"""
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

import numpy as np
import xarray as xr
import pandas as pd

import xesmf
import nnja_ai

from ufs2arco.transforms.horizontal_regrid import maybe_make_dataset_c_contiguous

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import postprocess
from eagle.tools.nested import prepare_regrid_target_mask

logger = logging.getLogger("eagle.tools")

BASE_METRICS = ["rmse", "mae", "bias", "count"]
ENSEMBLE_METRICS = ["rmse_ensmean", "mae_ensmean", "bias_ensmean", "count_ensmean", "spread", "fcrps"]


def _load_obs_config():
    """Load observation configuration from obs_metrics.yaml."""
    path = importlib.resources.files("eagle.tools.config") / "obs_metrics.yaml"
    with path.open("r") as f:
        return yaml.safe_load(f)


def _get_unit_conversion_funcs(gravity):
    """Build unit conversion functions using gravity from config."""
    return {
        "gp_to_gph": lambda x: x / gravity,
    }


def _dewpoint_to_specific_humidity(td_kelvin, pressure_hpa):
    """Convert dewpoint temperature to specific humidity.
    Uses Magnus formula consistent with NOAA MET vx_physics/thermo.cc.
    """
    e = 6.112 * np.exp(17.67 * (td_kelvin - 273.15) / (td_kelvin - 29.65))
    w = 0.622 * e / (pressure_hpa - e)
    return w / (1 + w)


def _is_upper_air(base_name, dataset_registry):
    """A variable is upper-air if it appears in any adpupa dataset."""
    return any(
        base_name in reg and "adpupa" in ds_name
        for ds_name, reg in dataset_registry.items()
    )


def build_variable_map(config, levels, obs_config=None):
    """Expand config into a flat map keyed by forecast variable name.

    Reads a flat ``variables`` list from config. Upper-air variables (those
    appearing in any adpupa dataset) are expanded across all levels; surface
    variables get a single entry with level=None.

    Args:
        config: User config dict with ``vars_of_interest``.
        levels: List of pressure levels for upper-air expansion.
        obs_config: Observation config from obs_metrics.yaml. Loaded if None.

    Returns:
        dict keyed by forecast variable name (e.g. "t_850", "10m_meridional_wind").
    """
    if obs_config is None:
        obs_config = _load_obs_config()

    dataset_registry = obs_config["dataset_registry"]
    wind_variables = obs_config["wind_variables"]
    derived_variables = obs_config.get("derived_variables", {})

    all_registry_vars = set()
    for reg in dataset_registry.values():
        all_registry_vars.update(reg.keys())

    # Get user requested variables and rename to common naming convention
    variables = config.get("vars_of_interest")
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rdict = yaml.safe_load(f)

    renamed = [rdict.get(key, key) for key in variables]

    variable_map = {}
    for base_name in renamed:
        if base_name not in all_registry_vars:
            raise ValueError(
                f"Variable '{base_name}' is not available in any dataset. "
                f"Available variables: {sorted(all_registry_vars)}"
            )

        # Look up unit_conversion and needs_dewpoint_conversion from the first registry entry
        conversion = None
        needs_dewpoint = False
        for reg in dataset_registry.values():
            if base_name in reg:
                conversion = reg[base_name].get("unit_conversion", None)
                needs_dewpoint = reg[base_name].get("needs_dewpoint_conversion", False)
                break

        upper_air = _is_upper_air(base_name, dataset_registry)
        is_derived = base_name in derived_variables

        # For derived wind speed variables, obs_col points to the WSPD column
        # from the corresponding wind group (already loaded by wind components)
        if is_derived:
            components = derived_variables[base_name]["forecast_components"]
            u_name = components["u"]
            group = wind_variables[u_name]["group"]

        if upper_air and levels:
            for level in levels:
                forecast_var = f"{base_name}_{level}"
                entry = {
                    "base_name": base_name,
                    "level": level,
                    "obs_col": f"obs_{base_name}_{level}",
                    "obs_qc_col": f"obs_qc_{base_name}_{level}",
                    "unit_conversion": conversion,
                    "needs_dewpoint_conversion": needs_dewpoint,
                }
                if base_name in wind_variables:
                    wgroup = wind_variables[base_name]["group"]
                    entry["obs_wspd_col"] = f"obs_wspd_{wgroup}_{level}"
                    entry["obs_wdir_col"] = f"obs_wdir_{wgroup}_{level}"
                    entry["obs_qc_col"] = f"obs_qc_{wgroup}_{level}"
                    entry["wind_component"] = wind_variables[base_name]["component"]
                if is_derived:
                    entry["forecast_components"] = components
                    entry["obs_col"] = f"obs_wspd_{group}_{level}"
                    entry["obs_qc_col"] = f"obs_qc_{group}_{level}"
                variable_map[forecast_var] = entry
        else:
            entry = {
                "base_name": base_name,
                "level": None,
                "obs_col": f"obs_{base_name}",
                "obs_qc_col": f"obs_qc_{base_name}",
                "unit_conversion": conversion,
                "needs_dewpoint_conversion": needs_dewpoint,
            }
            if base_name in wind_variables:
                wgroup = wind_variables[base_name]["group"]
                entry["obs_wspd_col"] = f"obs_wspd_{wgroup}"
                entry["obs_wdir_col"] = f"obs_wdir_{wgroup}"
                entry["obs_qc_col"] = f"obs_qc_{wgroup}"
                entry["wind_component"] = wind_variables[base_name]["component"]
            if is_derived:
                entry["forecast_components"] = components
                entry["obs_col"] = f"obs_wspd_{group}"
                entry["obs_qc_col"] = f"obs_qc_{group}"
            variable_map[base_name] = entry

    return variable_map


def _build_rename_map(registry, variable_map):
    """Build real-col -> standardized-col rename map for one dataset."""
    rename_map = {}
    for forecast_var, vinfo in variable_map.items():
        # Derived variables reuse obs columns already loaded by their components
        if "forecast_components" in vinfo:
            continue
        base_name = vinfo["base_name"]
        if base_name not in registry:
            continue
        reg = registry[base_name]
        level = vinfo["level"]

        if "obs_wspd_col" in vinfo:
            if level is not None:
                prlc_suffix = f"PRLC{level * 100}"
                real_wspd = f"{reg['obs_wspd_var']}_{prlc_suffix}"
                real_wdir = f"{reg['obs_wdir_var']}_{prlc_suffix}"
                real_qc = f"{reg['obs_qc_var']}_{prlc_suffix}"
            else:
                real_wspd = reg["obs_wspd_var"]
                real_wdir = reg["obs_wdir_var"]
                real_qc = reg["obs_qc_var"]
            rename_map[real_wspd] = vinfo["obs_wspd_col"]
            rename_map[real_wdir] = vinfo["obs_wdir_col"]
            rename_map[real_qc] = vinfo["obs_qc_col"]
        else:
            if level is not None:
                prlc_suffix = f"PRLC{level * 100}"
                real_obs_col = f"{reg['obs_var']}_{prlc_suffix}"
                real_qc_col = f"{reg['obs_qc_var']}_{prlc_suffix}"
            else:
                real_obs_col = reg["obs_var"]
                real_qc_col = reg["obs_qc_var"]
            rename_map[real_obs_col] = vinfo["obs_col"]
            rename_map[real_qc_col] = vinfo["obs_qc_col"]
    return rename_map


def _load_one_dataset(dc, dataset_name, rename_map, time_range):
    """Load and rename observations from a single dataset."""
    columns = list(dict.fromkeys(["LAT", "LON", "OBS_TIMESTAMP"] + list(rename_map.keys())))

    ds = dc[dataset_name]
    try:
        subds = ds.sel(
            time=slice(str(time_range[0]), str(time_range[1])),
            variables=columns,
        )
        obs_df = subds.load_dataset()
    except nnja_ai.exceptions.EmptyTimeSubsetError:
        logger.warning(f"No observations in {dataset_name} for {time_range[0]} to {time_range[1]}")
        return None

    obs_df = obs_df.rename(columns=rename_map)
    logger.info(f"Loaded {len(obs_df)} observations from {dataset_name}")
    return obs_df


def load_all_observations(time_range, variable_map, dataset_registry):
    """Load observations from all datasets in dataset_registry.

    For each dataset, determines which user-requested variables it supports,
    builds the real column names, loads from nnja_ai, renames to standardized
    names, and concatenates all DataFrames. Datasets are loaded in parallel
    using threads since the work is I/O-bound.

    Args:
        time_range: (start, end) tuple of pd.Timestamps for time selection.
        variable_map: Output of build_variable_map.
        dataset_registry: Dataset registry dict from obs config.

    Returns:
        pd.DataFrame with LAT, LON, OBS_TIMESTAMP, and standardized
        obs/QC columns (obs_{var}, obs_qc_{var}).
    """
    dc = nnja_ai.DataCatalog()

    tasks = {}
    for dataset_name, registry in dataset_registry.items():
        rename_map = _build_rename_map(registry, variable_map)
        if rename_map:
            tasks[dataset_name] = rename_map

    all_frames = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(_load_one_dataset, dc, ds_name, rmap, time_range): ds_name
            for ds_name, rmap in tasks.items()
        }
        for future in as_completed(futures):
            obs_df = future.result()
            if obs_df is not None:
                all_frames.append(obs_df)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        logger.info(f"Total observations across all datasets: {len(result)}")
    else:
        std_columns = ["LAT", "LON", "OBS_TIMESTAMP"]
        for vinfo in variable_map.values():
            std_columns.append(vinfo["obs_col"])
            std_columns.append(vinfo["obs_qc_col"])
            if "obs_wspd_col" in vinfo:
                std_columns.append(vinfo["obs_wspd_col"])
                std_columns.append(vinfo["obs_wdir_col"])
        result = pd.DataFrame(columns=list(dict.fromkeys(std_columns)))
        logger.warning("No observations loaded from any dataset")
    return result


def apply_qc_filter(obs_df, variable_map, max_qc_value=2):
    """Apply per-variable QC filtering.

    Masks obs values to NaN where QC is non-NaN AND > max_qc_value.
    NaN QC means not flagged -> keep. 0-2 = good -> keep. 3+ = suspect/rejected -> mask.

    For wind variables, masks the WSPD and WDIR source columns (obs_col
    doesn't exist yet — it is derived later).
    """
    for forecast_var, vinfo in variable_map.items():
        qc_col = vinfo["obs_qc_col"]
        if qc_col not in obs_df.columns:
            continue

        bad = obs_df[qc_col].notna() & (obs_df[qc_col] > max_qc_value)
        n_rejected = bad.sum()

        if "obs_wspd_col" in vinfo:
            for col in [vinfo["obs_wspd_col"], vinfo["obs_wdir_col"]]:
                if col in obs_df.columns:
                    if n_rejected > 0:
                        logger.info(f"QC filter: masking {n_rejected} obs in {col} for {forecast_var} (QC > {max_qc_value})")
                    obs_df.loc[bad, col] = np.nan
        else:
            obs_col = vinfo["obs_col"]
            if obs_col in obs_df.columns:
                if n_rejected > 0:
                    logger.info(f"QC filter: masking {n_rejected} obs for {forecast_var} (QC > {max_qc_value})")
                obs_df.loc[bad, obs_col] = np.nan
    return obs_df


def derive_wind_components(obs_df, variable_map):
    """Derive u/v wind components from WSPD/WDIR columns.

    For each wind variable in variable_map, computes:
        u = -wspd * sin(wdir_rad)
        v = -wspd * cos(wdir_rad)

    Tracks already-derived (wspd_col, wdir_col) pairs to avoid duplicate work
    when u and v share the same source columns.
    """
    derived = set()
    for forecast_var, vinfo in variable_map.items():
        if "obs_wspd_col" not in vinfo:
            continue
        wspd_col = vinfo["obs_wspd_col"]
        wdir_col = vinfo["obs_wdir_col"]
        obs_col = vinfo["obs_col"]
        component = vinfo["wind_component"]

        key = (wspd_col, wdir_col, component)
        if key in derived:
            continue
        derived.add(key)

        if wspd_col not in obs_df.columns or wdir_col not in obs_df.columns:
            continue

        wdir_rad = np.deg2rad(obs_df[wdir_col])
        wspd = obs_df[wspd_col]
        if component == "zonal_wind":
            obs_df[obs_col] = -wspd * np.sin(wdir_rad)
        else:
            obs_df[obs_col] = -wspd * np.cos(wdir_rad)
        logger.info(f"Derived {obs_col} from {wspd_col}/{wdir_col}")
    return obs_df


def convert_obs_units(obs_df, variable_map, unit_conversion_funcs):
    """Apply unit conversions to observation columns as specified in variable map."""
    for forecast_var, vinfo in variable_map.items():
        conversion = vinfo["unit_conversion"]
        if conversion is not None:
            obs_col = vinfo["obs_col"]
            if obs_col in obs_df.columns:
                obs_df[obs_col] = unit_conversion_funcs[conversion](obs_df[obs_col])
    return obs_df


def convert_dewpoint_to_specific_humidity(obs_df, variable_map):
    """Convert dewpoint temperature obs to specific humidity in-place.

    For variables with ``needs_dewpoint_conversion``, the obs column contains
    dewpoint temperature (K). This function converts it to specific humidity
    (kg/kg) using the Magnus formula.

    Pressure source:
    - Upper-air (level is not None): pressure = level in hPa.
    - Surface (level is None): pressure = observed surface pressure from the
      ``obs_surface_pressure`` column (Pa, divided by 100 to get hPa).
    """
    has_dewpoint_var = any(
        v.get("needs_dewpoint_conversion") for v in variable_map.values()
    )
    if not has_dewpoint_var:
        return obs_df

    sp_col = "obs_surface_pressure"
    for forecast_var, vinfo in variable_map.items():
        if not vinfo.get("needs_dewpoint_conversion"):
            continue
        if vinfo["level"] is None and sp_col not in obs_df.columns:
            raise ValueError(
                f"{vinfo['base_name']} requires 'surface_pressure' in variables list "
                "to provide pressure for dewpoint conversion"
            )

    for forecast_var, vinfo in variable_map.items():
        if not vinfo.get("needs_dewpoint_conversion"):
            continue
        obs_col = vinfo["obs_col"]
        if obs_col not in obs_df.columns:
            continue

        td = obs_df[obs_col]
        if vinfo["level"] is not None:
            pressure_hpa = float(vinfo["level"])
        else:
            pressure_hpa = obs_df[sp_col] / 100.0

        obs_df[obs_col] = _dewpoint_to_specific_humidity(td, pressure_hpa)
        logger.info(f"Converted dewpoint to specific humidity for {obs_col}")

    return obs_df


def align_obs_to_forecast_times(obs_df, forecast_valid_times, window):
    """Match observations to forecast valid times within a temporal window.

    Args:
        obs_df: DataFrame with OBS_TIMESTAMP column.
        forecast_valid_times: Array of forecast valid times (np.datetime64).
        window: pd.Timedelta for +/- matching window.

    Returns:
        dict mapping pd.Timestamp -> DataFrame of matched observations.
    """
    aligned = {}
    obs_times = obs_df["OBS_TIMESTAMP"]
    for vt in forecast_valid_times:
        vtimestamp = pd.Timestamp(vt, tz="UTC")
        mask = (obs_times >= vtimestamp - window) & (obs_times < vtimestamp + window)
        matched = obs_df.loc[mask]
        if len(matched) > 0:
            aligned[vtimestamp] = matched
    return aligned


def _interp_to_obs_locations(fds_time_slice, matched_obs_df):
    """Interpolate a forecast time-slice to observation locations using xesmf.

    Args:
        fds_time_slice: xr.Dataset for a single time (no time dim), with
            ``latitude`` and ``longitude`` coordinates.
        matched_obs_df: pandas DataFrame with ``LAT`` and ``LON`` columns.

    Returns:
        xr.Dataset interpolated to observation locations (dim ``locations``).
    """
    src = fds_time_slice.rename({"latitude": "lat", "longitude": "lon"})
    src = maybe_make_dataset_c_contiguous(src)

    obs_loc = xr.Dataset({
        "lat": xr.DataArray(matched_obs_df["LAT"].values, dims=("locations",)),
        "lon": xr.DataArray(matched_obs_df["LON"].values, dims=("locations",)),
    })

    regridder = xesmf.Regridder(
        src,
        obs_loc,
        method="bilinear",
        locstream_out=True,
        unmapped_to_nan=True,
    )
    return regridder(src)


def _compute_metrics_single(forecast_values, obs_values):
    """Compute verification metrics for a single set of forecast/obs pairs."""
    valid = np.isfinite(forecast_values) & np.isfinite(obs_values)
    n = valid.sum()
    if n == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "count": 0}
    f = forecast_values[valid]
    o = obs_values[valid]
    diff = f - o
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "count": int(n),
    }


def compute_obs_metrics(forecast_values, obs_values, vtime):
    """Compute verification metrics between forecast and observation values.

    Args:
        forecast_values: numpy array of forecast values at obs locations.
            1-D (n_obs,) for deterministic or 2-D (n_members, n_obs) for ensemble.
        obs_values: 1-D numpy array of observation values.
        vtime: numpy datetime64 valid time for the time coordinate.

    Returns:
        xr.Dataset with a time dimension. For deterministic (1-D input):
            rmse, mae, bias, count.
        For ensemble (2-D input):
            rmse, mae, bias with (member, time) dims (per-member metrics),
            count with (time,) dim,
            rmse_ensmean, mae_ensmean, bias_ensmean, count_ensmean with (time,) dim,
            spread with (time,) dim.
    """
    if forecast_values.ndim == 1:
        result = _compute_metrics_single(forecast_values, obs_values)
        xds = xr.Dataset(result)
        xds = xds.expand_dims({"time": [vtime]})
        return xds

    # Ensemble case: forecast_values is (n_members, n_obs)
    n_members = forecast_values.shape[0]
    forecast = xr.DataArray(
        forecast_values,
        dims=("member", "locations"),
        coords={"member": np.arange(n_members)},
    )
    obs = xr.DataArray(obs_values, dims="locations")

    valid = forecast.notnull().all("member") & obs.notnull()
    forecast = forecast.where(valid)
    obs = obs.where(valid)
    n_valid = valid.sum().item()

    diff = forecast - obs
    per_member_vars = {
        "rmse": np.sqrt((diff**2).mean("locations")),
        "mae": np.abs(diff).mean("locations"),
        "bias": diff.mean("locations"),
    }

    ensmean_diff = forecast.mean("member") - obs
    ensmean_metrics = {
        "rmse_ensmean": float(np.sqrt((ensmean_diff**2).mean("locations"))),
        "mae_ensmean": float(np.abs(ensmean_diff).mean("locations")),
        "bias_ensmean": float(ensmean_diff.mean("locations")),
        "count_ensmean": n_valid,
    }

    spread_val = float(forecast.std("member").mean("locations")) if n_valid > 0 else np.nan

    # Fair CRPS (1/N) * Σ|u_e - u*| - 1/(2*N*(N-1)) * ΣΣ|u_e - u_i|
    if n_valid == 0:
        fcrps_val = np.nan
    else:
        abs_err = np.abs(forecast - obs).mean("member")
        pairwise = np.abs(
            forecast - forecast.rename({"member": "_member"})
        ).mean(("member", "_member"))
        fcrps_val = float((abs_err - pairwise / (2 * (n_members - 1))).mean("locations"))

    scalar_vars = {
        "count": n_valid,
        **ensmean_metrics,
        "spread": spread_val,
        "fcrps": fcrps_val,
    }

    xds = xr.Dataset({**per_member_vars, **scalar_vars})
    xds = xds.expand_dims({"time": [vtime]})
    return xds


def _parse_subregions(config):
    """Parse subregion definitions from config.

    Returns dict of {name: {"latitude": (min, max), "longitude": (min, max)}}.
    Longitude bounds are converted from [-180, 180] to [0, 360] to match obs convention.
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


def _subregion_mask(obs_df, bounds):
    """Return a boolean numpy array selecting observations within a geographic subregion."""
    lat_min, lat_max = bounds["latitude"]
    lon_min, lon_max = bounds["longitude"]
    lat_mask = (obs_df["LAT"] >= lat_min) & (obs_df["LAT"] <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (obs_df["LON"] >= lon_min) & (obs_df["LON"] <= lon_max)
    else:
        lon_mask = (obs_df["LON"] >= lon_min) | (obs_df["LON"] <= lon_max)
    return (lat_mask & lon_mask).values


def _filter_obs_by_subregion(obs_df, bounds, drop_out_of_bounds=True):
    """Filter observations DataFrame to a geographic subregion.

    Handles longitude wrapping around 0/360.

    Note:
        drop_out_of_bounds is a hack to reuse this function.
        Set it to True when evaluating observations, set to False for creating subregion masks with an xarray dataset.
    """
    lat_min, lat_max = bounds["latitude"]
    lon_min, lon_max = bounds["longitude"]
    lat_mask = (obs_df["LAT"] >= lat_min) & (obs_df["LAT"] <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (obs_df["LON"] >= lon_min) & (obs_df["LON"] <= lon_max)
    else:
        lon_mask = (obs_df["LON"] >= lon_min) | (obs_df["LON"] <= lon_max)
    if drop_out_of_bounds:
        return obs_df.loc[lat_mask & lon_mask]
    else:
        return obs_df.where(lat_mask & lon_mask)


def create_subregion_masks(subregions):
    """Create dataset with subregion masks for visualization."""
    xds = xesmf.util.grid_global(1, 1, cf=True, lon1=360)
    for sr_name, sr_bounds in subregions.items():
        xds[sr_name] = _filter_obs_by_subregion(
            xr.ones_like(xds["lat"]*xds["lon"]).rename({"lon": "LON", "lat": "LAT"}),
            sr_bounds,
            drop_out_of_bounds=False,
        ).rename({"LON": "longitude", "LAT": "latitude"})
    return xds


def _assemble_metric_dataset(metric_data, variable_map):
    """Assemble per-variable metric lists into an xr.Dataset with level dims.

    Groups variables by base_name, concatenates along time, stacks levels
    for upper-air variables, and applies postprocessing.
    """
    base_groups = {}
    for varname, vals in metric_data.items():
        vinfo = variable_map[varname]
        bn = vinfo["base_name"]
        level = vinfo["level"]
        time_concat = xr.concat(vals, dim="time")
        base_groups.setdefault(bn, {})[level] = time_concat

    data_vars = {}
    for bn, level_dict in base_groups.items():
        if None in level_dict:
            data_vars[bn] = level_dict[None]
        else:
            level_arrays = [
                level_dict[lvl].expand_dims({"level": [lvl]})
                for lvl in sorted(level_dict.keys())
            ]
            data_vars[bn] = xr.concat(level_arrays, dim="level")

    return postprocess(xr.Dataset(data_vars))


def _concat_and_sort(datasets, use_mpi):
    """Flatten MPI-gathered results and concatenate along t0."""
    if use_mpi:
        datasets = [xds for sublist in datasets for xds in sublist]
    return xr.concat(sorted(datasets, key=lambda xds: xds.coords["t0"]), dim="t0")


def _write_metric_files(container, output_path, model_type, suffix=""):
    """Write metric datasets to NetCDF files."""
    for metric, xds in container.items():
        fname = f"{output_path}/{metric}.convobs.{model_type}{suffix}.nc"
        xds.to_netcdf(fname)
        logger.info(f"Stored result: {fname}")


def _compute_metrics_from_interpolated(interpolated, obs_df, variable_map, vtime, n_members, is_ensemble):
    """Compute metrics for all variables from already-interpolated forecast data.

    Args:
        interpolated: xr.Dataset of forecast values interpolated to obs locations
            (dim ``locations``).
        obs_df: pandas DataFrame of observations aligned to the same locations.
        variable_map: Output of build_variable_map.
        vtime: numpy datetime64 valid time.
        n_members: Number of ensemble members.
        is_ensemble: Whether this is an ensemble forecast.

    Returns:
        Tuple of (base_results, ensemble_results) where each is a dict
        mapping metric_name -> {varname: xr.DataArray}.
        ensemble_results is empty dict if not is_ensemble.
    """
    base_results = {m: {} for m in BASE_METRICS}
    ensemble_results = {m: {} for m in ENSEMBLE_METRICS} if is_ensemble else {}

    for varname, vinfo in variable_map.items():
        if "forecast_components" in vinfo:
            components = vinfo["forecast_components"]
            if "u" in components and "v" in components:
                u = interpolated[components["u"]]
                v = interpolated[components["v"]]
                if "level" in u.dims:
                    u = u.sel(level=vinfo["level"])
                    v = v.sel(level=vinfo["level"])
                fvals = np.sqrt(u**2 + v**2)
            else:
                raise ValueError(
                    f"Unknown forecast_components for '{varname}': {components}. "
                    "Only wind speed (u/v) derivation is currently supported."
                )
        else:
            fvals = interpolated[vinfo["base_name"]]
            if "level" in fvals.dims:
                fvals = fvals.sel(level=vinfo["level"])

        result = compute_obs_metrics(
            fvals.values,
            obs_df[vinfo["obs_col"]].values,
            vtime,
        )
        for metric in BASE_METRICS:
            base_results[metric][varname] = result[metric]
        if is_ensemble:
            for metric in ENSEMBLE_METRICS:
                ensemble_results[metric][varname] = result[metric]

    return base_results, ensemble_results


def _make_empty_results(forecast_var_names, n_members, is_ensemble, vtime):
    """Create NaN metric results for all variables (no observations case)."""
    if is_ensemble:
        empty_fvals = np.full((n_members, 1), np.nan)
    else:
        empty_fvals = np.array([np.nan])
    empty_result = compute_obs_metrics(empty_fvals, np.array([np.nan]), vtime)

    base_results = {m: {} for m in BASE_METRICS}
    ensemble_results = {m: {} for m in ENSEMBLE_METRICS} if is_ensemble else {}
    for varname in forecast_var_names:
        for metric in BASE_METRICS:
            base_results[metric][varname] = empty_result[metric]
        if is_ensemble:
            for metric in ENSEMBLE_METRICS:
                ensemble_results[metric][varname] = empty_result[metric]
    return base_results, ensemble_results


def _load_forecast(config, t0, member, levels, derived_var_names=None):
    """Load a single forecast member.

    Derived variable names (e.g. wind_speed) are filtered out of
    vars_of_interest since they don't exist in the raw forecast dataset.
    """
    st0 = t0.strftime("%Y-%m-%dT%H")
    model_type = config.get("model_type")
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)

    vars_of_interest = config.get("vars_of_interest")
    if vars_of_interest is not None and derived_var_names:
        vars_of_interest = [v for v in vars_of_interest if v not in derived_var_names]

    if config.get("from_anemoi", True):
        fname = f"{config['forecast_path']}/{st0}.{config['lead_time']}h.nc"
        if config.get("n_members", 1) > 1:
            fname = fname.replace(".nc", f".member{member:03d}.nc")
        return open_anemoi_inference_dataset(
            fname,
            model_type=model_type,
            lam_index=config.get("lam_index", None),
            trim_edge=config.get("trim_forecast_edge", None),
            vars_of_interest=vars_of_interest,
            levels=levels,
            load=True,
            lcc_info=config.get("lcc_info", None),
            horizontal_regrid_kwargs=forecast_regrid_kwargs if model_type == "nested-global" else None,
            reshape_cell_to_2d=True,
            rename_to_longnames=True,
        )
    else:
        return open_forecast_zarr_dataset(
            config["forecast_path"],
            t0=t0,
            trim_edge=config.get("trim_forecast_edge", None),
            vars_of_interest=vars_of_interest,
            levels=levels,
            load=True,
            lcc_info=config.get("lcc_info", None),
            reshape_cell_to_2d=True,
            rename_to_longnames=True,
        )


def main(config):
    """Verify forecasts against observations.

    See ``eagle-tools obs-metrics --help`` or cli.py for help.
    """
    if isinstance(config, str):
        from eagle.tools.utils import setup
        config = setup(config, "obs_metrics")

    topo = config["topo"]
    obs_config = _load_obs_config()
    dataset_registry = obs_config["dataset_registry"]
    gravity = obs_config["gravity"]
    unit_conversion_funcs = _get_unit_conversion_funcs(gravity)

    # Config options
    model_type = config.get("model_type")
    lead_time = config["lead_time"]
    temporal_window = pd.Timedelta(config.get("temporal_window", "30min"))
    max_qc_value = config.get("max_qc_value", 2)
    n_members = config.get("n_members", 1)
    is_ensemble = n_members > 1
    user_levels = config.get("levels", None)

    # Does the user want to evaluate on a different grid?
    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = (target_regrid_kwargs is not None) or \
            ((forecast_regrid_kwargs is not None) and (model_type != "nested-global"))
    if do_any_regridding:
        raise NotImplementedError

    if model_type == "nested-global":
        forecast_regrid_kwargs["target_grid_path"], _ = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=forecast_regrid_kwargs,
        )

    # Parse subregions
    subregions = _parse_subregions(config)
    if subregions:
        logger.info(f"Subregions: {list(subregions.keys())}")
        if topo.is_root:
            srds = create_subregion_masks(subregions)
            fname = f"{config['output_path']}/subregions.nc"
            srds.to_netcdf(fname)
            logger.info(f"Stored subregion masks at {fname}")

    # Generate initialization dates
    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    # Set up region names: "global" + any subregions
    region_names = ["global"] + list(subregions.keys())

    # Initialize containers: {region: {metric: []}}
    base_containers = {rn: {m: [] for m in BASE_METRICS} for rn in region_names}
    ensemble_containers = {rn: {m: [] for m in ENSEMBLE_METRICS} for rn in region_names} if is_ensemble else {}

    logger.info("Observation Verification")
    logger.info(f"Datasets: {list(dataset_registry.keys())}")
    logger.info(f"Temporal window: +/- {temporal_window}")
    logger.info(f"Max QC value: {max_qc_value}")
    logger.info(f"Initial Conditions:\n{dates}")

    # Derived variables (e.g. wind_speed) don't exist in raw forecast datasets
    # Collect both long names and any short-name aliases so they get filtered
    derived_variables = obs_config.get("derived_variables", {})
    derived_long_names = set(derived_variables.keys())
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rdict = yaml.safe_load(f)
    reverse_rename = {v: k for k, v in rdict.items()}
    derived_var_names = set(derived_long_names)
    for ln in derived_long_names:
        if ln in reverse_rename:
            derived_var_names.add(reverse_rename[ln])

    # Track whether we've discovered levels yet
    levels = user_levels
    variable_map = None

    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break

        try:
            t0 = dates[date_idx]
        except Exception:
            logger.error(f"Error getting this date: {date_idx} / {n_dates}")
            raise

        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        # Load forecast
        member_fds_list = [
            _load_forecast(config, t0, member, levels, derived_var_names=derived_var_names)
            for member in range(n_members)
        ]

        if is_ensemble:
            fds = xr.concat(member_fds_list, dim="member")
        else:
            fds = member_fds_list[0]

        # Discover levels from the first forecast if not user-specified
        if variable_map is None:
            if levels is None and "level" in fds.dims:
                levels = sorted(int(lvl) for lvl in fds["level"].values)
                logger.info(f"Discovered levels from forecast: {levels}")
            variable_map = build_variable_map(config, levels, obs_config=obs_config)
            forecast_var_names = list(variable_map.keys())
            base_var_names = list({vinfo["base_name"] for vinfo in variable_map.values()})
            logger.info(f"Variables to verify: {forecast_var_names}")

        # Get forecast valid times
        forecast_valid_times = fds["time"].values

        # Load observations for the full valid time range (padded by window)
        time_start = pd.Timestamp(forecast_valid_times[0], tz="UTC") - temporal_window - pd.Timedelta("24h")
        time_end = pd.Timestamp(forecast_valid_times[-1], tz="UTC") + temporal_window + pd.Timedelta("24h")
        obs_df = load_all_observations((time_start, time_end), variable_map, dataset_registry)

        # QC filter and unit conversion
        obs_df = apply_qc_filter(obs_df, variable_map, max_qc_value=max_qc_value)
        obs_df = convert_obs_units(obs_df, variable_map, unit_conversion_funcs)
        obs_df = convert_dewpoint_to_specific_humidity(obs_df, variable_map)
        obs_df = derive_wind_components(obs_df, variable_map)

        # Convert obs longitudes to 0-360 to match forecast convention
        obs_df["LON"] = obs_df["LON"] % 360

        # Align observations to forecast valid times
        aligned = align_obs_to_forecast_times(obs_df, forecast_valid_times, temporal_window)

        # Accumulate metrics per forecast valid time, per region
        # {region: {metric: {varname: [DataArray, ...]}}}
        per_ic_base = {rn: {m: {v: [] for v in forecast_var_names} for m in BASE_METRICS} for rn in region_names}
        per_ic_ens = {rn: {m: {v: [] for v in forecast_var_names} for m in ENSEMBLE_METRICS} for rn in region_names} if is_ensemble else {}

        for vtime in forecast_valid_times:
            vtimestamp = pd.Timestamp(vtime, tz="UTC")

            if vtimestamp not in aligned:
                empty_base, empty_ens = _make_empty_results(forecast_var_names, n_members, is_ensemble, vtime)
                for rn in region_names:
                    for metric in BASE_METRICS:
                        for varname in forecast_var_names:
                            per_ic_base[rn][metric][varname].append(empty_base[metric][varname])
                    if is_ensemble:
                        for metric in ENSEMBLE_METRICS:
                            for varname in forecast_var_names:
                                per_ic_ens[rn][metric][varname].append(empty_ens[metric][varname])
                continue

            matched_obs = aligned[vtimestamp]

            # Interpolate forecast to obs locations once for all regions
            interpolated = _interp_to_obs_locations(fds.sel(time=vtime), matched_obs)

            # Global metrics
            base_res, ens_res = _compute_metrics_from_interpolated(
                interpolated, matched_obs, variable_map, vtime, n_members, is_ensemble,
            )
            for metric in BASE_METRICS:
                for varname in forecast_var_names:
                    per_ic_base["global"][metric][varname].append(base_res[metric][varname])
            if is_ensemble:
                for metric in ENSEMBLE_METRICS:
                    for varname in forecast_var_names:
                        per_ic_ens["global"][metric][varname].append(ens_res[metric][varname])

            # Subregion metrics (subset the already-interpolated data)
            for sr_name, sr_bounds in subregions.items():
                mask = _subregion_mask(matched_obs, sr_bounds)
                if mask.sum() == 0:
                    empty_base, empty_ens = _make_empty_results(forecast_var_names, n_members, is_ensemble, vtime)
                    sr_base_res, sr_ens_res = empty_base, empty_ens
                else:
                    sr_base_res, sr_ens_res = _compute_metrics_from_interpolated(
                        interpolated.isel(locations=mask), matched_obs.loc[mask],
                        variable_map, vtime, n_members, is_ensemble,
                    )
                for metric in BASE_METRICS:
                    for varname in forecast_var_names:
                        per_ic_base[sr_name][metric][varname].append(sr_base_res[metric][varname])
                if is_ensemble:
                    for metric in ENSEMBLE_METRICS:
                        for varname in forecast_var_names:
                            per_ic_ens[sr_name][metric][varname].append(sr_ens_res[metric][varname])

        # Assemble into xr.Datasets
        for rn in region_names:
            for metric, thedata in per_ic_base[rn].items():
                base_containers[rn][metric].append(_assemble_metric_dataset(thedata, variable_map))
            if is_ensemble:
                for metric, thedata in per_ic_ens[rn].items():
                    ensemble_containers[rn][metric].append(_assemble_metric_dataset(thedata, variable_map))

        logger.info(f"Done with {st0}")

    logger.info("Done Computing Observation Verification Metrics")

    # Gather results on root process
    logger.info("Gathering Results on Root Process")
    for rn in region_names:
        for metric in BASE_METRICS:
            base_containers[rn][metric] = topo.gather(base_containers[rn][metric])
        if is_ensemble:
            for metric in ENSEMBLE_METRICS:
                ensemble_containers[rn][metric] = topo.gather(ensemble_containers[rn][metric])

    if topo.is_root:
        output_path = config["output_path"]
        use_mpi = config["use_mpi"]

        for rn in region_names:
            suffix = f".{rn}" if rn != "global" else ""

            # Concat and sort base metrics
            for metric in BASE_METRICS:
                base_containers[rn][metric] = _concat_and_sort(base_containers[rn][metric], use_mpi)
            _write_metric_files(base_containers[rn], output_path, model_type, suffix)

            # Concat and sort ensemble metrics
            if is_ensemble:
                for metric in ENSEMBLE_METRICS:
                    ensemble_containers[rn][metric] = _concat_and_sort(ensemble_containers[rn][metric], use_mpi)
                _write_metric_files(ensemble_containers[rn], output_path, model_type, suffix)

        logger.info("Done Storing Observation Verification Metrics")
