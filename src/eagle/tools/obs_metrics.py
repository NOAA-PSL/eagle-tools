"""
Compute metrics against observations.
TODO:
    * move global variables to yamls
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

GRAVITY = 9.80665

UNIT_CONVERSIONS = {
    "gp_to_gph": lambda x: x / GRAVITY,
}


def _dewpoint_to_specific_humidity(td_kelvin, pressure_hpa):
    """Convert dewpoint temperature to specific humidity.
    Uses Magnus formula consistent with NOAA MET vx_physics/thermo.cc.
    """
    e = 6.112 * np.exp(17.67 * (td_kelvin - 273.15) / (td_kelvin - 29.65))
    w = 0.622 * e / (pressure_hpa - e)
    return w / (1 + w)

DATASET_REGISTRY = {
    "conv-adpupa-NC002001": {
        "temperature": {"obs_var": "TMDB", "obs_qc_var": "QMAT"},
        "geopotential_height": {"obs_var": "GP10", "obs_qc_var": "QMGP", "unit_conversion": "gp_to_gph"},
        "zonal_wind": {"obs_wspd_var": "WSPD", "obs_wdir_var": "WDIR", "obs_qc_var": "QMWN"},
        "meridional_wind": {"obs_wspd_var": "WSPD", "obs_wdir_var": "WDIR", "obs_qc_var": "QMWN"},
        "specific_humidity": {"obs_var": "TMDP", "obs_qc_var": "QMDD", "needs_dewpoint_conversion": True},
    },
    "conv-adpsfc-NC000001": {
        "2m_temperature": {"obs_var": "TMPSQ1.TMDB", "obs_qc_var": "TMPSQ1.QMAT"},
        "surface_pressure": {"obs_var": "PRSSQ1.PRES", "obs_qc_var": "PRSSQ1.QMPR"},
        "10m_zonal_wind": {"obs_wspd_var": "WNDSQ1.WSPD", "obs_wdir_var": "WNDSQ1.WDIR", "obs_qc_var": "WNDSQ1.QMWN"},
        "10m_meridional_wind": {"obs_wspd_var": "WNDSQ1.WSPD", "obs_wdir_var": "WNDSQ1.WDIR", "obs_qc_var": "WNDSQ1.QMWN"},
        "2m_specific_humidity": {"obs_var": "TMPSQ1.TMDP", "obs_qc_var": "TMPSQ1.QMDD", "needs_dewpoint_conversion": True},
    },
    "conv-adpsfc-NC000002": {
        "2m_temperature": {"obs_var": "TMPSQ1.TMDB", "obs_qc_var": "TMPSQ1.QMAT"},
        "surface_pressure": {"obs_var": "PRSSQ1.PRES", "obs_qc_var": "PRSSQ1.QMPR"},
        "10m_zonal_wind": {"obs_wspd_var": "WNDSQ1.WSPD", "obs_wdir_var": "WNDSQ1.WDIR", "obs_qc_var": "WNDSQ1.QMWN"},
        "10m_meridional_wind": {"obs_wspd_var": "WNDSQ1.WSPD", "obs_wdir_var": "WNDSQ1.WDIR", "obs_qc_var": "WNDSQ1.QMWN"},
        "2m_specific_humidity": {"obs_var": "TMPSQ1.TMDP", "obs_qc_var": "TMPSQ1.QMDD", "needs_dewpoint_conversion": True},
    },
    "conv-adpsfc-NC000007": {
        "2m_temperature": {"obs_var": "MTRTMP.TMDB", "obs_qc_var": "MTRTMP.QMAT"},
        "10m_zonal_wind": {"obs_wspd_var": "MTRWND.WSPD", "obs_wdir_var": "MTRWND.WDIR", "obs_qc_var": "MTRWND.QMWN"},
        "10m_meridional_wind": {"obs_wspd_var": "MTRWND.WSPD", "obs_wdir_var": "MTRWND.WDIR", "obs_qc_var": "MTRWND.QMWN"},
    },
    "conv-adpsfc-NC000101": {
        "2m_temperature": {"obs_var": "TEMHUMDA.TMDB", "obs_qc_var": "QMAT"},
        "surface_pressure": {"obs_var": "PRESDATA.PRESSQ03.PRES", "obs_qc_var": "QMPR"},
        "10m_zonal_wind": {"obs_wspd_var": "BSYWND1.WSPD", "obs_wdir_var": "BSYWND1.WDIR", "obs_qc_var": "QMWN"},
        "10m_meridional_wind": {"obs_wspd_var": "BSYWND1.WSPD", "obs_wdir_var": "BSYWND1.WDIR", "obs_qc_var": "QMWN"},
        "2m_specific_humidity": {"obs_var": "TEMHUMDA.TMDP", "obs_qc_var": "QMDD", "needs_dewpoint_conversion": True},
    },
}

WIND_VARIABLES = {
    "zonal_wind":   {"group": "uv",   "component": "zonal_wind"},
    "meridional_wind":   {"group": "uv",   "component": "meridional_wind"},
    "10m_zonal_wind": {"group": "u10m_meridional_wind", "component": "zonal_wind"},
    "10m_meridional_wind": {"group": "u10m_meridional_wind", "component": "meridional_wind"},
}

DEFAULT_LEVELS = [500, 850]


def _is_upper_air(base_name):
    """A variable is upper-air if it appears in any adpupa dataset."""
    return any(
        base_name in reg and "adpupa" in ds_name
        for ds_name, reg in DATASET_REGISTRY.items()
    )


def build_variable_map(config):
    """Expand config into a flat map keyed by forecast variable name.

    Reads a flat ``variables`` list and a ``levels`` list from config.
    Upper-air variables (those appearing in any adpupa dataset) are expanded
    across all levels; surface variables get a single entry with level=None.

    Each entry uses level-specific obs column names so that multiple levels
    can coexist in the same DataFrame without collisions.

    Returns:
        dict keyed by forecast variable name (e.g. "t_850", "u_850", "10m_meridional_wind").
    """
    all_registry_vars = set()
    for reg in DATASET_REGISTRY.values():
        all_registry_vars.update(reg.keys())

    # get user requested variables and rename to common naming convention
    variables = config.get("vars_of_interest")
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rdict = yaml.safe_load(f)

    renamed = [rdict.get(key, key) for key in variables]

    levels = config.get("levels", DEFAULT_LEVELS)
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
        for reg in DATASET_REGISTRY.values():
            if base_name in reg:
                conversion = reg[base_name].get("unit_conversion", None)
                needs_dewpoint = reg[base_name].get("needs_dewpoint_conversion", False)
                break

        upper_air = _is_upper_air(base_name)

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
                if base_name in WIND_VARIABLES:
                    wind_info = WIND_VARIABLES[base_name]
                    group = wind_info["group"]
                    entry["obs_wspd_col"] = f"obs_wspd_{group}_{level}"
                    entry["obs_wdir_col"] = f"obs_wdir_{group}_{level}"
                    entry["obs_qc_col"] = f"obs_qc_{group}_{level}"
                    entry["wind_component"] = wind_info["component"]
                variable_map[forecast_var] = entry
        else:
            # Surface variable — no levels
            entry = {
                "base_name": base_name,
                "level": None,
                "obs_col": f"obs_{base_name}",
                "obs_qc_col": f"obs_qc_{base_name}",
                "unit_conversion": conversion,
                "needs_dewpoint_conversion": needs_dewpoint,
            }
            if base_name in WIND_VARIABLES:
                wind_info = WIND_VARIABLES[base_name]
                group = wind_info["group"]
                entry["obs_wspd_col"] = f"obs_wspd_{group}"
                entry["obs_wdir_col"] = f"obs_wdir_{group}"
                entry["obs_qc_col"] = f"obs_qc_{group}"
                entry["wind_component"] = wind_info["component"]
            variable_map[base_name] = entry

    return variable_map


def _build_rename_map(registry, variable_map):
    """Build real-col -> standardized-col rename map for one dataset."""
    rename_map = {}
    for forecast_var, vinfo in variable_map.items():
        base_name = vinfo["base_name"]
        if base_name not in registry:
            continue
        reg = registry[base_name]
        level = vinfo["level"]

        if "obs_wspd_col" in vinfo:
            # Wind variable: map WSPD, WDIR, and QC columns
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
            # Direct variable
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
    columns = ["LAT", "LON", "OBS_TIMESTAMP"] + list(rename_map.keys())
    columns = list(dict.fromkeys(columns))  # deduplicate, preserve order

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


def load_all_observations(time_range, variable_map):
    """Load observations from all datasets in DATASET_REGISTRY.

    For each dataset, determines which user-requested variables it supports,
    builds the real column names, loads from nnja_ai, renames to standardized
    names, and concatenates all DataFrames.  Datasets are loaded in parallel
    using threads since the work is I/O-bound.

    Args:
        time_range: (start, end) tuple of pd.Timestamps for time selection.
        variable_map: Output of build_variable_map.

    Returns:
        pd.DataFrame with LAT, LON, OBS_TIMESTAMP, and standardized
        obs/QC columns (obs_{var}, obs_qc_{var}).
    """
    dc = nnja_ai.DataCatalog()

    # Pre-compute rename maps and filter to datasets that have matching vars
    tasks = {}
    for dataset_name, registry in DATASET_REGISTRY.items():
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
        # Build empty DataFrame with expected columns
        std_columns = ["LAT", "LON", "OBS_TIMESTAMP"]
        for vinfo in variable_map.values():
            std_columns.append(vinfo["obs_col"])
            std_columns.append(vinfo["obs_qc_col"])
            if "obs_wspd_col" in vinfo:
                std_columns.append(vinfo["obs_wspd_col"])
                std_columns.append(vinfo["obs_wdir_col"])
        std_columns = list(dict.fromkeys(std_columns))
        result = pd.DataFrame(columns=std_columns)
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

        qc_vals = obs_df[qc_col]
        bad = qc_vals.notna() & (qc_vals > max_qc_value)
        n_rejected = bad.sum()

        if "obs_wspd_col" in vinfo:
            # Wind variable: mask WSPD and WDIR source columns
            cols_to_mask = [vinfo["obs_wspd_col"], vinfo["obs_wdir_col"]]
            for col in cols_to_mask:
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


def convert_obs_units(obs_df, variable_map):
    """Apply unit conversions to observation columns as specified in variable map."""
    for forecast_var, vinfo in variable_map.items():
        conversion = vinfo["unit_conversion"]
        if conversion is not None:
            obs_col = vinfo["obs_col"]
            if obs_col in obs_df.columns:
                obs_df[obs_col] = UNIT_CONVERSIONS[conversion](obs_df[obs_col])
    return obs_df


def convert_dewpoint_to_specific_humidity(obs_df, variable_map):
    """Convert dewpoint temperature obs to specific humidity in-place.

    For variables with ``needs_dewpoint_conversion``, the obs column contains
    dewpoint temperature (K).  This function converts it to specific humidity
    (kg/kg) using the Magnus formula (consistent with MET vx_physics/thermo.cc).

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

    # Validate that surface humidity vars have surface pressure available
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
    """Compute verification metrics for a single set of forecast/obs pairs.

    Args:
        forecast_values: 1-D numpy array of forecast values at obs locations.
        obs_values: 1-D numpy array of observation values.

    Returns:
        dict with rmse, mae, bias, count as scalars.
    """
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

    # Mask locations where any member or the obs is NaN
    valid = forecast.notnull().all("member") & obs.notnull()
    forecast = forecast.where(valid)
    obs = obs.where(valid)
    n_valid = valid.sum().item()

    # Per-member metrics
    diff = forecast - obs
    per_member_vars = {
        "rmse": np.sqrt((diff**2).mean("locations")),
        "mae": np.abs(diff).mean("locations"),
        "bias": diff.mean("locations"),
    }

    # Ensemble mean metrics
    ensmean_diff = forecast.mean("member") - obs
    ensmean_metrics = {
        "rmse_ensmean": float(np.sqrt((ensmean_diff**2).mean("locations"))),
        "mae_ensmean": float(np.abs(ensmean_diff).mean("locations")),
        "bias_ensmean": float(ensmean_diff.mean("locations")),
        "count_ensmean": n_valid,
    }

    # Spread: mean across locations of per-location std dev across members
    spread_val = float(forecast.std("member").mean("locations")) if n_valid > 0 else np.nan

    # Fair CRPS: (1/N) * Σ|u_e - u*| - 1/(2*N*(N-1)) * ΣΣ|u_e - u_i|
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

    xds = xr.Dataset({**per_member_vars, **{k: v for k, v in scalar_vars.items()}})
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
    """Create dataset with subregion masks for visualization

    Note:
        This could be its own workflow, but for now we just spit it out whenever we run the obs workflow.
    """

    xds = xesmf.util.grid_global(1, 1, cf=True, lon1=360)
    for sr_name, sr_bounds in subregions.items():
        xds[sr_name] = _filter_obs_by_subregion(
            xr.ones_like(xds["lat"]*xds["lon"]).rename({"lon":"LON", "lat": "LAT"}),
            sr_bounds,
            drop_out_of_bounds=False,
        ).rename({"LON": "longitude", "LAT": "latitude"})
    return xds


def main(config):
    """Verify forecasts against observations.

    See ``eagle-tools obs-metrics --help`` or cli.py for help.
    """

    topo = config["topo"]

    # Build variable map
    variable_map = build_variable_map(config)
    forecast_var_names = list(variable_map.keys())
    # Extract unique base variable names and levels for loading the forecast
    base_var_names = list({vinfo["base_name"] for vinfo in variable_map.values()})
    levels = sorted({v["level"] for v in variable_map.values() if v["level"] is not None})
    logger.info(f"Variables to verify: {forecast_var_names}")

    # Config options
    model_type = config.get("model_type")
    lam_index = config.get("lam_index", None)
    lead_time = config["lead_time"]
    temporal_window = pd.Timedelta(config.get("temporal_window", "30min"))
    max_qc_value = config.get("max_qc_value", 2)
    n_members = config.get("n_members", 1)
    is_ensemble = n_members > 1

    # Parse subregions
    subregions = _parse_subregions(config)
    if subregions:
        logger.info(f"Subregions: {list(subregions.keys())}")
        if topo.is_root:
            srds = create_subregion_masks(subregions)
            fname = f"{config['output_path']}/subregions.nc"
            srds.to_netcdf(fname)
            logger.info(f"Stored subregion masks at {fname}")

    # does the user want to evaluate on a different grid?
    # this doesn't include regridding the nested -> global resolution
    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = (target_regrid_kwargs is not None) or \
            ((forecast_regrid_kwargs is not None) and (model_type != "nested-global"))
    if do_any_regridding:
        raise NotImplementedError

    if model_type == "nested-global":
        forecast_regrid_kwargs["target_grid_path"] = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=forecast_regrid_kwargs,
        )

    # Generate initialization dates
    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    container = {"rmse": [], "mae": [], "bias": [], "count": []}
    subregion_containers = {name: {"rmse": [], "mae": [], "bias": [], "count": []} for name in subregions}

    ENSEMBLE_METRICS = ["rmse_ensmean", "mae_ensmean", "bias_ensmean", "count_ensmean", "spread", "fcrps"]
    if is_ensemble:
        ensemble_container = {m: [] for m in ENSEMBLE_METRICS}
        ensemble_subregion_containers = {name: {m: [] for m in ENSEMBLE_METRICS} for name in subregions}

    logger.info(f"Observation Verification")
    logger.info(f"Datasets: {list(DATASET_REGISTRY.keys())}")
    logger.info(f"Temporal window: +/- {temporal_window}")
    logger.info(f"Max QC value: {max_qc_value}")
    logger.info(f"Initial Conditions:\n{dates}")

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

        # Load forecast using base variable names and levels
        member_fds_list = []
        for member in range(n_members):
            if config.get("from_anemoi", True):
                fname = f"{config['forecast_path']}/{st0}.{lead_time}h.nc"
                if is_ensemble:
                    fname = fname.replace(".nc", f".member{member:03d}.nc")
                member_fds = open_anemoi_inference_dataset(
                    fname,
                    model_type=model_type,
                    lam_index=lam_index,
                    trim_edge=config.get("trim_forecast_edge", None),
                    vars_of_interest=config.get("vars_of_interest"),
                    levels=levels,
                    load=True,
                    lcc_info=config.get("lcc_info", None),
                    horizontal_regrid_kwargs=forecast_regrid_kwargs if model_type == "nested-global" else None,
                    reshape_cell_to_2d=True,
                    rename_to_longnames=True,
                )
            else:
                member_fds = open_forecast_zarr_dataset(
                    config["forecast_path"],
                    t0=t0,
                    trim_edge=config.get("trim_forecast_edge", None),
                    vars_of_interest=config.get("vars_of_interest"),
                    levels=levels,
                    load=True,
                    lcc_info=config.get("lcc_info", None),
                    reshape_cell_to_2d=True,
                    rename_to_longnames=True,
                )
            member_fds_list.append(member_fds)

        if is_ensemble:
            fds = xr.concat(member_fds_list, dim="member")
        else:
            fds = member_fds_list[0]

        # Get forecast valid times
        forecast_valid_times = fds["time"].values

        # Load observations for the full valid time range (padded by window)
        time_start = pd.Timestamp(forecast_valid_times[0], tz="UTC") - temporal_window - pd.Timedelta("24h")
        time_end = pd.Timestamp(forecast_valid_times[-1], tz="UTC") + temporal_window + pd.Timedelta("24h")
        obs_df = load_all_observations((time_start, time_end), variable_map)

        # QC filter and unit conversion
        obs_df = apply_qc_filter(obs_df, variable_map, max_qc_value=max_qc_value)
        obs_df = convert_obs_units(obs_df, variable_map)
        obs_df = convert_dewpoint_to_specific_humidity(obs_df, variable_map)
        obs_df = derive_wind_components(obs_df, variable_map)

        # Convert obs longitudes to 0-360 to match forecast convention
        obs_df["LON"] = obs_df["LON"] % 360

        # Align observations to forecast valid times
        aligned = align_obs_to_forecast_times(obs_df, forecast_valid_times, temporal_window)

        # Compute metrics per forecast valid time
        container_per_ic = {metric: {varname: [] for varname in forecast_var_names} for metric in container.keys()}
        subregion_container_per_ic = {
            sr_name: {metric: {varname: [] for varname in forecast_var_names} for metric in container.keys()}
            for sr_name in subregions
        }
        if is_ensemble:
            ensemble_container_per_ic = {metric: {varname: [] for varname in forecast_var_names} for metric in ENSEMBLE_METRICS}
            ensemble_subregion_container_per_ic = {
                sr_name: {metric: {varname: [] for varname in forecast_var_names} for metric in ENSEMBLE_METRICS}
                for sr_name in subregions
            }

        for vtime in forecast_valid_times:
            vtimestamp = pd.Timestamp(vtime, tz="UTC")

            # Handle case where we don't have obs
            if vtimestamp not in aligned:
                if is_ensemble:
                    empty_fvals = np.full((n_members, 1), np.nan)
                else:
                    empty_fvals = np.array([np.nan])
                empty_result = compute_obs_metrics(empty_fvals, np.array([np.nan]), vtime)
                for varname in forecast_var_names:
                    for metric in container.keys():
                        container_per_ic[metric][varname].append(empty_result[metric])
                        for sr_name in subregions:
                            subregion_container_per_ic[sr_name][metric][varname].append(empty_result[metric])
                    if is_ensemble:
                        for metric in ENSEMBLE_METRICS:
                            ensemble_container_per_ic[metric][varname].append(empty_result[metric])
                            for sr_name in subregions:
                                ensemble_subregion_container_per_ic[sr_name][metric][varname].append(empty_result[metric])
                continue

            matched_obs = aligned[vtimestamp]

            # Interp to obs locations and compute metrics
            interpolated = _interp_to_obs_locations(fds.sel(time=vtime), matched_obs)

            for varname, vinfo in variable_map.items():
                fvals = interpolated[vinfo["base_name"]]
                if "level" in fvals.dims:
                    fvals = fvals.sel(level=vinfo["level"])

                result = compute_obs_metrics(
                    fvals.values,
                    matched_obs[vinfo["obs_col"]].values,
                    vtime,
                )
                for metric in container.keys():
                    container_per_ic[metric][varname].append(result[metric])
                if is_ensemble:
                    for metric in ENSEMBLE_METRICS:
                        ensemble_container_per_ic[metric][varname].append(result[metric])

            # Subregion metrics
            for sr_name, sr_bounds in subregions.items():
                sr_obs = _filter_obs_by_subregion(matched_obs, sr_bounds)
                if len(sr_obs) == 0:
                    if is_ensemble:
                        empty_fvals = np.full((n_members, 1), np.nan)
                    else:
                        empty_fvals = np.array([np.nan])
                    sr_empty_result = compute_obs_metrics(empty_fvals, np.array([np.nan]), vtime)
                    for varname in forecast_var_names:
                        for metric in container.keys():
                            subregion_container_per_ic[sr_name][metric][varname].append(sr_empty_result[metric])
                        if is_ensemble:
                            for metric in ENSEMBLE_METRICS:
                                ensemble_subregion_container_per_ic[sr_name][metric][varname].append(sr_empty_result[metric])
                else:
                    sr_interpolated = _interp_to_obs_locations(fds.sel(time=vtime), sr_obs)
                    for varname, vinfo in variable_map.items():
                        fvals = sr_interpolated[vinfo["base_name"]]
                        if "level" in fvals.dims:
                            fvals = fvals.sel(level=vinfo["level"])
                        sr_result = compute_obs_metrics(
                            fvals.values,
                            sr_obs[vinfo["obs_col"]].values,
                            vtime,
                        )
                        for metric in container.keys():
                            subregion_container_per_ic[sr_name][metric][varname].append(sr_result[metric])
                        if is_ensemble:
                            for metric in ENSEMBLE_METRICS:
                                ensemble_subregion_container_per_ic[sr_name][metric][varname].append(sr_result[metric])

        # Assemble into xr.Dataset, grouping upper-air variables by base
        # name with a level dimension
        for metric, thedata in container_per_ic.items():
            base_groups = {}
            for varname, vals in thedata.items():
                vinfo = variable_map[varname]
                bn = vinfo["base_name"]
                level = vinfo["level"]
                time_concat = xr.concat(vals, dim="time")
                if bn not in base_groups:
                    base_groups[bn] = {}
                base_groups[bn][level] = time_concat

            data_vars = {}
            for bn, level_dict in base_groups.items():
                if None in level_dict:
                    # Surface variable — no level dimension
                    data_vars[bn] = level_dict[None]
                else:
                    # Upper-air variable — stack levels
                    level_arrays = []
                    for lvl in sorted(level_dict.keys()):
                        arr = level_dict[lvl].expand_dims({"level": [lvl]})
                        level_arrays.append(arr)
                    data_vars[bn] = xr.concat(level_arrays, dim="level")

            this_metric_ds = postprocess(xr.Dataset(data_vars))
            container[metric].append(this_metric_ds)

        # Assemble subregion metrics
        for sr_name, sr_cpi in subregion_container_per_ic.items():
            for metric, thedata in sr_cpi.items():
                base_groups = {}
                for varname, vals in thedata.items():
                    vinfo = variable_map[varname]
                    bn = vinfo["base_name"]
                    level = vinfo["level"]
                    time_concat = xr.concat(vals, dim="time")
                    if bn not in base_groups:
                        base_groups[bn] = {}
                    base_groups[bn][level] = time_concat

                data_vars = {}
                for bn, level_dict in base_groups.items():
                    if None in level_dict:
                        data_vars[bn] = level_dict[None]
                    else:
                        level_arrays = []
                        for lvl in sorted(level_dict.keys()):
                            arr = level_dict[lvl].expand_dims({"level": [lvl]})
                            level_arrays.append(arr)
                        data_vars[bn] = xr.concat(level_arrays, dim="level")

                this_metric_ds = postprocess(xr.Dataset(data_vars))
                subregion_containers[sr_name][metric].append(this_metric_ds)

        # Assemble ensemble metrics
        if is_ensemble:
            for metric, thedata in ensemble_container_per_ic.items():
                base_groups = {}
                for varname, vals in thedata.items():
                    vinfo = variable_map[varname]
                    bn = vinfo["base_name"]
                    level = vinfo["level"]
                    time_concat = xr.concat(vals, dim="time")
                    if bn not in base_groups:
                        base_groups[bn] = {}
                    base_groups[bn][level] = time_concat

                data_vars = {}
                for bn, level_dict in base_groups.items():
                    if None in level_dict:
                        data_vars[bn] = level_dict[None]
                    else:
                        level_arrays = []
                        for lvl in sorted(level_dict.keys()):
                            arr = level_dict[lvl].expand_dims({"level": [lvl]})
                            level_arrays.append(arr)
                        data_vars[bn] = xr.concat(level_arrays, dim="level")

                this_metric_ds = postprocess(xr.Dataset(data_vars))
                ensemble_container[metric].append(this_metric_ds)

            # Assemble ensemble subregion metrics
            for sr_name, sr_cpi in ensemble_subregion_container_per_ic.items():
                for metric, thedata in sr_cpi.items():
                    base_groups = {}
                    for varname, vals in thedata.items():
                        vinfo = variable_map[varname]
                        bn = vinfo["base_name"]
                        level = vinfo["level"]
                        time_concat = xr.concat(vals, dim="time")
                        if bn not in base_groups:
                            base_groups[bn] = {}
                        base_groups[bn][level] = time_concat

                    data_vars = {}
                    for bn, level_dict in base_groups.items():
                        if None in level_dict:
                            data_vars[bn] = level_dict[None]
                        else:
                            level_arrays = []
                            for lvl in sorted(level_dict.keys()):
                                arr = level_dict[lvl].expand_dims({"level": [lvl]})
                                level_arrays.append(arr)
                            data_vars[bn] = xr.concat(level_arrays, dim="level")

                    this_metric_ds = postprocess(xr.Dataset(data_vars))
                    ensemble_subregion_containers[sr_name][metric].append(this_metric_ds)

        logger.info(f"Done with {st0}")

    logger.info("Done Computing Observation Verification Metrics")

    logger.info("Gathering Results on Root Process")
    for name in container.keys():
        container[name] = topo.gather(container[name])
    for sr_name in subregions:
        for name in subregion_containers[sr_name].keys():
            subregion_containers[sr_name][name] = topo.gather(subregion_containers[sr_name][name])
    if is_ensemble:
        for name in ensemble_container.keys():
            ensemble_container[name] = topo.gather(ensemble_container[name])
        for sr_name in subregions:
            for name in ensemble_subregion_containers[sr_name].keys():
                ensemble_subregion_containers[sr_name][name] = topo.gather(ensemble_subregion_containers[sr_name][name])

    if topo.is_root:
        for name in container:
            c = container[name]
            if config["use_mpi"]:
                c = [xds for sublist in c for xds in sublist]
            c = sorted(c, key=lambda xds: xds.coords["t0"])
            container[name] = xr.concat(c, dim="t0")

        for metric, xds in container.items():
            fname = f"{config['output_path']}/{metric}.convobs.{model_type}.nc"
            xds.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")

        for sr_name, sr_container in subregion_containers.items():
            for name in sr_container:
                c = sr_container[name]
                if config["use_mpi"]:
                    c = [xds for sublist in c for xds in sublist]
                c = sorted(c, key=lambda xds: xds.coords["t0"])
                sr_container[name] = xr.concat(c, dim="t0")

            for metric, xds in sr_container.items():
                fname = f"{config['output_path']}/{metric}.convobs.{model_type}.{sr_name}.nc"
                xds.to_netcdf(fname)
                logger.info(f"Stored result: {fname}")

        # Write ensemble metric files
        if is_ensemble:
            for name in ensemble_container:
                c = ensemble_container[name]
                if config["use_mpi"]:
                    c = [xds for sublist in c for xds in sublist]
                c = sorted(c, key=lambda xds: xds.coords["t0"])
                ensemble_container[name] = xr.concat(c, dim="t0")

            for metric, xds in ensemble_container.items():
                fname = f"{config['output_path']}/{metric}.convobs.{model_type}.nc"
                xds.to_netcdf(fname)
                logger.info(f"Stored result: {fname}")

            for sr_name, sr_container in ensemble_subregion_containers.items():
                for name in sr_container:
                    c = sr_container[name]
                    if config["use_mpi"]:
                        c = [xds for sublist in c for xds in sublist]
                    c = sorted(c, key=lambda xds: xds.coords["t0"])
                    sr_container[name] = xr.concat(c, dim="t0")

                for metric, xds in sr_container.items():
                    fname = f"{config['output_path']}/{metric}.convobs.{model_type}.{sr_name}.nc"
                    xds.to_netcdf(fname)
                    logger.info(f"Stored result: {fname}")

        logger.info("Done Storing Observation Verification Metrics")
