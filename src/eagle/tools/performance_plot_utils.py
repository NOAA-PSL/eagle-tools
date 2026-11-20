from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

logger = logging.getLogger("eagle.tools")

DEFAULT_REGION_DIM_CANDIDATES = [
    "region",
    "mask",
    "vx_mask",
    "mask_name",
    "stat_region",
    "domain",
    "area",
]

DEFAULT_FHR_CANDIDATES = [
    "fhr",
    "lead",
    "lead_time",
    "forecast_hour",
    "fhour",
]

DEFAULT_LEVEL_CANDIDATES = [
    "level",
    "lev",
    "pressure",
    "pressure_level",
    "plev",
    "isobaricInhPa",
]

DEFAULT_MODELS = {
    "nested_eagle_global": {
        "label": "Nested-EAGLE-Global (AI)",
        "plot_label": "Nested-EAGLE",
        "plot_sublabel": "Global AI",
        "region_labels": {
            "global": "Nested-EAGLE\n(Global)",
            "conus": "Nested-EAGLE\n(CONUS)",
        },
        "color": "royalblue",
        "directory": "nested_eagle_global_2025",
        "model_type": "nested-global",
        "global_patterns": [
            "{metric}.convobs.nested-global.nc",
            "{metric}.convobs.global.nc",
        ],
        "regional_patterns": [
            "{metric}.convobs.nested-global.{region}.nc",
            "{metric}.convobs.global.{region}.nc",
        ],
    },
    "nested_eagle_lam": {
        "label": "Nested-EAGLE-LAM (AI-HR)",
        "plot_label": "Nested-EAGLE",
        "plot_sublabel": "LAM AI-HR",
        "color": "dodgerblue",
        "directory": "nested_eagle_lam_2025",
        "default_pattern": "{metric}.convobs.nested-lam.nc",
    },
    "gfs": {
        "label": "GFS",
        "color": "black",
        "directory": "gfs_2025",
        "model_type": "global",
        "global_pattern": "{metric}.convobs.global.nc",
        "regional_pattern": "{metric}.convobs.global.{region}.nc",
    },
    "aigfs": {
        "label": "AIGFS",
        "color": "forestgreen",
        "directory": "aigfs_2025",
        "model_type": "global",
        "global_pattern": "{metric}.convobs.global.nc",
        "regional_pattern": "{metric}.convobs.global.{region}.nc",
    },
    "aifs": {
        "label": "AIFS",
        "color": "firebrick",
        "directory": "aifs_2025",
        "model_type": "global",
        "global_pattern": "{metric}.convobs.global.nc",
        "regional_pattern": "{metric}.convobs.global.{region}.nc",
    },
    "ecmwf_ifs": {
        "label": "ECMWF IFS",
        "plot_label": "ECMWF IFS",
        "color": "purple",
        "directory": "ecmwf_ifs_2025",
        "model_type": "global",
        "global_pattern": "{metric}.convobs.global.nc",
        "regional_pattern": "{metric}.convobs.global.{region}.nc",
    },
    "hrrr": {
        "label": "HRRR",
        "color": "darkorange",
        "directory": "hrrr_2025",
        "default_pattern": "{metric}.convobs.lam.nc",
    },
}

DEFAULT_REGIONS = {
    "global": {
        "label": "Global",
        "file_token": "global",
        "aliases": ["global", "GLOBAL", "all", "ALL"],
    },
    "northern_hemisphere": {
        "label": "Northern Hemisphere",
        "file_token": "northern_hemisphere",
        "aliases": ["northern_hemisphere", "northern hemisphere", "north_hemisphere", "nh", "NH", "nhem", "NHEM"],
    },
    "southern_hemisphere": {
        "label": "Southern Hemisphere",
        "file_token": "southern_hemisphere",
        "aliases": ["southern_hemisphere", "southern hemisphere", "south_hemisphere", "sh", "SH", "shem", "SHEM"],
    },
    "conus": {
        "label": "CONUS",
        "file_token": "conus",
        "aliases": ["conus", "CONUS", "us", "US"],
    },
}

DEFAULT_VARIABLES_V2 = {
    "surface": [
        {"key": "2m_temperature", "label": "2m Temperature", "units": "K", "enabled": True},
        {"key": "10m_zonal_wind", "label": "10m Zonal Wind", "units": "m/s", "enabled": True},
        {"key": "10m_meridional_wind", "label": "10m Meridional Wind", "units": "m/s", "enabled": True},
        {"key": "10m_wind_speed", "label": "10m Wind Speed", "units": "m/s", "enabled": True},
    ],
    "upper": [
        {"key": "geopotential_height", "label": "Geopotential Height", "units": "m", "levels": [250, 500, 850], "enabled": True},
        {"key": "zonal_wind", "label": "Zonal Wind", "units": "m/s", "levels": [250, 500, 850], "enabled": True},
        {"key": "meridional_wind", "label": "Meridional Wind", "units": "m/s", "levels": [250, 500, 850], "enabled": True},
        {"key": "wind_speed", "label": "Wind Speed", "units": "m/s", "levels": [250, 500, 850], "enabled": True},
        {"key": "temperature", "label": "Temperature", "units": "K", "levels": [250, 500, 850], "enabled": True},
        {"key": "specific_humidity", "label": "Specific Humidity", "units": "kg/kg", "levels": [250, 500, 850], "enabled": True},
    ],
}

DEFAULT_VARIABLE_ALIASES = {
    "geopotential_height": ["geopotential_height", "geopotential", "gh", "z"],
    "zonal_wind": ["zonal_wind", "u_wind", "u_component_of_wind", "u"],
    "meridional_wind": ["meridional_wind", "v_wind", "v_component_of_wind", "v"],
    "temperature": ["temperature", "tmp", "t"],
    "specific_humidity": ["specific_humidity", "q", "spfh"],
    "wind_speed": ["wind_speed", "wind"],
    "surface_pressure": ["surface_pressure", "pressure_surface", "sp", "ps"],
    "10m_zonal_wind": ["10m_zonal_wind", "u10", "10u"],
    "10m_meridional_wind": ["10m_meridional_wind", "v10", "10v"],
    "2m_temperature": ["2m_temperature", "t2m", "2t"],
    "2m_specific_humidity": ["2m_specific_humidity", "q2m", "2m_q"],
    "10m_wind_speed": ["10m_wind_speed", "wind_speed_10m", "ws10"],
}


def load_config(config_filename: str) -> dict[str, Any]:
    suffix = Path(config_filename).suffix.lower()
    with open(config_filename, "r") as f:
        if suffix == ".json":
            config = json.load(f)
        else:
            if yaml is None:
                raise ModuleNotFoundError(
                    "PyYAML is required to read YAML config files. "
                    "Install pyyaml or use a .json config file."
                )
            config = yaml.safe_load(f)
    _expand_paths(config)
    return config


def _expand_paths(value: Any) -> Any:
    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, str) and "path" in key:
                value[key] = os.path.expandvars(val)
            else:
                _expand_paths(val)
    elif isinstance(value, list):
        for item in value:
            _expand_paths(item)
    return value


def get_output_path(config: dict[str, Any], plot_config: dict[str, Any]) -> Path:
    output_path = Path(plot_config.get("output_path", config.get("output_path", "outputs")))
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_model_path(config: dict[str, Any], model_key: str) -> Path:
    info = model_info(config, model_key)
    if "path" in info:
        return Path(info["path"])

    input_path = Path(config.get("input_path", config.get("data_base")))
    subdir = info.get("subdir", info.get("directory", model_key))
    return input_path / subdir


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def normalized_name(value: Any) -> str:
    return decode_value(value).strip().lower().replace("-", "_").replace(" ", "_")


def slug(value: Any) -> str:
    value = decode_value(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_name(names: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def model_info(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    info = dict(DEFAULT_MODELS.get(model_key, {}))
    info.update(config.get("models", {}).get(model_key, {}))
    return info


def model_label(config: dict[str, Any], model_key: str, region_key: str | None = None) -> str:
    info = model_info(config, model_key)
    labels = info.get("region_labels", {})
    if region_key is not None and region_key in labels:
        return labels[region_key]
    if "plot_label" in info:
        sublabel = info.get("plot_sublabel", "")
        return f"{info['plot_label']}\n{sublabel}" if sublabel else info["plot_label"]
    return info.get("label", model_key)


def model_color(config: dict[str, Any], model_key: str) -> str:
    return model_info(config, model_key).get("color", "gray")


def region_label(config: dict[str, Any], region: dict[str, Any] | str) -> str:
    if isinstance(region, dict):
        return region.get("title") or region.get("label") or region["key"]
    return config.get("regions", {}).get(region, {}).get("label", region)


def region_info(config: dict[str, Any], region: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(region, dict):
        return region

    regions = {**DEFAULT_REGIONS, **config.get("regions", {})}
    info = regions.get(region, {}) if isinstance(regions, dict) else {}
    aliases = info.get("aliases", [])
    file_token = info.get("file_token", region)
    return {
        "key": region,
        "title": info.get("title", info.get("label", region)),
        "aliases": [region, file_token, *aliases],
    }


def region_aliases(region: dict[str, Any]) -> list[str]:
    aliases = list(region.get("aliases", []))
    key = region.get("key")
    if key is not None:
        aliases.append(key)
    return [normalized_name(alias) for alias in aliases]


def iter_variable_rows(config: dict[str, Any], skip_levels: list[int] | None = None):
    skip_levels = set(skip_levels or [])
    group_breaks = []
    rows = []
    if "variables" not in config:
        variables_v2 = config.get("variables_v2", DEFAULT_VARIABLES_V2)
        for item in variables_v2.get("surface", []):
            if item.get("enabled", True):
                rows.append({"var": item["key"], "label": item.get("label", item["key"]), "levels": None})
        group_breaks.append(len(rows) - 0.5)
        for item in variables_v2.get("upper", []):
            if not item.get("enabled", True):
                continue
            for level in item.get("levels", []):
                if int(level) not in skip_levels:
                    rows.append({
                        "var": item["key"],
                        "label": f"{item.get('label', item['key'])} {int(level)}",
                        "levels": [int(level)],
                    })
            group_breaks.append(len(rows) - 0.5)
        return rows, group_breaks[:-1]

    for group in config.get("variables", []):
        for row in group.get("rows", []):
            levels = row.get("levels")
            if levels is not None and len(levels) == 1 and int(levels[0]) in skip_levels:
                continue
            rows.append(row)
        group_breaks.append(len(rows) - 0.5)
    return rows, group_breaks[:-1]


def open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)


def candidate_files(
    config: dict[str, Any],
    model_key: str,
    region_key: str,
    metric: str,
) -> list[Path]:
    info = model_info(config, model_key)
    directory = get_model_path(config, model_key)
    pattern_key = "global_pattern" if region_key == "global" else "regional_pattern"
    patterns_key = "global_patterns" if region_key == "global" else "regional_patterns"
    pattern = info.get(pattern_key)
    patterns = [pattern] if pattern is not None else info.get(patterns_key)
    if patterns is None:
        pattern = info.get("default_pattern")
        patterns = [pattern] if pattern is not None else info.get("default_patterns", info.get("patterns", []))

    files = []
    for pattern in patterns:
        rendered = pattern.format(
            metric=metric,
            model=model_key,
            model_type=info.get("model_type", model_key),
            region=region_key,
        )
        files.extend(sorted(directory.glob(rendered)))
    return sorted(set(files))


def find_var(ds: xr.Dataset, config: dict[str, Any], var_key: str) -> str | None:
    variable_aliases = {**DEFAULT_VARIABLE_ALIASES, **config.get("variable_aliases", {})}
    aliases = variable_aliases.get(var_key, [var_key])
    for name in aliases:
        if name in ds.data_vars:
            return name
    return None


def normalize_fhr(da: xr.DataArray, config: dict[str, Any]) -> xr.DataArray | None:
    candidates = config.get("fhr_candidates", DEFAULT_FHR_CANDIDATES)
    fhr_name = find_name(list(da.coords) + list(da.dims), candidates)
    if fhr_name is None:
        return None
    if fhr_name != "fhr":
        da = da.rename({fhr_name: "fhr"})
    values = np.asarray(da["fhr"].values)
    if np.issubdtype(values.dtype, np.timedelta64):
        da = da.assign_coords(fhr=(values / np.timedelta64(1, "h")).astype(int))
    return da


def normalize_level(da: xr.DataArray, config: dict[str, Any]) -> tuple[xr.DataArray, str | None]:
    candidates = config.get("level_candidates", DEFAULT_LEVEL_CANDIDATES)
    level_name = find_name(list(da.coords) + list(da.dims), candidates)
    if level_name is None:
        return da, None
    if level_name != "level":
        da = da.rename({level_name: "level"})
    return da, "level"


def select_region(
    da: xr.DataArray,
    config: dict[str, Any],
    region: dict[str, Any],
    allow_no_region: bool,
) -> xr.DataArray | None:
    candidates = config.get("region_dim_candidates", DEFAULT_REGION_DIM_CANDIDATES)
    region_dim = find_name(list(da.coords) + list(da.dims), candidates)
    if region_dim is None:
        return da if allow_no_region else None

    values = [normalized_name(value) for value in da[region_dim].values]
    aliases = region_aliases(region)
    for idx, value in enumerate(values):
        if value in aliases:
            return da.sel({region_dim: da[region_dim].values[idx]})
    return None


def select_level(
    da: xr.DataArray,
    config: dict[str, Any],
    levels: list[int] | None,
) -> xr.DataArray | None:
    da, level_name = normalize_level(da, config)
    if levels is None:
        if level_name is not None and "level" in da.dims:
            return None
        return da

    if level_name is None:
        return None

    have = [int(x) for x in da["level"].values]
    for level in levels:
        if int(level) in have:
            return da.sel(level=int(level))
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def temporal_filter_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    temporal = config.get("temporal", {})
    return {
        "start_date": config.get("start_date", temporal.get("start_date")),
        "end_date": config.get("end_date", temporal.get("end_date")),
        "years": config.get("years", temporal.get("years")),
        "months": config.get("months", temporal.get("months")),
    }


def describe_temporal_filter(config: dict[str, Any]) -> str:
    parts = []
    kwargs = temporal_filter_kwargs(config)
    if kwargs["start_date"] is not None:
        parts.append(f"start_date={kwargs['start_date']}")
    if kwargs["end_date"] is not None:
        parts.append(f"end_date={kwargs['end_date']}")
    if kwargs["years"] is not None:
        parts.append(f"years={_as_list(kwargs['years'])}")
    if kwargs["months"] is not None:
        parts.append(f"months={_as_list(kwargs['months'])}")
    return ", ".join(parts) if parts else "all available overlapping times"


def apply_temporal_filter(da: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    if "t0" not in da.coords:
        if any(value is not None for value in temporal_filter_kwargs(config).values()):
            logger.warning("Temporal filtering was requested, but data has no t0 coordinate")
        return da

    kwargs = temporal_filter_kwargs(config)
    t0 = da["t0"]
    mask = xr.ones_like(t0, dtype=bool)

    if kwargs["start_date"] is not None:
        mask = mask & (t0 >= np.datetime64(kwargs["start_date"]))
    if kwargs["end_date"] is not None:
        mask = mask & (t0 <= np.datetime64(kwargs["end_date"]))
    if kwargs["years"] is not None:
        years = [int(year) for year in _as_list(kwargs["years"])]
        mask = mask & t0.dt.year.isin(years)
    if kwargs["months"] is not None:
        months = [int(month) for month in _as_list(kwargs["months"])]
        mask = mask & t0.dt.month.isin(months)

    t0_dims = t0.dims
    if len(t0_dims) != 1:
        logger.warning("Cannot apply temporal filter to non-1D t0 coordinate with dims=%s", t0_dims)
        return da

    dim = t0_dims[0]
    return da.where(mask, drop=True) if dim not in da.dims else da.isel({dim: mask})


def combine_metric_arrays(arrays: list[xr.DataArray]) -> xr.DataArray:
    if len(arrays) == 1:
        return arrays[0]

    if all("t0" in arr.dims for arr in arrays):
        return xr.concat(
            arrays,
            dim="t0",
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="drop_conflicts",
        ).sortby("t0")

    return xr.concat(
        arrays,
        dim="sample_file",
        coords="minimal",
        compat="override",
        join="outer",
        combine_attrs="drop_conflicts",
    )


def align_metric_arrays(
    arrays: list[xr.DataArray],
    require_exact_time_match: bool = True,
) -> list[xr.DataArray]:
    if not arrays:
        return []

    join = "inner" if require_exact_time_match else "outer"
    aligned = xr.align(*arrays, join=join)
    if require_exact_time_match:
        if "t0" in aligned[0].sizes and aligned[0].sizes["t0"] == 0:
            raise ValueError("No overlapping t0 values found across selected models")
        if "fhr" in aligned[0].sizes and aligned[0].sizes["fhr"] == 0:
            raise ValueError("No overlapping forecast hours found across selected models")
    return list(aligned)


def apply_common_finite_mask(arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    if not arrays:
        return []
    mask = xr.ones_like(arrays[0], dtype=bool)
    for da in arrays:
        mask = mask & np.isfinite(da)
    return [da.where(mask) for da in arrays]


def load_metric_array(
    config: dict[str, Any],
    model_key: str,
    metric: str,
    variable: str,
    region: dict[str, Any],
    level: int | None,
    fhrs: list[int],
) -> xr.DataArray | None:
    files = candidate_files(config, model_key, region["key"], metric)
    if not files:
        logger.warning("No files found for model=%s region=%s", model_key, region["key"])
        return None

    arrays = []
    allow_no_region = region["key"] == "global"
    if region["key"] != "global":
        allow_no_region = True

    for path in files:
        ds = open_dataset(path)
        try:
            ds_var = find_var(ds, config, variable)
            if ds_var is None:
                continue

            da = normalize_fhr(ds[ds_var], config)
            if da is None or "fhr" not in da.coords:
                continue

            da = select_region(da, config, region, allow_no_region=allow_no_region)
            if da is None:
                continue

            levels = None if level is None else [level]
            da = select_level(da, config, levels)
            if da is None:
                continue

            have_fhr = [int(x) for x in da["fhr"].values]
            use_fhr = [int(fhr) for fhr in fhrs if int(fhr) in have_fhr]
            if not use_fhr:
                continue

            da = da.sel(fhr=use_fhr).reindex(fhr=fhrs)
            arrays.append(da.load())
        finally:
            ds.close()

    if not arrays:
        logger.warning(
            "No usable data for model=%s region=%s variable=%s level=%s",
            model_key,
            region["key"],
            variable,
            level,
        )
        return None

    return apply_temporal_filter(combine_metric_arrays(arrays), config)


def matched_mean_series(
    config: dict[str, Any],
    candidate_model: str,
    baseline_model: str,
    metric: str,
    row: dict[str, Any],
    region: dict[str, Any],
    fhrs: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    levels = row.get("levels")
    level = None if levels is None else int(levels[0])
    candidate = load_metric_array(config, candidate_model, metric, row["var"], region, level, fhrs)
    baseline = load_metric_array(config, baseline_model, metric, row["var"], region, level, fhrs)
    if candidate is None or baseline is None:
        missing = np.full(len(fhrs), np.nan)
        return missing, missing

    candidate, baseline = align_metric_arrays(
        [candidate, baseline],
        require_exact_time_match=config.get("require_exact_time_match", True),
    )
    candidate, baseline = apply_common_finite_mask([candidate, baseline])
    mean_dims = [dim for dim in candidate.dims if dim != "fhr"]
    candidate_vals = np.asarray(candidate.mean(mean_dims, skipna=True).reindex(fhr=fhrs).values, dtype=float).squeeze()
    baseline_vals = np.asarray(baseline.mean(mean_dims, skipna=True).reindex(fhr=fhrs).values, dtype=float).squeeze()
    if candidate_vals.shape != (len(fhrs),) or baseline_vals.shape != (len(fhrs),):
        logger.warning(
            "Bad matched shape for %s vs %s %s %s: %s %s",
            candidate_model,
            baseline_model,
            region["key"],
            row["label"],
            candidate_vals.shape,
            baseline_vals.shape,
        )
        missing = np.full(len(fhrs), np.nan)
        return missing, missing
    return candidate_vals, baseline_vals


def mean_series(
    config: dict[str, Any],
    model_key: str,
    metric: str,
    row: dict[str, Any],
    region: dict[str, Any],
    fhrs: list[int],
) -> np.ndarray:
    levels = row.get("levels")
    level = None if levels is None else int(levels[0])
    da = load_metric_array(config, model_key, metric, row["var"], region, level, fhrs)
    if da is None:
        return np.full(len(fhrs), np.nan)
    mean_dims = [dim for dim in da.dims if dim != "fhr"]
    vals = da.mean(mean_dims, skipna=True).reindex(fhr=fhrs).values
    vals = np.asarray(vals, dtype=float).squeeze()
    if vals.shape != (len(fhrs),):
        logger.warning("Bad shape for %s %s %s: %s", model_key, region["key"], row["label"], vals.shape)
        return np.full(len(fhrs), np.nan)
    return vals


def response_values(
    config: dict[str, Any],
    model_key: str,
    metric: str,
    variable: str,
    region: dict[str, Any],
    level: int | None,
    fhrs: list[int],
) -> np.ndarray:
    da = load_metric_array(config, model_key, metric, variable, region, level, fhrs)
    if da is None:
        return np.array([], dtype=float)
    vals = np.asarray(da.values, dtype=float).ravel()
    return vals[np.isfinite(vals)]


def matched_response_values(
    config: dict[str, Any],
    model_keys: list[str],
    metric: str,
    variable: str,
    region: dict[str, Any],
    level: int | None,
    fhrs: list[int],
) -> dict[str, np.ndarray]:
    arrays = {}
    for model_key in model_keys:
        da = load_metric_array(config, model_key, metric, variable, region, level, fhrs)
        if da is None:
            return {}
        arrays[model_key] = da

    aligned = align_metric_arrays(
        list(arrays.values()),
        require_exact_time_match=config.get("require_exact_time_match", True),
    )
    aligned = apply_common_finite_mask(aligned)
    return {
        model_key: np.asarray(da.values, dtype=float).ravel()
        for model_key, da in zip(arrays, aligned)
    }


def summarize(vals: np.ndarray) -> dict[str, float | int]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(vals.size),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
