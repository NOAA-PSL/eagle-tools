import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eagle.tools.performance_plot_utils import (
    get_output_path,
    iter_variable_rows,
    load_config,
    matched_mean_series,
    model_label,
    model_info,
    region_info,
    slug,
)

logger = logging.getLogger("eagle.tools")


def _forecast_hours(plot_config):
    if "forecast_hours" in plot_config:
        fhrs = [int(fhr) for fhr in plot_config["forecast_hours"]]
    else:
        days = plot_config.get("days", list(range(1, 11)))
        fhrs = [24 * int(day) for day in days]
    labels = plot_config.get("xlabels") or [f"D{int(fhr / 24)}" for fhr in fhrs]
    return fhrs, labels


def _default_regions():
    return ["global", "northern_hemisphere", "southern_hemisphere", "conus"]


def _lead_label(fhrs):
    if all(fhr % 24 == 0 for fhr in fhrs):
        return f"d{int(fhrs[0] / 24)}-d{int(fhrs[-1] / 24)}"
    return f"f{fhrs[0]:03d}-f{fhrs[-1]:03d}"


def _region_label_for_filename(config, regions):
    keys = [region_info(config, region)["key"] for region in regions]
    abbreviations = {
        "global": "global",
        "northern_hemisphere": "nh",
        "southern_hemisphere": "sh",
        "conus": "conus",
    }
    return "-".join(abbreviations.get(key, slug(key)) for key in keys)


def _default_output_filename(config, plot_config, fhrs):
    regions = plot_config.get("regions", _default_regions())
    region_part = _region_label_for_filename(config, regions)
    candidate = slug(plot_config["candidate_model"])
    baseline = slug(plot_config["baseline_model"])
    metric = slug(plot_config.get("metric", config.get("metric", "rmse")))
    return f"heatmap_regions-{region_part}_models-{candidate}-vs-{baseline}_{metric}_{_lead_label(fhrs)}.png"


def _box_text_color(value, vmax):
    if not np.isfinite(value) or vmax <= 0:
        return "black"
    return "white" if abs(value) >= 0.60 * vmax else "black"


def _single_line_model_label(config, model_key):
    return model_info(config, model_key).get("label", model_label(config, model_key).replace("\n", " "))


def _build_region_matrix(config, plot_config, region, rows, fhrs):
    region = region_info(config, region)
    metric = plot_config.get("metric", config.get("metric", "rmse"))
    candidate = plot_config["candidate_model"]
    baseline = plot_config["baseline_model"]
    matrix = np.full((len(rows), len(fhrs)), np.nan)

    for idx, row in enumerate(rows):
        logger.info("Region=%s row=%s", region["key"], row["label"])
        candidate_vals, baseline_vals = matched_mean_series(
            config,
            candidate_model=candidate,
            baseline_model=baseline,
            metric=metric,
            row=row,
            region=region,
            fhrs=fhrs,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            improvement = 100.0 * (baseline_vals - candidate_vals) / baseline_vals
        improvement[~np.isfinite(improvement)] = np.nan
        matrix[idx, :] = improvement
    return matrix


def _plot_heatmap(config, plot_config, region_mats, rows, group_breaks, fhrs, xlabels):
    regions = plot_config.get("regions", _default_regions())
    nrows = len(rows)
    ncols = len(fhrs)
    figsize = plot_config.get("figsize", [21, 10.5])
    vmin, vmax = plot_config.get("value_range", [-30.0, 30.0])

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(regions),
        figsize=figsize,
        sharey=True,
        squeeze=False,
        gridspec_kw=plot_config.get(
            "gridspec",
            {
                "left": 0.105,
                "right": 0.925,
                "top": 0.86,
                "bottom": 0.12,
                "wspace": 0.055,
            },
        ),
    )
    axes = axes[0]

    candidate = _single_line_model_label(config, plot_config["candidate_model"])
    baseline = _single_line_model_label(config, plot_config["baseline_model"])
    title = plot_config.get("title", f"{candidate} vs {baseline} | Global 0.25 degree data")
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.975)
    cmap = plt.get_cmap(plot_config.get("cmap", "RdBu")).copy()
    cmap.set_bad(plot_config.get("missing_color", "#f2f2f2"))

    im = None
    for ax, raw_region in zip(axes, regions):
        region = region_info(config, raw_region)
        mat = region_mats[region["key"]]
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(region.get("title", region["key"]), fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in group_breaks:
            ax.axhline(y, color="#333333", linewidth=1.15)

        if plot_config.get("annotate", True):
            for ii in range(nrows):
                for jj in range(ncols):
                    value = mat[ii, jj]
                    if np.isfinite(value):
                        ax.text(
                            jj,
                            ii,
                            f"{value:.1f}",
                            ha="center",
                            va="center",
                            fontsize=6.7,
                            color=_box_text_color(value, vmax),
                            clip_on=True,
                        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.9)

    axes[0].set_yticks(np.arange(nrows))
    axes[0].set_yticklabels([row["label"] for row in rows], fontsize=10)
    axes[0].tick_params(axis="y", length=0, pad=6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)

    cax = fig.add_axes(plot_config.get("colorbar_axes", [0.945, 0.18, 0.022, 0.70]))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(plot_config.get("colorbar_label", "RMSE improvement (%)"), fontsize=12, rotation=90, labelpad=16)
    cb.ax.tick_params(labelsize=10)

    footer = plot_config.get("footer")
    if footer is None:
        baseline = _single_line_model_label(config, plot_config["baseline_model"])
        candidate = _single_line_model_label(config, plot_config["candidate_model"])
        footer = (
            f"RMSE improvement (%) = 100 x ({baseline} RMSE - {candidate} RMSE) / {baseline} RMSE. "
            f"Positive values indicate lower RMSE for {candidate}."
        )
    fig.text(0.5, 0.055, footer, ha="center", va="center", fontsize=10, color="#333333")

    output_path = get_output_path(config, plot_config)
    outpng = output_path / plot_config.get("output", _default_output_filename(config, plot_config, fhrs))
    fig.savefig(outpng, dpi=int(plot_config.get("dpi", config.get("dpi", 200))), bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", outpng)


def plot_one(config, plot_config):
    fhrs, xlabels = _forecast_hours(plot_config)
    rows, group_breaks = iter_variable_rows(config, plot_config.get("skip_levels", [100]))
    region_mats = {}
    for region in plot_config.get("regions", _default_regions()):
        region = region_info(config, region)
        logger.info("Building heatmap region panel: %s", region["key"])
        region_mats[region["key"]] = _build_region_matrix(config, plot_config, region, rows, fhrs)
    _plot_heatmap(config, plot_config, region_mats, rows, group_breaks, fhrs, xlabels)


def main(config):
    if isinstance(config, str):
        config = load_config(config)

    plots = config.get("plots") or [config]
    for plot_config in plots:
        merged = {**config, **plot_config}
        plot_one(config, merged)
