import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eagle.tools.performance_plot_utils import (
    get_output_path,
    iter_variable_rows,
    load_config,
    matched_response_values,
    model_color,
    model_label,
    region_label,
    region_info,
    slug,
    summarize,
    write_csv,
)

logger = logging.getLogger("eagle.tools")


def _forecast_hours(plot_config):
    lead_start, lead_end = plot_config.get("lead_hours", [24, 240])
    step = int(plot_config.get("lead_step", 24))
    return list(range(int(lead_start), int(lead_end) + 1, step))


def _default_output_stem(config, plot_config):
    regions = "-".join(slug(region_info(config, region)["key"]) for region in plot_config["regions"])
    models = "-".join(slug(model) for model in plot_config["models"])
    metric = slug(plot_config.get("metric", config.get("metric", "rmse")))
    lead_start, lead_end = plot_config.get("lead_hours", [24, 240])
    return f"violin_regions-{regions}_models-{models}_{metric}_f{int(lead_start):03d}-f{int(lead_end):03d}"


def _normalized_response_values(config, plot_config, model_key, region, fhrs):
    region = region_info(config, region)
    chunks = []
    metric = plot_config.get("metric", config.get("metric", "rmse"))
    models = plot_config["models"]
    equal_samples = bool(plot_config.get("equal_samples", True))

    rows, _ = iter_variable_rows(config, plot_config.get("skip_levels", []))
    for row in rows:
        levels = row.get("levels")
        level = None if levels is None else int(levels[0])
        raw_by_model = matched_response_values(
            config,
            model_keys=models,
            metric=metric,
            variable=row["var"],
            region=region,
            level=level,
            fhrs=fhrs,
        )
        raw_by_model = {
            selected_model: vals[np.isfinite(vals)]
            for selected_model, vals in raw_by_model.items()
            if vals[np.isfinite(vals)].size > 0
        }

        if any(selected_model not in raw_by_model for selected_model in models):
            continue

        if equal_samples:
            min_count = min(raw_by_model[selected_model].size for selected_model in models)
            if min_count < 2:
                continue
            raw_by_model = {
                selected_model: raw_by_model[selected_model][:min_count]
                for selected_model in models
            }

        pooled = np.concatenate([raw_by_model[selected_model] for selected_model in models])
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            continue

        denom = np.nanmedian(pooled)
        if not np.isfinite(denom) or denom <= 0:
            continue

        normalized = raw_by_model[model_key] / denom
        normalized = normalized[np.isfinite(normalized)]
        if normalized.size:
            chunks.append(normalized)

    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def _clipped_plot_data(data, good, percentile):
    all_vals = np.concatenate([data[idx] for idx in good])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return [], 1.0

    cap = float(np.nanpercentile(all_vals, percentile))
    cap = max(cap, 1.25)
    plot_data = []
    for idx in good:
        vals = data[idx]
        vals = vals[np.isfinite(vals)]
        clipped = vals[(vals >= 0.0) & (vals <= cap)]
        if clipped.size < 2:
            clipped = vals
        plot_data.append(clipped)
    return plot_data, cap


def plot_one(config, plot_config):
    metric = plot_config.get("metric", config.get("metric", "rmse"))
    regions = plot_config["regions"]
    models = plot_config["models"]
    fhrs = _forecast_hours(plot_config)
    output_path = get_output_path(config, plot_config)
    dpi = int(plot_config.get("dpi", config.get("dpi", 200)))
    output_stem = _default_output_stem(config, plot_config)
    outpng = output_path / plot_config.get("output", f"{output_stem}.png")
    if "summary_output" in plot_config:
        outcsv = output_path / plot_config["summary_output"]
    else:
        outcsv = output_path / f"{outpng.stem}_summary.csv"

    fig, axes = plt.subplots(
        len(regions),
        1,
        figsize=plot_config.get("figsize", [11.5, 4.9 * len(regions) + 1.2]),
        squeeze=False,
    )
    axes = axes[:, 0]
    rows = []

    for ax, raw_region in zip(axes, regions):
        region = region_info(config, raw_region)
        data = []
        labels = []
        colors = []
        stats = []

        for model_key in models:
            vals = _normalized_response_values(config, plot_config, model_key, region, fhrs)
            stat = summarize(vals)
            logger.info(
                "%s | %s | %s Count=%s Median=%.3f Mean=%.3f",
                plot_config["name"],
                region["key"],
                model_key,
                stat["count"],
                stat["median"],
                stat["mean"],
            )
            data.append(vals)
            labels.append(model_label(config, model_key, region["key"]))
            colors.append(model_color(config, model_key))
            stats.append(stat)
            rows.append(
                {
                    "plot": plot_config["name"],
                    "region": region["key"],
                    "model": model_key,
                    "metric": metric,
                    "value_type": "normalized_metric_all_responses",
                    **stat,
                }
            )

        positions = np.arange(1, len(models) + 1)
        good = [idx for idx, vals in enumerate(data) if vals.size > 1]
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.55, zorder=0)

        if good:
            plot_data, cap = _clipped_plot_data(data, good, float(plot_config.get("clip_percentile", 98.5)))
            vp = ax.violinplot(
                plot_data,
                positions=[positions[idx] for idx in good],
                widths=float(plot_config.get("violin_width", 0.62)),
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, idx in zip(vp["bodies"], good):
                body.set_facecolor(colors[idx])
                body.set_edgecolor("black")
                body.set_alpha(0.78)
                body.set_linewidth(1.2)

            ax.boxplot(
                plot_data,
                positions=[positions[idx] for idx in good],
                widths=0.15,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 1.1},
                capprops={"color": "black", "linewidth": 1.1},
            )
            ax.set_ylim(0, cap * float(plot_config.get("ylim_scale", 1.75)))

            for idx in good:
                x = positions[idx]
                stat = stats[idx]
                ax.scatter(
                    [x],
                    [min(stat["mean"], cap)],
                    s=48,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=6,
                )
                ax.text(
                    x,
                    0.955,
                    f"Count = {stat['count']}\nMedian = {stat['median']:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
                )
                ax.text(
                    x,
                    0.825,
                    f"{stat['mean']:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
                )

        ax.set_title(region_label(config, region), fontsize=17, fontweight="bold", pad=6)
        ax.set_ylabel(plot_config.get("ylabel", "Normalized RMSE\nLower is Better"), fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
            tick.set_fontweight("bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.28)
        ax.set_xlim(0.45, len(models) + 0.55)

    fig.suptitle(plot_config["title"], fontsize=20, fontweight="bold", y=0.990)
    if plot_config.get("subtitle"):
        fig.text(0.5, 0.925, plot_config["subtitle"], ha="center", fontsize=12.5)
    fig.tight_layout(rect=plot_config.get("tight_layout_rect", [0.04, 0.03, 1.0, 0.890]))
    fig.savefig(outpng, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_csv(outcsv, rows)
    logger.info("Wrote %s", outpng)
    logger.info("Wrote %s", outcsv)


def main(config):
    if isinstance(config, str):
        config = load_config(config)

    plots = config.get("plots") or [config]
    for plot_config in plots:
        merged = {**config, **plot_config}
        plot_one(config, merged)
