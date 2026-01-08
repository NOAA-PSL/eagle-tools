import os
import yaml
import logging
import importlib.resources

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cmocean

import xmovie

from eagle.tools.data import open_anemoi_dataset, open_anemoi_inference_dataset
from eagle.tools.nested import get_nested_plot_box

logger = logging.getLogger("eagle.tools")

def get_extend(xds, vmin=None, vmax=None):
    minval = []
    maxval = []
    for key in xds.data_vars:
        minval.append(xds[key].min().values)
        maxval.append(xds[key].max().values)
    minval = np.min(minval)
    maxval = np.max(maxval)
    vmin = minval if vmin is None else vmin
    vmax = maxval if vmax is None else vmax

    extend = "neither"
    if minval < vmin:
        extend = "min"
    if maxval > vmax:
        extend = "max" if extend == "neither" else "both"
    return extend, vmin, vmax


def get_precip_kwargs():
    n = 1
    levels = np.concatenate(
        [
            np.linspace(0, .1, 2*n),
            np.linspace(.1, 1, 5*n),
            np.linspace(1, 10, 5*n),
            np.linspace(10, 50, 3*n),
        ],
    )
    return {
        "norm": BoundaryNorm(levels, len(levels)+1),
        "cmap": plt.get_cmap("cmo.rain", len(levels)+1),
        "cbar_kwargs": {"ticks": [0, 1, 10, 50]},
    }


def plot_nested_box(ax, box):
    kw = {
        "color": "gray",
        "transform": ccrs.PlateCarree(),
        "lw": 1,
        "alpha": .8,
    }
    for idx in [0, -1]:
        ax.plot(box["longitude"][box["y"], idx], box["latitude"][box["y"], idx], **kw)
        ax.plot(box["longitude"][idx, box["x"]], box["latitude"][idx, box["x"]], **kw)


def nested_scatter(ax, xds, varname, lam_index, box, lam_size=0.25, global_size=12, **kwargs):
    mappables = []
    for slc, s in zip(
        [slice(None, lam_index), slice(lam_index, None)],
        [lam_size, global_size],
    ):
        p = ax.scatter(
            xds["longitude"].isel(cell=slc),
            xds["latitude"].isel(cell=slc),
            c=xds[varname].isel(cell=slc),
            s=s,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
        mappables.append(p)

    plot_nested_box(ax, box)
    return mappables[0]


def plot_single_timestamp(xds, fig, time, *args, **kwargs):

    axs = []
    vtime = xds["time"].isel(time=time).values
    stime = pd.Timestamp(vtime).strftime("%Y-%m-%dT%H")

    # get these extra options
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    extend = kwargs.pop("extend", None)
    st0 = kwargs.pop("st0", "")
    model_type = kwargs.pop("model_type", "")
    lam_index = kwargs.pop("lam_index", None)
    box = kwargs.pop("box", None)
    lam_size = kwargs.pop("lam_size", None)
    global_size = kwargs.pop("global_size", None)

    subplot_kw = {}
    projection = kwargs.pop("projection", None)
    if projection is not None:
        Projection = getattr(ccrs, projection)
        subplot_kw = {
            "projection": Projection(**kwargs.pop("projection_kwargs", {})),
        }

    for ii, label in enumerate(["target", "prediction"]):
        ax = fig.add_subplot(1, 2, ii+1, **subplot_kw)

        if model_type == "nested":
            p = nested_scatter(ax, xds.isel(time=time), label, lam_index, box, lam_size=lam_size, global_size=global_size, **kwargs)

        else:
            p = ax.pcolormesh(
                xds.longitude,
                xds.latitude,
                xds[label].isel(time=time).values,
                transform=ccrs.PlateCarree(),
                **kwargs,
            )
        ax.set(title=xds[label].nice_name)
        axs.append(ax)

    # now the colorbar
    [ax.set(xlabel="", ylabel="") for ax in axs]
    [ax.coastlines("50m") for ax in axs]

    label = xds.attrs.get("label", "")
    label += f"\nt0: {st0}"
    label += f"\nvalid: {stime}"
    fig.colorbar(
        p,
        ax=axs,
        orientation="horizontal",
        shrink=.8,
        pad=0.05,
        aspect=35,
        label=label,
        extend=extend,
        **cbar_kwargs,
    )
    fig.set_constrained_layout(True)
    return None, None


def create_media(
    xds: xr.Dataset,
    mode: str,
    fname: str,
    dpi: int,
    width: int,
    height: int,
    options: dict,
) -> None:

    pixelwidth = width*dpi
    pixelheight = height*dpi

    if mode == "figure":

        path = fname + ".jpeg"
        fig = plt.figure(figsize=(width, height))
        plot_single_timestamp(
            xds=xds,
            fig=fig,
            time=0,
            **options,
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Stored figure at: {path}")

    else:
        path = fname + ".gif"
        mov = xmovie.Movie(
            xds,
            plot_single_timestamp,
            framedim="time",
            input_check=False,
            pixelwidth=pixelwidth,
            pixelheight=pixelheight,
            dpi=dpi,
            **options,
        )
        mov.save(
            path,
            progress=True,
            overwrite_existing=True,
            remove_frames=True,
            framerate=10,
            gif_framerate=10,
            remove_movie=False,
            gif_palette=True,
            gif_scale=["trunc(iw/2)", "trunc(ih/2)"],
        )
        logger.info(f"Stored movie at: {path}")
    return


def main(config, mode):
    """Create figures or movies visually comparing predictions to targets

    See ``eagle-tools visualize --help`` or cli.py for help
    """
    topo = config["topo"]
    if config["use_mpi"]:
        raise NotImplementedError

    # options used for verification and inference datasets
    model_type = config.get("model_type")
    lam_index = config.get("lam_index", None)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
        "lcc_info": config.get("lcc_info", None),
        "member": config.get("member", None),
    }
    output_dir = config["output_path"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    t0 = pd.Timestamp(config["start_date"])
    tf = pd.Timestamp(config["end_date"])
    st0 = t0.strftime("%Y-%m-%dT%H")
    stf = tf.strftime("%Y-%m-%dT%H")
    logger.info(f"Time Bounds:\n\tt0 = {st0}\n\ttf = {stf}")

    # Target dataset
    tds = open_anemoi_dataset(
        model_type=model_type,
        t0=str(t0),
        tf=str(tf),
        rename_to_longnames=True,
        reshape_cell_to_2d=model_type != "nested",
        **subsample_kwargs,
        **config["verification_dataset_kwargs"],
    )
    tds = tds.squeeze("member")
    if mode == "figure":
        tds = tds.sel(time=[tf])
    else:
        tds = tds.sel(time=slice(t0, tf))
    logger.info(f"Opened Target dataset:\n{tds}")

    # Prediction dataset
    pds = open_anemoi_inference_dataset(
        f"{config['forecast_path']}/{st0}.{config['lead_time']}h.nc",
        model_type=model_type,
        lam_index=lam_index,
        trim_edge=config.get("trim_forecast_edge", None),
        rename_to_longnames=True,
        reshape_cell_to_2d=True,
        **subsample_kwargs,
    )
    if mode == "figure":
        pds = pds.sel(time=[tf])
    else:
        pds = pds.sel(time=slice(t0, tf))
    logger.info(f"Opened Prediction dataset:\n{pds}")

    # setup plot options with user overrides
    defaults_path = importlib.resources.files("eagle.tools.config") / "defaults.yaml"
    with defaults_path.open("r") as f:
        defaults = yaml.safe_load(f)
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rename = yaml.safe_load(f)

    fig_kwargs = defaults["fig_kwargs"].copy()
    fig_kwargs.update(config.get("fig_kwargs", {}))
    per_variable_kwargs = defaults["per_variable_kwargs"].copy()
    per_variable_kwargs["total_precipitation_6hr"] = get_precip_kwargs()
    pvk_user = config.get("per_variable_kwargs", {})
    for key, val in pvk_user.items():
        k = rename.get(key, key)
        per_variable_kwargs[k].update(val)
    units = defaults["units"].copy()
    units.update(config.get("units", {}))

    # filter: get kwargs for desired variables only
    per_variable_kwargs = {
        key: per_variable_kwargs.get(key, {})
        for key, val in per_variable_kwargs.items()
        if key in pds.data_vars
    }

    # for nested, create the little box around the LAM region
    if model_type == "nested":
        assert lam_index is not None
        assert subsample_kwargs["lcc_info"] is not None
        box = get_nested_plot_box(tds, lam_index, subsample_kwargs["lcc_info"])

    # Plot each variable individually
    for varname, options in per_variable_kwargs.items():

        # xmovie requires a single dataset, so package up predictions + target for each variable
        ds = xr.Dataset({
            "prediction": pds[varname].reset_coords(drop=True).load(),
            "target": tds[varname].load(),
        })
        ds["prediction"].attrs["nice_name"] = "Prediction: " + config.get("model_name", "")
        ds["target"].attrs["nice_name"] = "Target: " + config.get("target_name", "")

        # Convert to degC
        if "temperature" in varname:
            for key in ds.data_vars:
                ds[key] -= 273.15
                ds[key].attrs["units"] = "degC"

            logger.info(f"\tconverted {varname} K -> degC")

        label = " ".join([x.capitalize() for x in varname.split("_")])
        ds.attrs["label"] = f"{label} ({units.get(varname, '')})"

        # colorbar extension options
        options["extend"], vmin, vmax = get_extend(
            ds,
            vmin=options.get("vmin", None),
            vmax=options.get("vmax", None),
        )

        # precip is weird, since we don't do vmin/vmax, we do BoundaryNorm colorbar map blah blah
        # since we know it's bounded to be positive in anemoi... at least in this model..
        # then just worry about max
        if "total_precipitation" in varname:
            options["extend"] = "max" if vmax > 50 else "neither"
            logger.info(f"\ttotal_precipitation hack: setting extend based on upper limit of 50")

        options["st0"] = st0
        options["projection"] = fig_kwargs["projection"]
        options["projection_kwargs"] = fig_kwargs.get("projection_kwargs", {})
        options["model_type"] = model_type
        if model_type == "nested":
            options["lam_index"] = lam_index
            options["box"] = box
            options["lam_size"] = fig_kwargs["lam_size"]
            options["global_size"] = fig_kwargs["global_size"]

        logger.info(f"Plotting {varname} with options")
        for key, val in options.items():
            if key == "box" and box is not None:
                logger.info(f"\t{key}: {val.keys()}")

            else:
                logger.info(f"\t{key}: {val}")

        if "level" in ds.dims:
            for level in ds["level"].values:
                fname = f"{output_dir}/{varname}.level{level}.{st0}.{stf}"
                create_media(
                    xds=ds.sel(level=level),
                    mode=mode,
                    fname=fname,
                    dpi=fig_kwargs["dpi"],
                    width=fig_kwargs["width"],
                    height=fig_kwargs["height"],
                    options=options,
                )

        else:
            fname = f"{output_dir}/{varname}.{st0}.{stf}"
            create_media(
                xds=ds,
                mode=mode,
                fname=fname,
                dpi=fig_kwargs["dpi"],
                width=fig_kwargs["width"],
                height=fig_kwargs["height"],
                options=options,
            )
