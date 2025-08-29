"""TODO List for this

* separate comparisons to ERA5/Replay and GFS/HRRR... this is getting nasty
* pass the model name as argument to main
* variable list should be an input argument
* eventually we'll want to pass the dot size as an argument too, TBD
"""
import os
import sys
import logging

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cmocean

import xmovie

from anemoi.datasets import open_dataset as open_anemoi_dataset
from ufs2arco.utils import expand_anemoi_dataset
from ufs2arco.log import setup_simple_log

logger = logging.getLogger("eagle.tools")

_projection = ccrs.Orthographic(
    central_longitude = -120,
    central_latitude = 20,
)

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
            #np.linspace(50, 80, 2),
        ],
    )
    norm = BoundaryNorm(levels, len(levels)+1)
    cmap = plt.get_cmap("cmo.rain", len(levels)+1)
    return {"norm": norm, "cmap": cmap, "cbar_kwargs": {"ticks": [0, 1, 10, 50]}}


def plot_single_timestamp(xds, fig, time, *args, **kwargs):

    axs = []

    vtime = xds["time"].isel(time=time).values
    stime = str(vtime)[:13]

    # get these extra options
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    extend = kwargs.pop("extend", None)
    t0 = kwargs.pop("t0", "")
    hds = kwargs.pop("hds", None)

    lon = np.unique(xds["longitudes"])
    lat = np.unique(xds["latitudes"])[::-1]
    shape = (len(lat), len(lon))

    for ii, label in enumerate(["truth", "prediction"]):
        ax = fig.add_subplot(1, 2, ii+1, projection=_projection)
#        p = ax.scatter(
        p = ax.pcolormesh(
            lon,
            lat,
            xds[label].isel(time=time).values.reshape(shape),
            #s=1/4,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
        ax.set(title=xds[label].nice_name)
        axs.append(ax)

    # now the colorbar
    [ax.set(xlabel="", ylabel="") for ax in axs]
    [ax.coastlines("50m") for ax in axs]

    label = xds.attrs.get("label", "")
    label += f"\nt0: {t0}"
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

def calc_wind_speed(xds):
    if "ugrd10m" in xds:
        ws = np.sqrt(xds["ugrd10m"]**2 + xds["vgrd10m"]**2)
    elif "u10" in xds:
        ws = np.sqrt(xds["u10"]**2 + xds["v10"]**2)
    else:
        ws = np.sqrt(xds["10m_u_component_of_wind"]**2 + xds["10m_v_component_of_wind"]**2)
    ws.attrs["units"] = "m/s"
    ws.attrs["long_name"] = "10m Wind Speed"
    xds["10m_wind_speed"] = ws

    if "80m_u_component_of_wind" in xds:
        ws80 = np.sqrt(xds["80m_u_component_of_wind"]**2 + xds["80m_v_component_of_wind"]**2)
        xds["80m_wind_speed"] = ws80
        xds["80m_wind_speed"].attrs["units"] = "m/s"
    return xds

def get_reanalysis(name):
    if name.lower() == "era5":
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        rename = {}
    elif name.lower() == "replay":
        url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"
        rename = {"pfull": "level", "grid_yt": "latitude", "grid_xt": "longitude"}

    truth = xr.open_zarr(
        url,
        storage_options={"token": "anon"},
    )
    truth = truth.rename(rename)
    truth.attrs["name"] = name
    truth["x"], truth["y"] = np.meshgrid(truth["longitude"], truth["latitude"])
    return truth

#def get_truth(name, t0, tf):
#    if name in ["era5", "replay"]:
#        truth = get_reanalysis(name)
#    else:
#        data_dir = "/pscratch/sd/t/timothys/anemoi-house/gfs/0.25-degree/data"
#        ads = open_anemoi_dataset(f"{data_dir}/gfs.zarr")
#
#        start = ads.to_index(date=t0, variable=0)[0]
#        end = ads.to_index(date=tf, variable=0)[0] + 1
#        truth = xr.DataArray(
#            ads[start:end,:,:,:],
#            coords={
#                "time": np.arange(end-start),
#                "variable": np.arange(ads.shape[1]),
#                "ensemble": np.arange(ads.shape[2]),
#                "cell": np.arange(ads.shape[3]),
#            },
#            dims=("time", "variable", "ensemble", "cell"),
#        ).load().squeeze()
#        truth = truth.to_dataset(name="data")
#        truth["latitudes"] = xr.DataArray(ads.latitudes, coords=truth.cell.coords)
#        truth["longitudes"] = xr.DataArray(ads.longitudes, coords=truth.cell.coords)
#        truth["dates"] = xr.DataArray(
#            ads.dates[start:end],
#            dims="time",
#        )
#        truth = expand_anemoi_dataset(truth, "data", ads.variables)
#
#        for varname, unit in zip(
#            ["t2m", "sh2", "accum_tp", "u10", "v10", "u", "v", "w", "u80", "v80"],
#            ["K", "kg/kg", "mm", "m/s", "m/s", "m/s", "m/s", "m/s", "m/s", "m/s"],
#        ):
#            truth[varname].attrs["units"] = unit
#        truth.attrs["n_conus"] = ads.grids[0]
#
#    truth = rename_short_to_long(truth)
#    truth = calc_wind_speed(truth)
#    return truth


def rename_short_to_long(xds):

    rename = {
        "accum_tp": "total_precipitation_6hr",
        "t2m": "2m_temperature",
        "sh2": "2m_specific_humidity",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "u80": "80m_u_component_of_wind",
        "v80": "80m_v_component_of_wind",
    }
    return xds.rename({key: val for key, val in rename.items() if key in xds})

def main(config)
    """A note about t0
    In the inference yaml, I think this means "the very first initial condition"... makes sense

    But when I'm visualizing the data, I think about t0 as the last initial condition...
    as in the last data given to the model before making a forecast.
    So... the t0 given here is that one... the last IC.
    """

    setup_simple_log()



    assert mode in ["figure", "movie"]

    plot_options = {
    #    "total_precipitation_6hr": get_precip_kwargs(),
    #    "80m_wind_speed": {
    #        "cmap": "cmo.tempo_r",
    #        "vmin": 0,
    #        "vmax": 20,
    #    },
        "2m_temperature": {
            "cmap": "cmo.thermal",
            "vmin": -10,
            "vmax": 30,
        },
        "10m_wind_speed": {
            "cmap": "cmo.tempo_r",
            "vmin": 0,
            "vmax": 20,
        },
        "2m_specific_humidity": {
            "cmap": "cmo.rain",
            "vmin": 0,
            "vmax": 0.025,
        },

    }

    # options used for verification and inference datasets
    model_type = config.get("model_type")
    lam_index = config.get("lam_index", None)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
    }

    # Verification dataset
    vds = open_anemoi_dataset(
        path=config["verification_dataset_path"],
        trim_edge=config.get("trim_edge", None),
        **subsample_kwargs,
    )

    # Forecast dataset
    t0 = pd.Timestamp(config["start_date"])
    tf = pd.Timestamp(config["end_date"])
    st0 = t0.strftime("%Y-%m-%dT%H")
    fds = open_anemoi_inference_dataset(
        f"{config['forecast_path']}/{st0}.{config['lead_time']}.nc",
        model_type=model_type,
        lam_index=lam_index,
    )
    fds.attrs["nice_name"] = "Prediction: " + config.get("model_name", "")

    logging.info(f"Time Bounds:\n\tt0 = {t0}\n\ttf = {tf}\n")
    psl = rename_short_to_long(psl)
    psl = calc_wind_speed(psl)

    logging.info(f"Ready to make {mode}s with dataset:\n{psl}\n")

    for tname in ["GFS"]: #["ERA5"]:

        truth = get_truth(tname, t0=t0, tf=tf)

        logging.info(f"Retrieved truth = {tname}\n{truth}\n")
        fig_dir = os.path.join(store_dir, f"{mode}s", f"{tname.lower().replace('/','-')}-vs-eagle")
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
            logging.info(f"Created fig_dir: {fig_dir}")

        for varname, options in plot_options.items():

            logging.info(f"Plotting {varname} with options")
            for key, val in options.items():
                logging.info(f"\t{key}: {val}")

            ds = xr.Dataset({
                "prediction": psl[varname].load(),
                "truth": truth[varname].sel(
                    time=psl.time.values,
                ).load(),
            })
            ds["prediction"].attrs["nice_name"] = psl.nice_name
            ds["truth"].attrs["nice_name"] = tname
            ds["truth"].attrs["n_conus"] = truth.attrs["n_conus"]

            # Convert to degC
            if varname[:3] == "tmp" or "temperature" in varname:
                for key in ds.data_vars:
                    ds[key] -= 273.15
                    ds[key].attrs["units"] = "degC"

                logging.info(f"\tconverted {varname} K -> degC")

            # Convert to mm->m
            if "total_precipitation" in varname and tname == "ERA5":
                ds["truth"].attrs["units"] = "mm"

            label = " ".join([x.capitalize() for x in varname.split("_")])
            ds.attrs["label"] = f"{label} ({ds.truth.units})"

            # colorbar extension options
            options["extend"], vmin, vmax = get_extend(
                ds,
                vmin=options.get("vmin", None),
                vmax=options.get("vmax", None),
            )
            logging.info(f"\tcolorbar extend = {options['extend']}")

            # precip is weird, since we don't do vmin/vmax, we do BoundaryNorm colorbar map blah blah
            # since we know it's bounded to be positive in anemoi... at least in this model..
            # then just worry about max
            if "total_precipitation" in varname:
                options["extend"] = "max" if vmax > 50 else "neither"
                logging.info(f"\ttotal_precipitation hack: setting extend based on upper limit of 50")

            options["t0"] = t0

            dpi = 300
            width = 10
            height = 7
            pixelwidth = width*dpi
            pixelheight = height*dpi

            if mode == "figure":

                fig = plt.figure(figsize=(width, height))
                itime = list(pd.Timestamp(x) for x in ds["time"].values).index(pd.Timestamp(tf))
                plot_single_timestamp(
                    xds=ds,
                    fig=fig,
                    time=itime,
                    **options,
                )
                fname = f"{fig_dir}/{varname}.{t0}.{tf}.jpeg"
                fig.savefig(fname, dpi=dpi, bbox_inches="tight")
                logging.info(f"Stored figure at: {fname}\n")

            else:
                mov = xmovie.Movie(
                    ds,
                    plot_single_timestamp,
                    framedim="time",
                    input_check=False,
                    pixelwidth=pixelwidth,
                    pixelheight=pixelheight,
                    dpi=dpi,
                    **options
                )
                fname = f"{fig_dir}/{varname}.{t0}.{tf}.gif"
                mov.save(
                    fname,
                    progress=True,
                    overwrite_existing=True,
                    remove_frames=True,
                    framerate=10,
                    gif_framerate=10,
                    remove_movie=False,
                    gif_palette=True,
                    gif_scale=["trunc(iw/2)", "trunc(ih/2)"],
                )
                logging.info(f"Stored movie at: {fname}\n")
