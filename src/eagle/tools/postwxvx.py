import logging
import io
import importlib
import yaml

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("eagle.tools")

def parse_lead_time(lead_str: str) -> int | float:
    """
    Parses MET lead time strings (e.g., "6", "12", "0600", "060000")
    into an int with forecast hours.
    """
    s_len = len(lead_str)
    if s_len <= 3:  # Handles H, HH, HHH (e.g., "6", "12", "120")
        return int(lead_str)
    elif s_len == 4:  # Handles HHMM (e.g., "0600")
        hours = int(lead_str[0:2])
        minutes = int(lead_str[2:4])
        if minutes > 0:
            return hours + minutes / 60
        else:
            return hours
    else:  # Assumes HHMMSS (e.g., "060000", "120000")
        hours = int(lead_str[0:-4])
        minutes = int(lead_str[-4:-2])
        seconds = int(lead_str[-2:])
        if minutes + seconds > 0:
            return hours + minutes / 60 + seconds / 3600
        else:
            return hours


def met_txtfile_to_dict(filename):
    """Read a .txt file from met, convert to dict"""

    with open(filename, 'r') as f:
        met_output = f.read()

    # Use io.StringIO to treat the file's content as a virtual file for consistent parsing
    file_stream = io.StringIO(met_output)

    # Read all lines and find the split between header and data
    all_lines = [line.strip() for line in file_stream if line.strip()]
    split_index = -1
    for i, line in enumerate(all_lines):
        if line.startswith('V11'): # Assumes data line starts with MET version
            split_index = i
            break

    # Concatenate and split header and data lines
    header_str = " ".join(all_lines[:split_index])
    data_str = " ".join(all_lines[split_index:])

    headers = header_str.split()
    values = data_str.split()

    # Create a dictionary of all parsed data
    parsed_data = dict(zip(headers, values))
    return parsed_data


def met_dict_to_dataset(mdict):

    attributes = {}
    data_vars = {}

    metmetapath = importlib.resources.files("eagle.tools.config") / "met.yaml"
    with metmetapath.open("r") as f:
        metmeta = yaml.safe_load(f)

    # The coordinate for our dataset will be the forecast lead time
    # Convert HHMMSS to a pandas Timedelta for better usability
    lead_time = parse_lead_time(mdict["FCST_LEAD"])

    # Loop through the parsed data to separate attributes from variables
    for key, value in mdict.items():
        # Attempt to convert to numeric, otherwise keep as string
        try:
            if value == 'NA':
                num_value = np.nan
            else:
                num_value = pd.to_numeric(value)
        except ValueError:
            num_value = value

        if key in metmeta["metadata_keys"]:
            attributes[key] = num_value
        else:
            # Create a DataArray for each metric
            data_vars[key] = xr.DataArray(
                data=[num_value],  # Data must be in a list or array
                dims=['fhr'],
                coords={'fhr': [lead_time]},
                name=key,
                attrs={'long_name': metmeta["long_names"].get(key, 'Unknown')}
            )

    return xr.Dataset(data_vars, attrs=attributes)


def main(config):

    topo = config["topo"]
    if config["use_mpi"]:
        raise NotImplementedError

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    lead_times = np.arange(config["leadtimes"]["start"], config["leadtimes"]["end"]+1, config["leadtimes"]["step"])
    stat_prefix = config["stat_prefix"]
    variable_prefixes = config["variable_prefixes"]
    work_path = config["work_path"]
    metmetapath = importlib.resources.files("eagle.tools.config") / "met.yaml"
    with metmetapath.open("r") as f:
        rename = yaml.safe_load(f)["rename"]

    logger.info(f"Gathering metrics from wxvx stats")
    logger.info(f"Initial Conditions:\n{dates}")
    logger.info(f"Variables:\n{variable_prefixes}")
    for varname in variable_prefixes:
        logger.info(f"Processing {varname}")
        dslist2 = []
        num_failures = 0
        for t0 in dates:
            st0 = t0.strftime("%Y-%m-%dT%H")
            logger.debug(f"Processing {st0}")

            dslist = []
            for fhr in lead_times:

                slead = f"{fhr:02d}0000"
                vtime = t0 + pd.Timedelta(hours=fhr)
                st01 = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
                st02 = f"{t0.hour:02d}{t0.minute:02d}{t0.second:02d}"
                svt1 = f"{vtime.year:04d}{vtime.month:02d}{vtime.day:02d}"
                svt2 = f"{vtime.hour:02d}{vtime.minute:02d}{vtime.second:02d}"

                filename = f"{work_path}/run/stats/{st01}/{st02[:2]}/{fhr:03d}/{stat_prefix}_{varname}_{slead}L_{svt1}_{svt2}V_cnt.txt"
                try:
                    mdict = met_txtfile_to_dict(filename)
                    xds = met_dict_to_dataset(mdict)
                except:
                    num_failures += 1
                    svtime = vtime.strftime("%Y-%m-%dT%H")
                    msg = f"Failure for {varname} at (t0, fhr) = ({st0}, {fhr}), valid time = {svtime}"
                    logger.warning(msg)
                    xds = xr.Dataset({"fhr": xr.DataArray([fhr], coords={"fhr": [fhr]})})
                dslist.append(xds)

            fds = xr.concat(dslist, dim="fhr")
            fds = fds.expand_dims({"t0": [t0]})
            dslist2.append(fds)
        result = xr.concat(dslist2, dim="t0")
        nicename = rename.get(varname, varname)
        result.to_netcdf(f"{work_path}/{nicename}.nc")

        if num_failures > 0:
            msg = f" ... {varname} had {num_failures} failures out of {len(lead_times)*len(dates)} timestamps"
            logger.info(msg)
    logger.info(f"Done with postwxvx workflow")
