import logging

import pandas as pd

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner

logger = logging.getLogger("eagle.tools")


def create_anemoi_config(
    init_date: pd.Timestamp,
    main_config: dict,
) -> dict:
    """
    Create the config that will be passed to anemoi-inference.
    As of right now the "extract_lam" functionality is really only used to save out a static lam file when wanted.
    This could be easily updated if we think we would ever be interested in running inference over just CONUS.

    Args:
        init_date (str): date of initialization.
        extract_lam (bool): logic to extract lam domain, or run whole nested domain.
        lead_time (int): desired lead time to save out. Default=LEAD_TIME.
        lam (bool): true/false indication if LAM. Default=LAM.
        checkpoint_path (str): path to checkpoint. Default=CHECKPOINT.
        input_data_path (str): path to input data when not using LAM (e.g. you trained on 1 source). Default=INPUT_DATA_PATH.
        lam_path (str): path to regional data when using LAM. Default=LAM_PATH.
        global_path (str: path to global data when using LAM. Default=GLOBAL_PATH.
        output_path (str): path for saving files. Default=OUTPUT_PATH.

    Returns:
        dict -- config for anemoi-inference.
    """
    date_str = init_date.strftime("%Y-%m-%dT%H")

    lead_time = main_config.get("lead_time")
    config = {
        "checkpoint": main_config["checkpoint_path"],
        "date": date_str,
        "lead_time": lead_time,
        "input": {"dataset": main_config["input_dataset_kwargs"]},
        "runner": main_config.get("runner", "default"),
    }
    if main_config.get("extract_lam", False):
        config["output"] = {
            "extract_lam": {
                "output": {
                    "netcdf": {
                        "path": f"{main_config['output_path']}/{date_str}.{lead_time}h.lam.nc",
                    },
                },
            },
        }
    else:
        config["output"] = {
            "netcdf": f"{main_config['output_path']}/{date_str}.{lead_time}h.nc",
        }

    return config


def run_forecast(
    init_date: pd.Timestamp,
    main_config: dict,
) -> None:
    """
    Inference pipeline.

    Args:
        init_date (str): date of initialization.

    Returns:
        None -- files saved out to output path.
    """
    anemoi_config = create_anemoi_config(
        init_date=init_date,
        main_config=main_config,
    )
    run_config = RunConfiguration.load(anemoi_config)
    runner = create_runner(run_config)
    runner.execute()
    return


def main(config):
    """Runs Anemoi inference pipeline over many initialization dates.

    See ``eagle-tools inference --help`` or cli.py for help
    """

    dates = pd.date_range(start=config["start_date"], end=config["end_date"], freq=config["freq"])

    logger.info(f"Running Inference")
    logger.info(f"Initial Conditions:\n{dates}")
    for d in dates:
        logger.info(f"Processing {d}")
        run_forecast(
            init_date=d,
            main_config=config,
        )
        logger.info(f"Done with {d}")
    logger.info(f"Done Running Inference")
