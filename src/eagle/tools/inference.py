import logging

import numpy as np
import pandas as pd

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner

logger = logging.getLogger("eagle.tools")


def create_anemoi_config(
    init_date: pd.Timestamp,
    main_config: dict,
    member: int | None = None,
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

    fname = f"{main_config['output_path']}/{date_str}.{lead_time}h.nc"
    if member is not None:
        fname = fname.replace(".nc", f".member{member:03d}.nc")

    if main_config.get("extract_lam", False):
        fname = fname.replace(".nc", "lam.nc")
        config["output"] = {
            "extract_lam": {
                "output": {
                    "netcdf": {
                        "path": fname,
                    },
                },
            },
        }
    else:
        config["output"] = {
            "netcdf": fname,
        }

    return config


def _load_model_once(main_config: dict):
    """Load the ML model once by creating a temporary runner.

    This avoids reloading the model from disk for every initialization date.
    The returned model can be injected into subsequent runners via ``preloaded_model``.

    Args:
        main_config (dict): The main configuration dictionary.

    Returns:
        torch.nn.Module: The loaded model.
    """
    dummy_date = pd.Timestamp(main_config["start_date"])
    anemoi_config = create_anemoi_config(init_date=dummy_date, main_config=main_config)
    run_config = RunConfiguration.load(anemoi_config)
    runner = create_runner(run_config)
    model = runner.model
    return model


def _seed_rank(topo, config):
    """Set a deterministic, per-rank torch RNG seed for ensemble diversity.

    Follows the anemoi convention: reads ANEMOI_BASE_SEED (or SLURM_JOB_ID)
    from the environment, then offsets by rank so each MPI process generates
    distinct noise in SimpleNoiseConditioning.

    If no seed source is found, logs a warning and skips seeding (falls back
    to PyTorch's default random seed, which is still per-process unique but
    not reproducible).
    """
    import os

    import torch

    base_seed = None
    for env_var in ("ANEMOI_BASE_SEED", "SLURM_JOB_ID"):
        if env_var in os.environ:
            base_seed = int(os.environ[env_var])
            break

    if config.get("base_seed") is not None:
        base_seed = config["base_seed"]

    if base_seed is None:
        logger.warning(
            "No base seed found (set ANEMOI_BASE_SEED, SLURM_JOB_ID, or config 'base_seed'). "
            "Ensemble members will differ across ranks but results will not be reproducible."
        )
        return

    if base_seed < 1000:
        base_seed *= 1000

    rank_seed = base_seed * (topo.rank + 1)
    torch.manual_seed(rank_seed)
    logger.info(f"Seeded rank {topo.rank} with torch seed {rank_seed} (base_seed={base_seed})")


def run_forecast(
    init_date: pd.Timestamp,
    main_config: dict,
    preloaded_model=None,
    member: int | None = None,
) -> None:
    """
    Inference pipeline.

    Args:
        init_date (str): date of initialization.
        preloaded_model (torch.nn.Module, optional): A previously loaded model to
            inject into the runner, avoiding repeated ``torch.load()`` calls.

    Returns:
        None -- files saved out to output path.
    """
    anemoi_config = create_anemoi_config(
        init_date=init_date,
        main_config=main_config,
        member=member,
    )
    run_config = RunConfiguration.load(anemoi_config)
    runner = create_runner(run_config)
    if preloaded_model is not None:
        runner.__dict__["model"] = preloaded_model
    runner.execute()
    return


def main(config):
    """Runs Anemoi inference pipeline over many initialization dates.

    See ``eagle-tools inference --help`` or cli.py for help
    """

    topo = config["topo"]

    if config["use_mpi"] and config.get("runner", "default") == "parallel":
        raise ValueError(
            "Cannot combine use_mpi=True with runner='parallel'. "
            "MPI date-parallelism and the ParallelRunnerMixin both try to manage "
            "SLURM process groups, which conflict with each other. "
            "Use runner='default' when distributing dates across MPI ranks."
        )

    dates = pd.date_range(start=config["start_date"], end=config["end_date"], freq=config["freq"])
    n_members = config.get("n_members", 1)

    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    logger.info(f"Running Inference")
    logger.info(f"Initial Conditions:\n{dates}")

    logger.info("Loading model")
    model = _load_model_once(config)
    logger.info("Model loaded")

    _seed_rank(topo, config)

    for batch_idx in range(n_batches):
        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx >= n_dates:
            break

        d = dates[date_idx]
        logger.info(f"Processing {d} for {n_members} members")
        for member in range(n_members):
            run_forecast(
                init_date=d,
                main_config=config,
                preloaded_model=model,
                member=member if n_members>1 else None,
            )

        logger.info(f"Done with {d}")

    topo.barrier()
    logger.info(f"Done Running Inference")
