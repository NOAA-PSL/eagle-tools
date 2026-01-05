import os
import logging
import yaml
import warnings

from ufs2arco.mpi import MPITopology, SerialTopology

logger = logging.getLogger("eagle.tools")

def setup(config_filename: str, command: str):
    config = open_yaml_config(config_filename)
    topo, use_mpi = init_topo(config, command)
    config["topo"] = topo
    config["use_mpi"] = use_mpi

    # if output_path is not created, make it here
    if topo.is_root:
        if "output_path" in config and not os.path.isdir(config["output_path"]):
            logger.info(f"Creating output_path: {config['output_path']}")
            os.makedirs(config["output_path"])
    topo.barrier()

    return config

def open_yaml_config(config_filename: str):
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    # expand any environment variables
    for key, val in config.items():
        if "path" in key:
            if isinstance(val, str):
                config[key] = os.path.expandvars(val)
            else:
                logger.warning(f"Not expanding environment variables in {key} in config, since it could be many different types")
    return config

def init_topo(config: dict, command: str) -> MPITopology | SerialTopology:

    use_mpi = are_we_using_mpi(config)
    if use_mpi:
        topo = MPITopology(log_dir=config.get("log_path", f"eagle-logs/{command}"))

    else:
        topo = SerialTopology()

    logger.setLevel(logging.INFO)
    logger.addHandler(topo.log_handler)

    ufs2arco_logger = logging.getLogger("ufs2arco")
    ufs2arco_logger.setLevel(logging.INFO)
    ufs2arco_logger.addHandler(topo.log_handler)

    return topo, use_mpi


def are_we_using_mpi(config: dict) -> bool:

    use_mpi = config.get("use_mpi", False)
    srun_used = _srun_used()
    mpirun_used = _mpirun_used()

    if not use_mpi and (srun_used or mpirun_used):
        assert not (srun_used and mpirun_used), "We shouldn't see this, please raise an issue"
        whichone = "srun" if srun_used else "mpirun"
        msg = f"Could not find 'use_mpi = True' in config, but found that {whichone}_used = True. Setting config['use_mpi'] = True."
        warnings.warn(msg)
        use_mpi = True

    return use_mpi


def _srun_used() -> bool:
    """Note that pytorch lightning has a similar function, but it ignores when srun is used in an interactive session.
    This works when srun is used in an interactive session
    """
    if "SLURM_NTASKS" in os.environ:
        try:
            ppid = os.getppid()
            with open(f'/proc/{ppid}/comm') as f:
                parent = f.read().strip()
            return parent in ('srun', 'slurmstepd')
        except:
            return False
def _mpirun_used() -> bool:
    """Detect if the process was launched via mpirun or mpiexec.

    Checks for common MPI environment variables and walks up the process tree
    looking for known MPI launcher processes.
    """
    # Common MPI environment variables set by various implementations
    mpi_env_vars = (
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",              # MPICH/Intel MPI
        "MV2_COMM_WORLD_SIZE",   # MVAPICH2
    )

    if not any(var in os.environ for var in mpi_env_vars):
        return False

    # Known MPI launcher process names
    mpi_launchers = {'mpirun', 'mpiexec', 'mpiexec.hydra', 'orterun', 'orted', 'hydra_pmi_proxy', 'pmi_proxy'}

    try:
        pid = os.getppid()
        for _ in range(10):
            if pid <= 1:
                break
            if sys.platform == 'linux':
                with open(f'/proc/{pid}/comm') as f:
                    parent = f.read().strip()
                with open(f'/proc/{pid}/stat') as f:
                    stat = f.read().split()
                    ppid = int(stat[3])
            else:
                # macOS/BSD: use ps command
                result = subprocess.run(
                    ['ps', '-o', 'comm=,ppid=', '-p', str(pid)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    break
                parts = result.stdout.strip().rsplit(None, 1)
                if len(parts) != 2:
                    break
                parent, ppid = parts[0], int(parts[1])

            if os.path.basename(parent) in mpi_launchers:
                return True
            pid = ppid
        return False
    except:
        return False
