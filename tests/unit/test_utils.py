import subprocess
from unittest import mock

import pytest
import yaml
from ufs2arco.mpi import SerialTopology

from eagle.tools import utils

MPI_ENV_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")


@pytest.fixture(autouse=True)
def no_launcher_env(monkeypatch):
    """Keep a real srun/mpirun session from leaking into these tests."""
    for var in MPI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def write_yaml(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


# --- open_yaml_config -------------------------------------------------------


def test_open_yaml_config_expands_env_vars_in_path_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE", "/scratch")
    path = write_yaml(
        tmp_path,
        {
            "output_path": "$BASE/out",
            "model_type": "$BASE",
            "regrid_kwargs": {"target_grid_path": "$BASE/grid.nc"},
        },
    )

    config = utils.open_yaml_config(path)

    assert config["output_path"] == "/scratch/out"
    assert config["regrid_kwargs"]["target_grid_path"] == "/scratch/grid.nc"
    assert config["model_type"] == "$BASE", "only keys containing 'path' are expanded"


def test_open_yaml_config_leaves_non_string_paths_alone(tmp_path, caplog):
    path = write_yaml(tmp_path, {"input_path": ["$HOME/a", "$HOME/b"]})

    config = utils.open_yaml_config(path)

    assert config["input_path"] == ["$HOME/a", "$HOME/b"]
    assert "Not expanding environment variables in input_path" in caplog.text


# --- setup / init_topo --------------------------------------------------------


def test_setup_creates_output_path_and_serial_topology(tmp_path):
    output_path = tmp_path / "new" / "output"
    path = write_yaml(tmp_path, {"output_path": str(output_path)})

    config = utils.setup(path, "metrics")

    assert output_path.is_dir()
    assert isinstance(config["topo"], SerialTopology)
    assert config["use_mpi"] is False


def test_init_topo_uses_mpi_when_requested():
    with mock.patch.object(utils, "MPITopology") as mpi_topology:
        topo, use_mpi = utils.init_topo({"use_mpi": True}, "metrics")

    assert use_mpi is True
    assert topo is mpi_topology.return_value
    mpi_topology.assert_called_once_with(log_dir="eagle-logs/metrics")


# --- are_we_using_mpi ---------------------------------------------------------


@pytest.mark.parametrize(
    ("use_mpi", "srun", "mpirun", "expected"),
    [
        (False, False, False, False),
        (True, False, False, True),
        (True, True, False, True),
    ],
)
def test_are_we_using_mpi(use_mpi, srun, mpirun, expected):
    with (
        mock.patch.object(utils, "_srun_used", return_value=srun),
        mock.patch.object(utils, "_mpirun_used", return_value=mpirun),
    ):
        assert utils.are_we_using_mpi({"use_mpi": use_mpi}) is expected


@pytest.mark.parametrize(("srun", "mpirun", "launcher"), [(True, False, "srun"), (False, True, "mpirun")])
def test_are_we_using_mpi_turns_on_mpi_for_a_launcher(srun, mpirun, launcher):
    with (
        mock.patch.object(utils, "_srun_used", return_value=srun),
        mock.patch.object(utils, "_mpirun_used", return_value=mpirun),
        pytest.warns(UserWarning, match=f"{launcher}_used = True"),
    ):
        assert utils.are_we_using_mpi({}) is True


# --- _srun_used ---------------------------------------------------------------


def test_srun_used_is_false_outside_slurm():
    assert utils._srun_used() is False


@pytest.mark.parametrize(("parent", "expected"), [("srun", True), ("slurmstepd", True), ("bash", False)])
def test_srun_used_checks_parent_process(parent, expected, monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "4")

    with mock.patch("builtins.open", mock.mock_open(read_data=f"{parent}\n")):
        assert utils._srun_used() is expected


def test_srun_used_is_false_without_proc(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "4")

    with mock.patch("builtins.open", side_effect=FileNotFoundError):
        assert utils._srun_used() is False


# --- _mpirun_used -------------------------------------------------------------


def test_mpirun_used_is_false_without_mpi_env():
    assert utils._mpirun_used() is False


def ps_result(comm, ppid, returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=f"{comm} {ppid}\n")


@pytest.fixture
def macos_with_mpi_env(monkeypatch):
    monkeypatch.setenv("PMI_SIZE", "4")
    monkeypatch.setattr(utils.sys, "platform", "darwin")
    monkeypatch.setattr(utils.os, "getppid", lambda: 100)


def test_mpirun_used_walks_up_the_process_tree_on_macos(macos_with_mpi_env):
    tree = [ps_result("/bin/bash", 50), ps_result("/usr/local/bin/mpiexec.hydra", 1)]

    with mock.patch.object(utils.subprocess, "run", side_effect=tree) as run:
        assert utils._mpirun_used() is True

    assert run.call_count == 2


@pytest.mark.parametrize(
    "tree",
    [
        [ps_result("bash", 1)],  # reaches init without finding a launcher
        [ps_result("bash", 50, returncode=1)],  # ps fails
        [subprocess.CompletedProcess([], 0, stdout="garbled\n")],  # unparseable ps output
    ],
)
def test_mpirun_used_is_false_without_launcher_on_macos(tree, macos_with_mpi_env):
    with mock.patch.object(utils.subprocess, "run", side_effect=tree):
        assert utils._mpirun_used() is False


def test_mpirun_used_reads_proc_on_linux(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setattr(utils.sys, "platform", "linux")
    monkeypatch.setattr(utils.os, "getppid", lambda: 100)
    proc = {
        "/proc/100/comm": "python\n",
        "/proc/100/stat": "100 (python) S 50 rest",
        "/proc/50/comm": "mpirun\n",
        "/proc/50/stat": "50 (mpirun) S 1 rest",
    }

    def fake_open(path, *args, **kwargs):
        return mock.mock_open(read_data=proc[path])()

    with mock.patch("builtins.open", fake_open):
        assert utils._mpirun_used() is True


def test_mpirun_used_is_false_on_error(macos_with_mpi_env):
    with mock.patch.object(utils.subprocess, "run", side_effect=OSError):
        assert utils._mpirun_used() is False

