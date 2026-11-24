import sys
import types
from unittest import mock

import pytest
from click.testing import CliRunner

from eagle.tools.cli import cli

# command name -> module whose main() it calls, and any extra kwargs it passes
COMMANDS = {
    "inference": ("inference", {}),
    "postprocess": ("postprocess", {}),
    "metrics": ("metrics", {}),
    "spatial": ("spatial", {}),
    "spatial-forecast": ("spatial_forecast", {}),
    "spectra": ("spectra", {}),
    "performance-heatmap": ("performance_heatmap", {}),
    "performance-violin": ("performance_violin", {}),
    "figures": ("visualize", {"mode": "figure"}),
    "movies": ("visualize", {"mode": "movie"}),
    "prewxvx": ("prewxvx", {}),
    "postwxvx": ("postwxvx", {}),
    "obs-metrics": ("obs_metrics", {}),
}


def fake_module(monkeypatch, name):
    """Stand in for eagle.tools.<name>, so the CLI's lazy import gets a mock main."""
    module = types.ModuleType(f"eagle.tools.{name}")
    module.main = mock.Mock()
    monkeypatch.setitem(sys.modules, f"eagle.tools.{name}", module)
    return module.main


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model_type: global\n")
    return str(path)


def test_help_lists_every_command():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    for command in [*COMMANDS, "amlsync"]:
        assert command in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_passes_config_path_to_main(command, config_file, monkeypatch):
    module, kwargs = COMMANDS[command]
    main = fake_module(monkeypatch, module)

    result = CliRunner().invoke(cli, [command, config_file])

    assert result.exit_code == 0, result.output
    main.assert_called_once_with(config_file, **kwargs)


def test_missing_config_argument_fails():
    result = CliRunner().invoke(cli, ["metrics"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_nonexistent_config_file_fails():
    result = CliRunner().invoke(cli, ["metrics", "/nonexistent/config.yaml"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_amlsync_passes_options_to_main(tmp_path, monkeypatch):
    main = fake_module(monkeypatch, "amlsync")

    result = CliRunner().invoke(
        cli,
        ["amlsync", "--offline_path", str(tmp_path), "--local_id", "abc", "--remote_name", "exp"],
    )

    assert result.exit_code == 0, result.output
    main.assert_called_once_with(offline_path=str(tmp_path), local_id="abc", remote_name="exp")
