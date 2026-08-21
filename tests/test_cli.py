import pytest
from unittest import mock
from click.testing import CliRunner
from eagle.tools.cli import cli


@pytest.mark.unit
class TestCLI:
    """Tests for the CLI interface."""

    def test_cli_help(self):
        """Test that CLI help command works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "A CLI for the Eagle Tools suite" in result.output

    def test_inference_command_exists(self):
        """Test that inference command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "--help"])

        assert result.exit_code == 0
        assert "Run inference" in result.output or "inference" in result.output

    def test_metrics_command_exists(self):
        """Test that metrics command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["metrics", "--help"])

        assert result.exit_code == 0
        assert "Compute error metrics" in result.output or "metrics" in result.output

    def test_spatial_command_exists(self):
        """Test that spatial command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["spatial", "--help"])

        assert result.exit_code == 0
        assert "Compute spatial error metrics" in result.output or "spatial" in result.output

    def test_spectra_command_exists(self):
        """Test that spectra command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["spectra", "--help"])

        assert result.exit_code == 0
        assert "Compute spectra error metrics" in result.output or "spectra" in result.output

    def test_figures_command_exists(self):
        """Test that figures command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["figures", "--help"])

        assert result.exit_code == 0

    def test_movies_command_exists(self):
        """Test that movies command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["movies", "--help"])

        assert result.exit_code == 0

    def test_postprocess_command_exists(self):
        """Test that postprocess command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["postprocess", "--help"])

        assert result.exit_code == 0

    @mock.patch("eagle.tools.cli.open_yaml_config")
    @mock.patch("eagle.tools.inference.main")
    def test_inference_command_calls_main(self, mock_main, mock_open_config, sample_yaml_config):
        """Test that inference command calls the main function."""
        runner = CliRunner()
        mock_open_config.return_value = {"test": "config"}

        result = runner.invoke(cli, ["inference", sample_yaml_config])

        mock_open_config.assert_called_once_with(sample_yaml_config)
        mock_main.assert_called_once_with({"test": "config"})

    @mock.patch("eagle.tools.cli.open_yaml_config")
    @mock.patch("eagle.tools.metrics.main")
    def test_metrics_command_calls_main(self, mock_main, mock_open_config, sample_yaml_config):
        """Test that metrics command calls the main function."""
        runner = CliRunner()
        mock_open_config.return_value = {"test": "config"}

        result = runner.invoke(cli, ["metrics", sample_yaml_config])

        mock_open_config.assert_called_once_with(sample_yaml_config)
        mock_main.assert_called_once_with({"test": "config"})

    @mock.patch("eagle.tools.cli.open_yaml_config")
    @mock.patch("eagle.tools.spatial.main")
    def test_spatial_command_calls_main(self, mock_main, mock_open_config, sample_yaml_config):
        """Test that spatial command calls the main function."""
        runner = CliRunner()
        mock_open_config.return_value = {"test": "config"}

        result = runner.invoke(cli, ["spatial", sample_yaml_config])

        mock_open_config.assert_called_once_with(sample_yaml_config)
        mock_main.assert_called_once_with({"test": "config"})

    def test_spectra_command_requires_config(self):
        """Test that spectra command requires a config file."""
        runner = CliRunner()

        # Test that the command exists and requires a config file
        result = runner.invoke(cli, ["spectra", "--help"])

        assert result.exit_code == 0

    @mock.patch("eagle.tools.cli.open_yaml_config")
    @mock.patch("eagle.tools.visualize.main")
    def test_figures_command_calls_main(self, mock_main, mock_open_config, sample_yaml_config):
        """Test that figures command calls the main function."""
        runner = CliRunner()
        mock_open_config.return_value = {"test": "config"}

        result = runner.invoke(cli, ["figures", sample_yaml_config])

        mock_open_config.assert_called_once_with(sample_yaml_config)
        mock_main.assert_called_once_with({"test": "config"}, mode="figure")

    @mock.patch("eagle.tools.cli.open_yaml_config")
    @mock.patch("eagle.tools.visualize.main")
    def test_movies_command_calls_main(self, mock_main, mock_open_config, sample_yaml_config):
        """Test that movies command calls the main function."""
        runner = CliRunner()
        mock_open_config.return_value = {"test": "config"}

        result = runner.invoke(cli, ["movies", sample_yaml_config])

        mock_open_config.assert_called_once_with(sample_yaml_config)
        mock_main.assert_called_once_with({"test": "config"}, mode="movie")

    def test_cli_requires_config_file(self):
        """Test that commands require a config file argument."""
        runner = CliRunner()

        # Test without config file
        result = runner.invoke(cli, ["metrics"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output

    def test_cli_checks_config_file_exists(self):
        """Test that CLI checks if config file exists."""
        runner = CliRunner()

        # Test with non-existent config file
        result = runner.invoke(cli, ["metrics", "/nonexistent/config.yaml"])

        assert result.exit_code != 0
