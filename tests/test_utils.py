import pytest
import os
import yaml
from pathlib import Path
from eagle.tools.utils import open_yaml_config


@pytest.mark.unit
class TestOpenYamlConfig:
    """Tests for the open_yaml_config function."""

    def test_open_yaml_config_basic(self, temp_dir):
        """Test basic YAML config loading."""
        config_data = {
            "model_type": "global",
            "start_date": "2024-01-01",
            "levels": [850, 500, 250],
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = open_yaml_config(str(config_file))

        assert config["model_type"] == "global"
        assert config["start_date"] == "2024-01-01"
        assert config["levels"] == [850, 500, 250]

    def test_open_yaml_config_with_env_vars(self, temp_dir, monkeypatch):
        """Test that environment variables in paths are expanded."""
        test_path = str(temp_dir / "test_base")
        monkeypatch.setenv("TEST_VAR", test_path)

        config_data = {
            "input_path": "$TEST_VAR/input",
            "output_path": "$TEST_VAR/output",
            "model_type": "global",
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = open_yaml_config(str(config_file))

        assert config["input_path"] == f"{test_path}/input"
        assert config["output_path"] == f"{test_path}/output"
        # output_path should be created
        assert Path(config["output_path"]).exists()

    def test_open_yaml_config_creates_output_path(self, temp_dir):
        """Test that output_path is created if it doesn't exist."""
        output_path = temp_dir / "new_output_dir"
        assert not output_path.exists()

        config_data = {
            "output_path": str(output_path),
            "model_type": "global",
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = open_yaml_config(str(config_file))

        assert output_path.exists()
        assert output_path.is_dir()

    def test_open_yaml_config_doesnt_expand_non_path_keys(self, temp_dir, monkeypatch):
        """Test that environment variables in non-path keys are not expanded."""
        monkeypatch.setenv("MODEL_TYPE", "global")

        config_data = {
            "model_type": "$MODEL_TYPE",  # Not a path key, shouldn't be expanded
            "output_path": str(temp_dir),
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = open_yaml_config(str(config_file))

        # Should not be expanded since "model_type" doesn't contain "path"
        assert config["model_type"] == "$MODEL_TYPE"

    def test_open_yaml_config_with_list_path_value(self, temp_dir):
        """Test that list values for path keys generate a warning and are not expanded."""
        config_data = {
            "input_path": ["/path1", "/path2"],  # List, not string
            "output_path": str(temp_dir),
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Should not raise an error, just log a warning
        config = open_yaml_config(str(config_file))

        assert config["input_path"] == ["/path1", "/path2"]
