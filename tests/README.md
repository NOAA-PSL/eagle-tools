# Eagle Tools Test Suite

This directory contains the unit test suite for the eagle-tools package.

## Overview

The test suite is built using pytest and covers the core functionality of the eagle-tools package. Tests are organized by module and use fixtures defined in `conftest.py` for common test data.

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_utils.py
pytest tests/test_reshape.py
```

### Run Tests with Coverage

```bash
pytest --cov=eagle.tools --cov-report=html
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow
```

## Test Organization

- `conftest.py` - Shared pytest fixtures for creating test datasets
- `test_utils.py` - Tests for utility functions (config loading, etc.)
- `test_log.py` - Tests for logging functionality
- `test_reshape.py` - Tests for array reshaping functions
- `test_data.py` - Tests for data loading and processing functions
- `test_metrics.py` - Tests for error metric calculations (RMSE, MAE)
- `test_spatial.py` - Tests for spatial metric computations
- `test_cli.py` - Tests for the command-line interface

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Fast unit tests that don't require external data
- `@pytest.mark.integration` - Integration tests that may require datasets
- `@pytest.mark.slow` - Slow-running tests

## Fixtures

Common fixtures available in `conftest.py`:

- `sample_dataset` - Basic xarray dataset with time and cell dimensions
- `sample_dataset_2d` - 2D gridded dataset with x/y dimensions
- `sample_dataset_with_levels` - Dataset with vertical levels
- `sample_dataset_with_wind` - Dataset with wind components
- `sample_dataset_with_member` - Dataset with ensemble member dimension
- `sample_latlon_dataset` - Dataset with lat/lon grid
- `temp_dir` - Temporary directory for file operations
- `sample_yaml_config` - Sample YAML configuration file

## Writing New Tests

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Add appropriate markers (`@pytest.mark.unit`, etc.)
4. Test both success cases and edge cases
5. Use descriptive test names that explain what is being tested

Example:

```python
import pytest

@pytest.mark.unit
class TestMyFunction:
    """Tests for my_function."""

    def test_basic_functionality(self, sample_dataset):
        """Test basic functionality of my_function."""
        result = my_function(sample_dataset)
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(None)
```

## Notes

- Most tests are designed to run without requiring large external datasets
- Tests use randomly generated data to ensure reproducibility (with fixed random seeds)
- Integration tests that require actual anemoi datasets should be marked with `@pytest.mark.integration`
- The test suite focuses on unit testing individual functions rather than end-to-end workflows
