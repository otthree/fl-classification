# Tests Directory

This directory contains comprehensive unit tests for the ADNI classification project, specifically focusing on the configuration management system.

## Structure

```
tests/
├── __init__.py                 # Main tests package
├── conftest.py                 # Shared pytest fixtures
├── README.md                   # This documentation
└── config/                     # Configuration tests
    ├── __init__.py            # Config tests subpackage
    ├── test_config.py         # Tests for main config classes
    └── test_fl_config.py      # Tests for federated learning config
```

## Test Coverage

### Configuration Module Tests (`tests/config/`)

#### `test_config.py`
Tests for `adni_classification.config.config` module:

- **DataConfig**: Tests dataset configuration with various options (cache types, transforms, classification modes)
- **ModelConfig**: Tests model configuration for different architectures (ResNet, DenseNet)
- **CheckpointConfig**: Tests checkpoint saving configurations
- **TrainingConfig**: Tests training parameters, loss functions, and optimization settings
- **WandbConfig**: Tests Weights & Biases logging configuration
- **Config**: Tests the main configuration class including:
  - Creation from dictionaries and YAML files
  - Post-processing (run name generation, output directory setup)
  - Serialization (to_dict, to_yaml)
  - Error handling

#### `test_fl_config.py`
Tests for `adni_classification.config.fl_config` module:

- **SSHConfig**: Tests SSH connection parameters for multi-machine setups
- **ClientMachineConfig**: Tests client machine configuration for federated learning
- **ServerMachineConfig**: Tests server machine configuration
- **MultiMachineConfig**: Tests distributed federated learning setup including:
  - Server and client configuration management
  - Environment variable password handling
  - Configuration dictionary conversion
- **FLConfig**: Tests federated learning parameters including:
  - Basic FL strategies (FedAvg, FedProx)
  - Differential privacy settings
  - Secure aggregation (SecAgg+) parameters
  - Client management and evaluation settings

## Fixtures (`conftest.py`)

The `conftest.py` file provides shared fixtures for all tests:

- **Configuration Dictionaries**: Sample configurations for all config classes
- **Temporary Files**: Temporary directories and YAML files for testing file operations
- **Multi-machine Setup**: Complete federated learning setup configurations

## Running the Tests

### Prerequisites

Make sure you have the development dependencies installed:

```bash
pip install -e ".[dev]"
```

This will install pytest, pytest-cov, and other testing dependencies.

### Running All Tests

Run all tests in the project:

```bash
pytest
```

### Running Specific Test Modules

Run only configuration tests:

```bash
pytest tests/config/
```

Run tests for a specific module:

```bash
pytest tests/config/test_config.py
pytest tests/config/test_fl_config.py
```

### Running with Coverage

Generate a coverage report:

```bash
pytest --cov=adni_classification --cov-report=html
```

This will create an HTML coverage report in `htmlcov/index.html`.

### Running Specific Test Cases

Run a specific test class:

```bash
pytest tests/config/test_config.py::TestDataConfig
```

Run a specific test method:

```bash
pytest tests/config/test_config.py::TestConfig::test_config_creation_from_dict
```

### Verbose Output

Run tests with verbose output to see individual test results:

```bash
pytest -v
```

## Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=adni_classification --cov-report=term-missing"
```

## Writing New Tests

When adding new functionality to the configuration system:

1. **Add fixtures** to `conftest.py` if they'll be reused across multiple test files
2. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `TestClassName`
   - Test methods: `test_method_description`
3. **Use descriptive docstrings** to explain what each test verifies
4. **Test edge cases** including error conditions and boundary values
5. **Mock external dependencies** using `unittest.mock` when appropriate

## Example Test Structure

```python
class TestMyConfig:
    """Test cases for MyConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test MyConfig creation with default values."""
        config = MyConfig(required_param="value")

        assert config.required_param == "value"
        assert config.optional_param == "default_value"

    def test_creation_with_custom_values(self):
        """Test MyConfig creation with custom values."""
        config = MyConfig(
            required_param="custom",
            optional_param="custom_value"
        )

        assert config.required_param == "custom"
        assert config.optional_param == "custom_value"
```

## Troubleshooting

### Import Errors
If you encounter import errors, make sure:
1. The package is installed in development mode: `pip install -e .`
2. Your `PYTHONPATH` includes the project root
3. All `__init__.py` files are present in the package structure

### Test Discovery Issues
If pytest can't find your tests:
1. Ensure test files follow the naming convention (`test_*.py`)
2. Check that test directories have `__init__.py` files
3. Verify the `testpaths` setting in `pyproject.toml`

### Coverage Issues
If coverage seems incomplete:
1. Check that you're testing the correct package: `--cov=adni_classification`
2. Ensure all modules have corresponding test files
3. Review the coverage report to identify untested code paths
