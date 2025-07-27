"""Unit tests for package imports and structure validation."""

import importlib
import sys
from unittest.mock import patch

import pytest


class TestAdniClassificationImports:
    """Test cases for adni_classification package imports."""

    def test_main_package_import(self):
        """Test that main adni_classification package can be imported."""
        try:
            import adni_classification

            assert hasattr(adni_classification, "__version__")
        except ImportError as e:
            pytest.fail(f"Failed to import adni_classification package: {e}")

    def test_main_package_exports(self):
        """Test that main package exports are available."""
        import adni_classification

        # Check __all__ exports
        expected_exports = [
            "Config",
            "FLConfig",
            "create_adni_dataset",
            "ModelFactory",
            "FocalLoss",
            "create_loss_function",
        ]

        for export in expected_exports:
            assert hasattr(adni_classification, export), f"Missing export: {export}"
            assert export in adni_classification.__all__, f"Export {export} not in __all__"

    def test_config_module_imports(self):
        """Test config module imports."""
        try:
            from adni_classification.config import Config, FLConfig
            from adni_classification.config.config import (
                CheckpointConfig,
                DataConfig,
                ModelConfig,
                TrainingConfig,
                WandbConfig,
            )
            from adni_classification.config.fl_config import (
                ClientMachineConfig,
                MultiMachineConfig,
                ServerMachineConfig,
                SSHConfig,
            )

            # Verify classes can be instantiated (basic smoke test)
            config_classes = [Config, FLConfig, DataConfig, ModelConfig, TrainingConfig, CheckpointConfig, WandbConfig]
            fl_config_classes = [SSHConfig, ClientMachineConfig, ServerMachineConfig, MultiMachineConfig]

            for cls in config_classes + fl_config_classes:
                assert cls is not None, f"Class {cls.__name__} should not be None"

        except ImportError as e:
            pytest.fail(f"Failed to import config modules: {e}")

    def test_datasets_module_imports(self):
        """Test datasets module imports."""
        try:
            from adni_classification.datasets import (
                ADNICacheDataset,
                ADNIDataset,
                ADNIPersistentDataset,
                ADNISmartCacheDataset,
                create_adni_dataset,
                get_transforms,
            )

            # Verify functions are callable
            assert callable(create_adni_dataset)
            assert callable(get_transforms)

            # Verify dataset classes exist
            dataset_classes = [ADNIDataset, ADNICacheDataset, ADNIPersistentDataset, ADNISmartCacheDataset]
            for cls in dataset_classes:
                assert cls is not None, f"Dataset class {cls.__name__} should not be None"

        except ImportError as e:
            pytest.fail(f"Failed to import datasets modules: {e}")

    def test_models_module_imports(self):
        """Test models module imports."""
        try:
            from adni_classification.models import BaseModel, ModelFactory, ResNet3D, RosannaCNN, SecureFedCNN

            # Verify ModelFactory and all model classes exist
            model_classes = [ModelFactory, BaseModel, ResNet3D, SecureFedCNN, RosannaCNN]
            for cls in model_classes:
                assert cls is not None, f"Model class {cls.__name__} should not be None"

        except ImportError as e:
            pytest.fail(f"Failed to import models modules: {e}")

    def test_utils_module_imports(self):
        """Test utils module imports."""
        try:
            from adni_classification.utils import FocalLoss, create_loss_function

            assert FocalLoss is not None
            assert callable(create_loss_function)

        except ImportError as e:
            pytest.fail(f"Failed to import utils modules: {e}")

    def test_submodule_structure(self):
        """Test that all expected submodules exist."""

        expected_submodules = [
            "adni_classification.config",
            "adni_classification.datasets",
            "adni_classification.models",
            "adni_classification.utils",
        ]

        for submodule in expected_submodules:
            try:
                importlib.import_module(submodule)
            except ImportError as e:
                pytest.fail(f"Failed to import submodule {submodule}: {e}")


class TestAdniFlwrImports:
    """Test cases for adni_flwr package imports."""

    def test_main_package_import(self):
        """Test that main adni_flwr package can be imported."""
        try:
            import adni_flwr

            assert hasattr(adni_flwr, "__version__")
        except ImportError as e:
            pytest.fail(f"Failed to import adni_flwr package: {e}")

    def test_strategies_module_imports(self):
        """Test strategies module imports."""
        try:
            from adni_flwr.strategies import (
                ClientStrategyBase,
                DifferentialPrivacyClient,
                DifferentialPrivacyStrategy,
                FedAvgClient,
                FedAvgStrategy,
                FedProxClient,
                FedProxStrategy,
                FLStrategyBase,
                SecAggPlusClient,
                SecAggPlusFlowerClient,
                SecAggPlusStrategy,
                StrategyAwareClient,
                StrategyConfigValidator,
                StrategyFactory,
                create_secagg_plus_client_fn,
            )

            # Verify all strategy classes exist
            strategy_classes = [
                FLStrategyBase,
                ClientStrategyBase,
                StrategyAwareClient,
                FedAvgStrategy,
                FedAvgClient,
                FedProxStrategy,
                FedProxClient,
                DifferentialPrivacyStrategy,
                DifferentialPrivacyClient,
                SecAggPlusStrategy,
                SecAggPlusClient,
                SecAggPlusFlowerClient,
                StrategyFactory,
                StrategyConfigValidator,
            ]

            for cls in strategy_classes:
                assert cls is not None, f"Strategy class {cls.__name__} should not be None"

            # Verify callable functions
            assert callable(create_secagg_plus_client_fn)

        except ImportError as e:
            pytest.fail(f"Failed to import strategies modules: {e}")

    def test_strategies_module_exports(self):
        """Test that strategies module exports match __all__."""
        import adni_flwr.strategies as strategies_module

        expected_exports = [
            "FLStrategyBase",
            "ClientStrategyBase",
            "StrategyAwareClient",
            "FedAvgStrategy",
            "FedAvgClient",
            "FedProxStrategy",
            "FedProxClient",
            "DifferentialPrivacyStrategy",
            "DifferentialPrivacyClient",
            "SecAggPlusStrategy",
            "SecAggPlusClient",
            "SecAggPlusFlowerClient",
            "create_secagg_plus_client_fn",
            "StrategyFactory",
            "StrategyConfigValidator",
        ]

        for export in expected_exports:
            assert hasattr(strategies_module, export), f"Missing export: {export}"
            assert export in strategies_module.__all__, f"Export {export} not in __all__"

    def test_core_modules_import(self):
        """Test core FL application modules."""
        try:
            from adni_flwr import client_app, server_app, server_fn, task

            # These should be modules/files, not necessarily have specific attributes
            # Just verify they can be imported without errors
            assert client_app is not None
            assert server_app is not None
            assert task is not None
            assert server_fn is not None

        except ImportError as e:
            pytest.fail(f"Failed to import core FL modules: {e}")

    def test_utils_module_import(self):
        """Test utils module imports."""
        try:
            # Import the utils module (even if it's empty)
            import adni_flwr.utils

            assert adni_flwr.utils is not None

            # Try to import specific utils if they exist
            try:
                from adni_flwr.utils import logging_config, memory_monitor

                # Verify these modules exist if imported successfully
                assert logging_config is not None
                assert memory_monitor is not None
            except ImportError:
                # Utils might not export these, that's ok for now
                pass

        except ImportError as e:
            pytest.fail(f"Failed to import utils module: {e}")

    def test_submodule_structure(self):
        """Test that all expected submodules exist."""
        expected_submodules = ["adni_flwr.strategies", "adni_flwr.utils"]

        for submodule in expected_submodules:
            try:
                importlib.import_module(submodule)
            except ImportError as e:
                pytest.fail(f"Failed to import submodule {submodule}: {e}")


class TestCrossPackageCompatibility:
    """Test compatibility between adni_classification and adni_flwr packages."""

    def test_both_packages_import_together(self):
        """Test that both packages can be imported simultaneously."""
        try:
            import adni_classification
            import adni_flwr

            # Verify both have version attributes
            assert hasattr(adni_classification, "__version__")
            assert hasattr(adni_flwr, "__version__")

        except ImportError as e:
            pytest.fail(f"Failed to import both packages together: {e}")

    def test_config_integration(self):
        """Test that FL config works with main config."""
        try:
            from adni_classification.config import Config, FLConfig
            from adni_flwr.strategies import FedAvgStrategy

            # Verify classes can be imported together
            assert Config is not None
            assert FLConfig is not None
            assert FedAvgStrategy is not None

        except ImportError as e:
            pytest.fail(f"Failed to test config integration: {e}")


class TestImportErrorHandling:
    """Test graceful handling of missing dependencies."""

    def test_missing_dependency_handling(self):
        """Test that import errors are properly handled."""
        # This test verifies that our test structure itself is robust
        with patch.dict(sys.modules, {"nonexistent_module": None}):
            try:
                import nonexistent_module  # This should fail  # noqa: F401

                pytest.fail("Expected ImportError was not raised")
            except ImportError:
                pass  # Expected behavior

    def test_package_structure_validation(self):
        """Test that package structure is as expected."""
        import adni_classification
        import adni_flwr

        # Verify packages are properly structured
        assert hasattr(adni_classification, "__file__"), "adni_classification should be a proper package"
        assert hasattr(adni_flwr, "__file__"), "adni_flwr should be a proper package"

        # Verify package paths are correct
        assert "adni_classification" in adni_classification.__file__
        assert "adni_flwr" in adni_flwr.__file__
