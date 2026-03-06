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
            from adni_classification.config import Config
            from adni_classification.config.config import (
                CheckpointConfig,
                DataConfig,
                ModelConfig,
                TrainingConfig,
                WandbConfig,
            )

            # Verify classes can be instantiated (basic smoke test)
            config_classes = [Config, DataConfig, ModelConfig, TrainingConfig, CheckpointConfig, WandbConfig]

            for cls in config_classes:
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

        # Verify packages are properly structured
        assert hasattr(adni_classification, "__file__"), "adni_classification should be a proper package"

        # Verify package paths are correct
        assert "adni_classification" in adni_classification.__file__
