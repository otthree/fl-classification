"""ADNI Classification Package.

A comprehensive package for Alzheimer's Disease Neuroimaging Initiative (ADNI)
3-way classification (CN vs MCI vs AD) using deep learning on 3D MRI data.

This package provides:
- Dataset handling and preprocessing utilities
- Deep learning models for medical image classification
- Configuration management
- Utilities for training and evaluation
"""

import os
import sys

from .config import Config

# Check if we're in a test environment to avoid loading heavy dependencies
_is_testing = (
    "pytest" in sys.modules
    or "coverage" in sys.modules
    or "PYTEST_CURRENT_TEST" in os.environ
    or any("test" in arg for arg in sys.argv)
)

if not _is_testing:
    # Only import heavy dependencies when not testing
    try:
        from .datasets import create_adni_dataset
        from .models import ModelFactory
        from .utils import FocalLoss, create_loss_function
    except ImportError:
        # If imports fail, use lazy loading
        def __getattr__(name: str):
            """Lazy import for heavy dependencies."""
            if name == "create_adni_dataset":
                from .datasets import create_adni_dataset

                return create_adni_dataset
            elif name == "ModelFactory":
                from .models import ModelFactory

                return ModelFactory
            elif name == "FocalLoss":
                from .utils import FocalLoss

                return FocalLoss
            elif name == "create_loss_function":
                from .utils import create_loss_function

                return create_loss_function
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
else:
    # During testing, use lazy imports to avoid PyTorch/MONAI conflicts
    def __getattr__(name: str):
        """Lazy import for heavy dependencies during testing."""
        if name == "create_adni_dataset":
            from .datasets import create_adni_dataset

            return create_adni_dataset
        elif name == "ModelFactory":
            from .models import ModelFactory

            return ModelFactory
        elif name == "FocalLoss":
            from .utils import FocalLoss

            return FocalLoss
        elif name == "create_loss_function":
            from .utils import create_loss_function

            return create_loss_function
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__version__ = "1.0.0"

__all__ = [
    "Config",
    "create_adni_dataset",
    "ModelFactory",
    "FocalLoss",
    "create_loss_function",
]
