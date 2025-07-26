"""ADNI Classification Package.

A comprehensive package for Alzheimer's Disease Neuroimaging Initiative (ADNI)
classification using federated learning and deep learning techniques.

This package provides:
- Dataset handling and preprocessing utilities
- Deep learning models for medical image classification
- Configuration management
- Utilities for training and evaluation
"""

from .config import Config, FLConfig
from .datasets import create_adni_dataset
from .models import ModelFactory
from .utils import FocalLoss, create_loss_function

__version__ = "1.0.0"

__all__ = [
    "Config",
    "FLConfig",
    "create_adni_dataset",
    "ModelFactory",
    "FocalLoss",
    "create_loss_function",
]
