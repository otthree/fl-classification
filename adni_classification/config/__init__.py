"""Configuration package for ADNI classification."""

from .config import Config, DataConfig, ModelConfig, TrainingConfig
from .fl_config import FLConfig

__all__ = ["Config", "DataConfig", "ModelConfig", "TrainingConfig", "FLConfig"]
