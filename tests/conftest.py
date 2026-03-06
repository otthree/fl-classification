"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_data_config_dict() -> Dict[str, Any]:
    """Sample DataConfig dictionary for testing."""
    return {
        "train_csv_path": "/path/to/train.csv",
        "val_csv_path": "/path/to/val.csv",
        "img_dir": "/path/to/images",
        "dataset_type": "normal",
        "resize_size": [160, 160, 160],
        "resize_mode": "trilinear",
        "use_spacing": False,
        "spacing_size": [1.0, 1.0, 1.0],
        "cache_rate": 1.0,
        "cache_num_workers": 8,
        "cache_dir": "./persistent_cache",
        "use_multiprocessing_transforms": False,
        "transform_device": None,
        "multiprocessing_context": "spawn",
        "classification_mode": "CN_MCI_AD",
        "mci_subtype_filter": None,
    }


@pytest.fixture
def sample_model_config_dict() -> Dict[str, Any]:
    """Sample ModelConfig dictionary for testing."""
    return {
        "name": "resnet3d",
        "num_classes": 3,
        "pretrained_checkpoint": None,
        "model_depth": 18,
        "growth_rate": None,
        "block_config": None,
        "freeze_encoder": None,
        "dropout": None,
        "input_channels": None,
    }


@pytest.fixture
def sample_checkpoint_config_dict() -> Dict[str, Any]:
    """Sample CheckpointConfig dictionary for testing."""
    return {
        "save_best": True,
        "save_latest": True,
        "save_regular": False,
        "save_frequency": 10,
    }


@pytest.fixture
def sample_training_config_dict(sample_checkpoint_config_dict) -> Dict[str, Any]:
    """Sample TrainingConfig dictionary for testing."""
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_workers": 4,
        "output_dir": "outputs",
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "mixed_precision": False,
        "visualize": False,
        "lr_scheduler": "plateau",
        "val_epoch_freq": 5,
        "use_class_weights": False,
        "class_weight_type": "inverse",
        "manual_class_weights": None,
        "loss_type": "cross_entropy",
        "focal_alpha": None,
        "focal_gamma": 2.0,
        "checkpoint": sample_checkpoint_config_dict,
    }


@pytest.fixture
def sample_wandb_config_dict() -> Dict[str, Any]:
    """Sample WandbConfig dictionary for testing."""
    return {
        "use_wandb": True,
        "project": "adni1",
        "entity": None,
        "tags": ["test", "unit"],
        "notes": "Test run",
        "run_name": "",
        "enable_shared_mode": True,
        "shared_run_id": None,
    }


@pytest.fixture
def sample_full_config_dict(
    sample_data_config_dict,
    sample_model_config_dict,
    sample_training_config_dict,
    sample_wandb_config_dict,
) -> Dict[str, Any]:
    """Complete sample configuration dictionary for testing."""
    return {
        "data": sample_data_config_dict,
        "model": sample_model_config_dict,
        "training": sample_training_config_dict,
        "wandb": sample_wandb_config_dict,
    }


@pytest.fixture
def sample_yaml_file(temp_dir, sample_full_config_dict):
    """Create a temporary YAML config file for testing."""
    yaml_path = temp_dir / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(sample_full_config_dict, f, default_flow_style=False)
    return yaml_path


