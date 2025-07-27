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
def sample_fl_config_dict() -> Dict[str, Any]:
    """Sample FLConfig dictionary for testing."""
    return {
        "num_rounds": 10,
        "strategy": "fedavg",
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
        "local_epochs": 1,
        "client_config_files": None,
        "evaluate_frequency": 1,
        "fedprox_mu": 0.01,
        "dp_clipping_norm": 1.0,
        "dp_sensitivity": 1.0,
        "dp_epsilon": 1.0,
        "dp_delta": 1e-5,
        "secagg_num_shares": 3,
        "secagg_reconstruction_threshold": 3,
        "secagg_max_weight": 16777216,
        "secagg_timeout": 30.0,
        "secagg_clipping_range": 1.0,
        "secagg_quantization_range": 1048576,
        "client_id": None,
        "multi_machine": None,
    }


@pytest.fixture
def sample_wandb_config_dict() -> Dict[str, Any]:
    """Sample WandbConfig dictionary for testing."""
    return {
        "use_wandb": True,
        "project": "fl-adni-classification",
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
    sample_fl_config_dict,
    sample_wandb_config_dict,
) -> Dict[str, Any]:
    """Complete sample configuration dictionary for testing."""
    return {
        "data": sample_data_config_dict,
        "model": sample_model_config_dict,
        "training": sample_training_config_dict,
        "fl": sample_fl_config_dict,
        "wandb": sample_wandb_config_dict,
    }


@pytest.fixture
def sample_yaml_file(temp_dir, sample_full_config_dict):
    """Create a temporary YAML config file for testing."""
    yaml_path = temp_dir / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(sample_full_config_dict, f, default_flow_style=False)
    return yaml_path


@pytest.fixture
def sample_ssh_config_dict() -> Dict[str, Any]:
    """Sample SSHConfig dictionary for testing."""
    return {
        "timeout": 30,
        "banner_timeout": 30,
        "auth_timeout": 30,
    }


@pytest.fixture
def sample_client_machine_config_dict() -> Dict[str, Any]:
    """Sample ClientMachineConfig dictionary for testing."""
    return {
        "host": "client1.example.com",
        "username": "testuser",
        "password": "testpass",
        "partition_id": 0,
        "project_dir": "/home/testuser/project",
        "config_file": "client1.yaml",
        "sequential_experiment": False,
        "train_sequential_labels": None,
        "val_sequential_labels": None,
    }


@pytest.fixture
def sample_server_machine_config_dict() -> Dict[str, Any]:
    """Sample ServerMachineConfig dictionary for testing."""
    return {
        "host": "server.example.com",
        "username": "serveruser",
        "password": "serverpass",
        "port": 9092,
        "config_file": "server.yaml",
        "sequential_experiment": False,
        "train_sequential_labels": None,
        "val_sequential_labels": None,
    }


@pytest.fixture
def sample_multi_machine_config_dict(
    sample_server_machine_config_dict,
    sample_client_machine_config_dict,
    sample_ssh_config_dict,
) -> Dict[str, Any]:
    """Sample MultiMachineConfig dictionary for testing."""
    return {
        "server": sample_server_machine_config_dict,
        "clients": [sample_client_machine_config_dict],
        "project_dir": "/home/user/project",
        "venv_path": "/home/user/.venv",
        "venv_activate": "source /home/user/.venv/bin/activate",
        "ssh": sample_ssh_config_dict,
    }
