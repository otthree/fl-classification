"""Unit tests for adni_classification.config.config module."""

import os
from unittest.mock import patch

import pytest
import yaml

from adni_classification.config.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)


class TestDataConfig:
    """Test cases for DataConfig dataclass."""

    def test_data_config_creation_with_defaults(self):
        """Test DataConfig creation with minimal required fields."""
        config = DataConfig(
            train_csv_path="/path/to/train.csv",
            val_csv_path="/path/to/val.csv",
            img_dir="/path/to/images",
        )

        assert config.train_csv_path == "/path/to/train.csv"
        assert config.val_csv_path == "/path/to/val.csv"
        assert config.img_dir == "/path/to/images"
        assert config.dataset_type == "normal"
        assert config.resize_size == [160, 160, 160]
        assert config.resize_mode == "trilinear"
        assert config.use_spacing is False
        assert config.spacing_size == [1.0, 1.0, 1.0]
        assert config.cache_rate == 1.0
        assert config.cache_num_workers == 8
        assert config.cache_dir == "./persistent_cache"
        assert config.use_multiprocessing_transforms is False
        assert config.transform_device is None
        assert config.multiprocessing_context == "spawn"
        assert config.classification_mode == "CN_MCI_AD"
        assert config.mci_subtype_filter is None

    def test_data_config_creation_with_custom_values(self):
        """Test DataConfig creation with custom values."""
        config = DataConfig(
            train_csv_path="/custom/train.csv",
            val_csv_path="/custom/val.csv",
            img_dir="/custom/images",
            dataset_type="cache",
            resize_size=[128, 128, 128],
            resize_mode="nearest",
            use_spacing=True,
            spacing_size=[0.5, 0.5, 0.5],
            cache_rate=0.8,
            cache_num_workers=4,
            cache_dir="/custom/cache",
            use_multiprocessing_transforms=True,
            transform_device="cuda",
            multiprocessing_context="fork",
            classification_mode="CN_AD",
            mci_subtype_filter=["EMCI", "LMCI"],
        )

        assert config.train_csv_path == "/custom/train.csv"
        assert config.val_csv_path == "/custom/val.csv"
        assert config.img_dir == "/custom/images"
        assert config.dataset_type == "cache"
        assert config.resize_size == [128, 128, 128]
        assert config.resize_mode == "nearest"
        assert config.use_spacing is True
        assert config.spacing_size == [0.5, 0.5, 0.5]
        assert config.cache_rate == 0.8
        assert config.cache_num_workers == 4
        assert config.cache_dir == "/custom/cache"
        assert config.use_multiprocessing_transforms is True
        assert config.transform_device == "cuda"
        assert config.multiprocessing_context == "fork"
        assert config.classification_mode == "CN_AD"
        assert config.mci_subtype_filter == ["EMCI", "LMCI"]


class TestModelConfig:
    """Test cases for ModelConfig dataclass."""

    def test_model_config_creation_with_defaults(self):
        """Test ModelConfig creation with minimal required fields."""
        config = ModelConfig(name="resnet3d")

        assert config.name == "resnet3d"
        assert config.num_classes == 3
        assert config.pretrained_checkpoint is None
        assert config.model_depth is None
        assert config.growth_rate is None
        assert config.block_config is None
        assert config.freeze_encoder is None
        assert config.dropout is None
        assert config.input_channels is None

    def test_model_config_creation_resnet(self):
        """Test ModelConfig creation for ResNet model."""
        config = ModelConfig(
            name="resnet3d",
            num_classes=2,
            model_depth=18,
            pretrained_checkpoint="/path/to/checkpoint.pth",
        )

        assert config.name == "resnet3d"
        assert config.num_classes == 2
        assert config.model_depth == 18
        assert config.pretrained_checkpoint == "/path/to/checkpoint.pth"

    def test_model_config_creation_densenet(self):
        """Test ModelConfig creation for DenseNet model."""
        config = ModelConfig(
            name="densenet3d",
            num_classes=3,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            dropout=0.2,
        )

        assert config.name == "densenet3d"
        assert config.num_classes == 3
        assert config.growth_rate == 32
        assert config.block_config == (6, 12, 24, 16)
        assert config.dropout == 0.2


class TestCheckpointConfig:
    """Test cases for CheckpointConfig dataclass."""

    def test_checkpoint_config_creation_with_defaults(self):
        """Test CheckpointConfig creation with default values."""
        config = CheckpointConfig()

        assert config.save_best is True
        assert config.save_latest is True
        assert config.save_regular is False
        assert config.save_frequency == 10

    def test_checkpoint_config_creation_custom(self):
        """Test CheckpointConfig creation with custom values."""
        config = CheckpointConfig(
            save_best=False,
            save_latest=False,
            save_regular=True,
            save_frequency=5,
        )

        assert config.save_best is False
        assert config.save_latest is False
        assert config.save_regular is True
        assert config.save_frequency == 5


class TestTrainingConfig:
    """Test cases for TrainingConfig dataclass."""

    def test_training_config_creation_with_required_fields(self):
        """Test TrainingConfig creation with minimal required fields."""
        config = TrainingConfig(
            batch_size=8,
            num_epochs=10,
            learning_rate=0.001,
            weight_decay=1e-5,
            num_workers=4,
            output_dir="outputs",
        )

        assert config.batch_size == 8
        assert config.num_epochs == 10
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-5
        assert config.num_workers == 4
        assert config.output_dir == "outputs"
        assert config.seed == 42
        assert config.gradient_accumulation_steps == 1
        assert config.mixed_precision is False
        assert config.visualize is False
        assert config.lr_scheduler == "plateau"
        assert config.val_epoch_freq == 5
        assert config.use_class_weights is False
        assert config.class_weight_type == "inverse"
        assert config.manual_class_weights is None
        assert config.loss_type == "cross_entropy"
        assert config.focal_alpha is None
        assert config.focal_gamma == 2.0
        assert isinstance(config.checkpoint, CheckpointConfig)

    def test_training_config_creation_with_custom_checkpoint(self):
        """Test TrainingConfig creation with custom checkpoint config."""
        checkpoint_config = CheckpointConfig(save_frequency=5)
        config = TrainingConfig(
            batch_size=16,
            num_epochs=20,
            learning_rate=0.01,
            weight_decay=1e-4,
            num_workers=8,
            output_dir="custom_outputs",
            checkpoint=checkpoint_config,
        )

        assert config.checkpoint.save_frequency == 5
        assert config.batch_size == 16
        assert config.num_epochs == 20

    def test_training_config_focal_loss_parameters(self):
        """Test TrainingConfig with focal loss parameters."""
        config = TrainingConfig(
            batch_size=8,
            num_epochs=10,
            learning_rate=0.001,
            weight_decay=1e-5,
            num_workers=4,
            output_dir="outputs",
            loss_type="focal",
            focal_alpha=0.25,
            focal_gamma=2.5,
        )

        assert config.loss_type == "focal"
        assert config.focal_alpha == 0.25
        assert config.focal_gamma == 2.5


class TestWandbConfig:
    """Test cases for WandbConfig dataclass."""

    def test_wandb_config_creation_with_required_fields(self):
        """Test WandbConfig creation with minimal required fields."""
        config = WandbConfig(
            use_wandb=True,
            project="test-project",
        )

        assert config.use_wandb is True
        assert config.project == "test-project"
        assert config.entity is None
        assert config.tags == []
        assert config.notes == ""
        assert config.run_name == ""
        assert config.enable_shared_mode is True
        assert config.shared_run_id is None

    def test_wandb_config_creation_with_custom_values(self):
        """Test WandbConfig creation with custom values."""
        config = WandbConfig(
            use_wandb=False,
            project="custom-project",
            entity="test-entity",
            tags=["experiment", "test"],
            notes="Test experiment",
            run_name="test_run",
            enable_shared_mode=False,
            shared_run_id="shared_123",
        )

        assert config.use_wandb is False
        assert config.project == "custom-project"
        assert config.entity == "test-entity"
        assert config.tags == ["experiment", "test"]
        assert config.notes == "Test experiment"
        assert config.run_name == "test_run"
        assert config.enable_shared_mode is False
        assert config.shared_run_id == "shared_123"


class TestConfig:
    """Test cases for the main Config class."""

    def test_config_creation_from_dict(self, sample_full_config_dict):
        """Test Config creation from dictionary."""
        config = Config.from_dict(sample_full_config_dict)

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.wandb, WandbConfig)

        assert config.data.train_csv_path == "/path/to/train.csv"
        assert config.model.name == "resnet3d"
        assert config.training.batch_size == 8
        assert config.wandb.project == "adni1"

    def test_config_creation_from_yaml(self, sample_yaml_file):
        """Test Config creation from YAML file."""
        config = Config.from_yaml(str(sample_yaml_file))

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.wandb, WandbConfig)

    def test_config_from_dict_with_partial_data(self):
        """Test Config creation from dictionary with minimal data."""
        minimal_dict = {
            "data": {
                "train_csv_path": "/path/to/train.csv",
                "val_csv_path": "/path/to/val.csv",
                "img_dir": "/path/to/images",
            },
            "model": {"name": "resnet3d"},
            "training": {
                "batch_size": 8,
                "num_epochs": 10,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "num_workers": 4,
                "output_dir": "outputs",
            },
            "wandb": {
                "use_wandb": False,
                "project": "test",
            },
        }

        config = Config.from_dict(minimal_dict)

        assert config.data.dataset_type == "normal"  # default value
        assert config.model.num_classes == 3  # default value
        assert config.training.seed == 42  # default value

    @patch("datetime.datetime")
    def test_config_post_process_run_name_generation(self, mock_datetime, sample_full_config_dict):
        """Test that _post_process generates appropriate run names."""
        # Mock datetime to return a fixed timestamp
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        # Test with empty run_name - should include model depth since it's specified in sample config
        sample_full_config_dict["wandb"]["run_name"] = ""
        config = Config.from_dict(sample_full_config_dict)
        assert config.wandb.run_name == "resnet3d18_20240101_120000"

        # Test with existing run_name
        sample_full_config_dict["wandb"]["run_name"] = "my_experiment"
        config = Config.from_dict(sample_full_config_dict)
        assert config.wandb.run_name == "my_experiment_20240101_120000"

    @patch("datetime.datetime")
    def test_config_post_process_output_directory(self, mock_datetime, sample_full_config_dict):
        """Test that _post_process sets appropriate output directories."""
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        # Test with default output_dir - should include model depth since it's specified in sample config
        sample_full_config_dict["training"]["output_dir"] = "outputs"
        sample_full_config_dict["wandb"]["run_name"] = ""
        config = Config.from_dict(sample_full_config_dict)
        assert config.training.output_dir == "outputs/resnet3d18_20240101_120000"

    def test_config_checkpoint_dir_property(self, sample_full_config_dict):
        """Test the checkpoint_dir property."""
        config = Config.from_dict(sample_full_config_dict)
        expected_checkpoint_dir = os.path.join(config.training.output_dir, "checkpoints")
        assert config.checkpoint_dir == expected_checkpoint_dir

    def test_config_to_dict(self, sample_full_config_dict):
        """Test Config conversion to dictionary."""
        config = Config.from_dict(sample_full_config_dict)
        config_dict = config.to_dict()

        assert "data" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert "wandb" in config_dict

        assert config_dict["data"]["train_csv_path"] == "/path/to/train.csv"
        assert config_dict["model"]["name"] == "resnet3d"
        assert config_dict["training"]["batch_size"] == 8

    def test_config_to_yaml(self, temp_dir, sample_full_config_dict):
        """Test Config conversion to YAML file."""
        config = Config.from_dict(sample_full_config_dict)
        yaml_path = temp_dir / "output_config.yaml"

        config.to_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Load the saved YAML and verify content
        with open(yaml_path, "r") as f:
            saved_dict = yaml.safe_load(f)

        assert saved_dict["data"]["train_csv_path"] == "/path/to/train.csv"
        assert saved_dict["model"]["name"] == "resnet3d"

    def test_config_to_yaml_creates_directory(self, temp_dir, sample_full_config_dict):
        """Test that to_yaml creates necessary directories."""
        config = Config.from_dict(sample_full_config_dict)
        nested_path = temp_dir / "subdir" / "config.yaml"

        config.to_yaml(str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_config_from_yaml_file_not_found(self, temp_dir):
        """Test Config creation from non-existent YAML file."""
        non_existent_path = temp_dir / "non_existent.yaml"

        with pytest.raises(FileNotFoundError):
            Config.from_yaml(str(non_existent_path))

    def test_config_from_dict_with_checkpoint_config(self):
        """Test Config creation with nested checkpoint configuration."""
        config_dict = {
            "data": {
                "train_csv_path": "/path/to/train.csv",
                "val_csv_path": "/path/to/val.csv",
                "img_dir": "/path/to/images",
            },
            "model": {"name": "resnet3d"},
            "training": {
                "batch_size": 8,
                "num_epochs": 10,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "num_workers": 4,
                "output_dir": "outputs",
                "checkpoint": {
                    "save_best": False,
                    "save_frequency": 5,
                },
            },
            "wandb": {
                "use_wandb": False,
                "project": "test",
            },
        }

        config = Config.from_dict(config_dict)

        assert config.training.checkpoint.save_best is False
        assert config.training.checkpoint.save_frequency == 5
        assert config.training.checkpoint.save_latest is True  # default value

    def test_config_model_name_in_run_name_with_depth(self, sample_full_config_dict):
        """Test that model depth is included in run name for ResNet models."""
        sample_full_config_dict["model"]["model_depth"] = 50
        sample_full_config_dict["wandb"]["run_name"] = ""

        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            config = Config.from_dict(sample_full_config_dict)
            assert config.wandb.run_name == "resnet3d50_20240101_120000"
