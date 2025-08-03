"""Application utilities for device management and common operations."""

import torch
from flwr.common import Context
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.task import create_criterion, load_data, load_model


class DeviceManager:
    """Handles device selection and management for FL applications."""

    @staticmethod
    def get_device(context: Context) -> torch.device:
        """Get appropriate device based on context configuration.

        Args:
            context: Flower context containing device configuration

        Returns:
            PyTorch device (CUDA or CPU)
        """
        gpu_idx = context.node_config.get("gpu-id", 0)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_idx}")
            logger.info(f"🔧 Using GPU device: {device}")
        else:
            device = torch.device("cpu")
            logger.info("🔧 Using CPU device (CUDA not available)")
        return device

    @staticmethod
    def get_partition_id(context: Context) -> int:
        """Get partition ID from context.

        Args:
            context: Flower context containing partition configuration

        Returns:
            Partition ID (defaults to 0)
        """
        partition_id = context.node_config.get("partition-id", 0)
        logger.info(f"🔧 Using partition ID: {partition_id}")
        return partition_id


class AppUtils:
    """Common utilities for FL app creation and management."""

    @staticmethod
    def create_model_components(config: Config, device: torch.device):
        """Create model, optimizer, and criterion components.

        Args:
            config: Configuration object
            device: PyTorch device

        Returns:
            Tuple of (model, optimizer, criterion, train_loader, val_loader)

        Raises:
            RuntimeError: If component creation fails
        """
        try:
            # Load model
            model = load_model(config).to(device)
            logger.info("✅ Model loaded successfully")

            # Load data
            train_loader, val_loader = load_data(config)
            logger.info("✅ Data loaders created successfully")

            # Create criterion
            criterion = create_criterion(config, train_loader.dataset, device)
            logger.info("✅ Criterion created successfully")

            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            logger.info("✅ Optimizer created successfully")

            return model, optimizer, criterion, train_loader, val_loader

        except Exception as e:
            raise RuntimeError(f"Failed to create model components: {e}") from e

    @staticmethod
    def create_adam_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Adam:
        """Create Adam optimizer with config parameters.

        Args:
            model: PyTorch model
            config: Configuration object

        Returns:
            Adam optimizer

        Raises:
            RuntimeError: If optimizer creation fails
        """
        try:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            logger.info(f"✅ Adam optimizer created with lr={config.training.learning_rate}")
            return optimizer
        except Exception as e:
            raise RuntimeError(f"Failed to create Adam optimizer: {e}") from e

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer, config: Config, train_loader, num_rounds: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler if specified in config.

        Args:
            optimizer: PyTorch optimizer
            config: Configuration object
            train_loader: Training data loader
            num_rounds: Number of FL rounds

        Returns:
            LR scheduler if specified, None otherwise

        Raises:
            RuntimeError: If scheduler creation fails
        """
        scheduler = None
        if hasattr(config.training, "lr_scheduler") and config.training.lr_scheduler == "cosine":
            try:
                total_steps = config.fl.local_epochs * len(train_loader) * num_rounds
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
                logger.info(f"✅ Cosine scheduler created with {total_steps} total steps")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create cosine scheduler: {e}. Continuing without scheduler.")
        return scheduler

    @staticmethod
    def validate_dp_parameters_for_opacus(config: Config) -> tuple[float, float, float, float]:
        """Validate and extract DP parameters for Opacus with range checking.

        Args:
            config: Configuration object with DP parameters

        Returns:
            Tuple of (epsilon, sensitivity, clipping_norm, delta)

        Raises:
            ValueError: If DP parameters are invalid
        """
        # Extract DP parameters with strict validation
        epsilon = float(config.fl.dp_epsilon)
        sensitivity = float(config.fl.dp_sensitivity)
        clipping_norm = float(config.fl.dp_clipping_norm)
        delta = float(config.fl.dp_delta)

        # Validate DP parameter values
        if epsilon <= 0:
            raise ValueError(f"dp_epsilon must be positive, got: {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"dp_sensitivity must be positive, got: {sensitivity}")
        if clipping_norm <= 0:
            raise ValueError(f"dp_clipping_norm must be positive, got: {clipping_norm}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"dp_delta must be between 0 and 1 (exclusive), got: {delta}")

        # Convert epsilon to noise multiplier for Opacus
        noise_multiplier = sensitivity / epsilon

        # Validate parameters for Opacus numerical stability
        stable_range = f"epsilon between {sensitivity / 2.0:.1f} and {sensitivity / 0.5:.1f} (noise_multiplier 0.5-2.0)"

        if noise_multiplier < 0.5:
            logger.warning(f"⚠️ Small noise multiplier ({noise_multiplier:.6f}) - privacy accounting will be SKIPPED")
            logger.warning(f"   For stable Opacus accounting, use {stable_range}")
        elif noise_multiplier > 2.0:
            logger.warning(f"⚠️ Large noise multiplier ({noise_multiplier:.6f}) - privacy accounting will be SKIPPED")
            logger.warning(f"   For stable Opacus accounting, use {stable_range}")

        # Additional validation for epsilon values
        if epsilon > 2.0:
            logger.warning(f"⚠️ Epsilon ({epsilon}) > 2.0 - privacy accounting will be SKIPPED")
            logger.warning("⚠️ For stable Opacus accounting, use epsilon 0.5-2.0")
        elif epsilon < 0.5:
            logger.warning(f"⚠️ Epsilon ({epsilon}) < 0.5 - privacy accounting will be SKIPPED")
            logger.warning("⚠️ For stable Opacus accounting, use epsilon 0.5-2.0")

        logger.info("🔧 DP parameters validation passed:")
        logger.info(f"   epsilon: {epsilon}")
        logger.info(f"   sensitivity: {sensitivity}")
        logger.info(f"   noise_multiplier: {noise_multiplier:.6f}")
        logger.info(f"   max_grad_norm: {clipping_norm}")

        return epsilon, sensitivity, clipping_norm, delta

    @staticmethod
    def get_client_id(config: Config, partition_id: int) -> int:
        """Get client ID from config or fall back to partition ID.

        Args:
            config: Configuration object
            partition_id: Partition ID as fallback

        Returns:
            Client ID
        """
        client_id = getattr(config.fl, "client_id", partition_id)
        logger.info(f"🔧 Using client ID: {client_id}")
        return client_id

    @staticmethod
    def log_strategy_initialization(strategy_name: str, client_id: int, config_path: str, device: torch.device) -> None:
        """Log strategy initialization information.

        Args:
            strategy_name: Name of the FL strategy
            client_id: Client identifier
            config_path: Path to configuration file
            device: PyTorch device being used
        """
        logger.info(f"🚀 Initializing client {client_id} with {strategy_name} strategy")
        logger.info(f"   Config: {config_path}")
        logger.info(f"   Device: {device}")

    @staticmethod
    def log_fl_rounds(total_fl_rounds: int) -> None:
        """Log FL rounds information.

        Args:
            total_fl_rounds: Total number of FL rounds
        """
        logger.info(f"🔄 Total FL rounds: {total_fl_rounds}")
