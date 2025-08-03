"""Opacus-based differential privacy client factory."""

import torch
from flwr.client import ClientApp
from flwr.common import Context
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.apps.base import ClientAppFactoryMixin, DifferentialPrivacyMixin
from adni_flwr.strategies import StrategyAwareClient
from adni_flwr.strategies.differential_privacy_opacus import DifferentialPrivacyClient as OpacusDPClient
from adni_flwr.task import create_criterion, load_data, load_model
from adni_flwr.utils import AppUtils


class OpacusDPClientFactory(ClientAppFactoryMixin, DifferentialPrivacyMixin):
    """Factory for creating Opacus-based differential privacy client apps."""

    @staticmethod
    def create_opacus_dp_app(config: Config = None) -> ClientApp:
        """Create a ClientApp using Opacus-based gradient-level DP instead of LocalDpMod.

        This approach is superior to LocalDpMod because:
        1. Applies noise to gradients during training (DP-SGD)
        2. Noise is relative to gradient magnitudes, not absolute parameter values
        3. Better convergence as parameters get smaller during training

        Args:
            config: Optional pre-loaded config. If None, will be loaded from context.

        Returns:
            ClientApp with Opacus DP strategy

        Raises:
            ValueError: If config is invalid or strategy is not differential_privacy
            FileNotFoundError: If config file cannot be found
            RuntimeError: If differential privacy client creation fails
        """

        def opacus_dp_client_fn(context: Context):
            """Client factory function for Opacus DP."""
            try:
                # Create common client components
                device, partition_id, loaded_config, client_id = (
                    OpacusDPClientFactory._create_client_context_components(context, config)
                )

                # Validate DP configuration
                OpacusDPClientFactory._validate_dp_config(loaded_config, f"client partition {partition_id} config")

                # Load model and data components
                model, train_loader, val_loader = OpacusDPClientFactory._create_model_data_components(
                    loaded_config, device
                )

                # Create optimizer and scheduler
                optimizer, scheduler = OpacusDPClientFactory._create_optimizer_scheduler(
                    model, loaded_config, train_loader
                )

                # Create criterion
                criterion = create_criterion(loaded_config, train_loader.dataset, device)

                # Extract and validate DP parameters
                epsilon, sensitivity, clipping_norm, delta = AppUtils.validate_dp_parameters_for_opacus(loaded_config)

                # Convert epsilon to noise multiplier for Opacus
                noise_multiplier = sensitivity / epsilon

                # Create Opacus differential privacy client strategy
                client_strategy = OpacusDPClient(
                    config=loaded_config,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    scheduler=scheduler,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=clipping_norm,
                    target_epsilon=epsilon,
                    target_delta=delta,
                )

                # Create strategy-aware client
                client = StrategyAwareClient(
                    config=loaded_config,
                    device=device,
                    client_strategy=client_strategy,
                    context=context,
                    total_fl_rounds=loaded_config.fl.num_rounds,
                    wandb_logger=None,  # Add wandb_logger if needed
                )

                return client.to_client()

            except Exception as e:
                raise RuntimeError(f"Failed to create Opacus differential privacy client: {e}") from e

        return ClientApp(client_fn=opacus_dp_client_fn)

    @staticmethod
    def _create_model_data_components(config: Config, device: torch.device):
        """Create model and data components with error handling.

        Args:
            config: Configuration object
            device: PyTorch device

        Returns:
            Tuple of (model, train_loader, val_loader)

        Raises:
            RuntimeError: If component creation fails
        """
        try:
            model = load_model(config).to(device)
            train_loader, val_loader = load_data(config)
            logger.info("✅ Model and data components created for Opacus DP client")
            return model, train_loader, val_loader
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model/data components: {e}") from e

    @staticmethod
    def _create_optimizer_scheduler(model: torch.nn.Module, config: Config, train_loader):
        """Create optimizer and scheduler with error handling.

        Args:
            model: PyTorch model
            config: Configuration object
            train_loader: Training data loader

        Returns:
            Tuple of (optimizer, scheduler)

        Raises:
            RuntimeError: If optimizer creation fails
        """
        try:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )

            # Create scheduler if specified
            scheduler = AppUtils.create_scheduler(optimizer, config, train_loader, config.fl.num_rounds)

            logger.info("✅ Optimizer and scheduler created for Opacus DP client")
            return optimizer, scheduler

        except Exception as e:
            raise RuntimeError(f"Failed to create optimizer/scheduler: {e}") from e
