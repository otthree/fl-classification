"""Experimental | Differential Privacy (DP) strategy implementation for Federated Learning using Opacus."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common import EvaluateIns, FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from loguru import logger
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    logger.info("Successfully imported Opacus PrivacyEngine and BatchMemoryManager")
except ImportError as e:
    logger.error(f"Failed to import Opacus: {e}")
    raise ImportError("Opacus is required for differential privacy. Please install it with: pip install opacus") from e

from adni_classification.config.config import Config
from adni_flwr.task import get_params, safe_parameters_to_ndarrays, set_params

from .base import ClientStrategyBase, FLStrategyBase


class DifferentialPrivacyStrategy(FLStrategyBase):
    """Server-side Differential Privacy strategy for Federated Learning using Opacus.

    This strategy uses the standard FedAvg aggregation method since differential privacy
    is applied on the client side during training using Opacus PrivacyEngine.
    Privacy protection comes from DP-SGD during client training, not from server aggregation.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        target_epsilon: float = 8.0,
        target_delta: Optional[float] = None,
        **kwargs,
    ):
        """Initialize Differential Privacy strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
            target_epsilon: Target epsilon for privacy budget
            target_delta: Target delta for privacy budget (if None, computed automatically)
            **kwargs: Additional parameters passed to FedAvg
        """
        super().__init__(config, model, wandb_logger, **kwargs)

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # Extract specific parameters for FedAvg
        fedavg_params = {
            "fraction_fit": getattr(config.fl, "fraction_fit", 1.0),
            "fraction_evaluate": getattr(config.fl, "fraction_evaluate", 1.0),
            "min_fit_clients": getattr(config.fl, "min_fit_clients", 2),
            "min_evaluate_clients": getattr(config.fl, "min_evaluate_clients", 2),
            "min_available_clients": getattr(config.fl, "min_available_clients", 2),
        }

        # Add aggregation functions if provided
        if "evaluate_metrics_aggregation_fn" in kwargs:
            fedavg_params["evaluate_metrics_aggregation_fn"] = kwargs["evaluate_metrics_aggregation_fn"]
        if "fit_metrics_aggregation_fn" in kwargs:
            fedavg_params["fit_metrics_aggregation_fn"] = kwargs["fit_metrics_aggregation_fn"]

        # Use standard FedAvg strategy for aggregation
        self.fedavg_strategy = FedAvg(**fedavg_params)

        self.current_round = 0

        # Load server-side validation dataset for global evaluation
        self._load_server_validation_data()

        logger.info(f"DP Strategy initialized with noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "differential_privacy"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
        }

    def get_strategy_specific_metrics(self) -> Dict[str, Any]:
        """Return strategy-specific metrics for Differential Privacy."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "target_epsilon": self.target_epsilon,
        }

    # Delegate most methods to FedAvg since DP is handled client-side
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        from flwr.common import ndarrays_to_parameters

        logger.info("DP Strategy: Initializing parameters from server model")
        ndarrays = get_params(self.model)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with DP parameters."""
        # Get the base configuration from FedAvg
        client_instructions = self.fedavg_strategy.configure_fit(server_round, parameters, client_manager)

        # Add DP parameters to the config for each client
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            config = fit_ins.config.copy() if fit_ins.config else {}
            config.update(
                {
                    "server_round": server_round,
                    "noise_multiplier": self.noise_multiplier,
                    "max_grad_norm": self.max_grad_norm,
                    "target_epsilon": self.target_epsilon,
                    "target_delta": self.target_delta,
                }
            )

            updated_fit_ins = FitIns(parameters=fit_ins.parameters, config=config)
            updated_instructions.append((client_proxy, updated_fit_ins))

        # Add WandB run ID to instructions
        return self.add_wandb_config_to_instructions(updated_instructions)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        client_instructions = self.fedavg_strategy.configure_evaluate(server_round, parameters, client_manager)

        # Add server_round to the config for each client
        updated_instructions = []
        for client_proxy, evaluate_ins in client_instructions:
            config = evaluate_ins.config.copy() if evaluate_ins.config else {}
            config["server_round"] = server_round

            updated_evaluate_ins = EvaluateIns(parameters=evaluate_ins.parameters, config=config)
            updated_instructions.append((client_proxy, updated_evaluate_ins))

        return self.add_wandb_config_to_instructions(updated_instructions)

    def aggregate_fit(self, server_round: int, results, failures) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using standard FedAvg."""
        self.current_round = server_round

        # Log client training metrics including privacy budget
        if self.wandb_logger:
            for client_proxy, fit_res in results:
                client_metrics = fit_res.metrics
                if client_metrics:
                    client_id = client_metrics.get("client_id", f"client_{client_proxy.cid}")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(metrics_to_log, prefix=f"client_{client_id}/fit", step=server_round)

        # Use standard FedAvg aggregation
        aggregated_parameters, metrics = self.fedavg_strategy.aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Update server model with aggregated parameters
            param_arrays = safe_parameters_to_ndarrays(aggregated_parameters)
            set_params(self.model, param_arrays)

            # Add DP-specific metrics
            metrics.update(
                {
                    "dp_noise_multiplier": self.noise_multiplier,
                    "dp_max_grad_norm": self.max_grad_norm,
                    "dp_target_epsilon": self.target_epsilon,
                }
            )

        # Log server metrics
        if self.wandb_logger and metrics:
            self.wandb_logger.log_metrics(metrics, prefix="server", step=server_round)

        # Save checkpoint if configured
        if (
            self.config.training.checkpoint.save_regular
            and server_round % self.config.training.checkpoint.save_frequency == 0
        ):
            self._save_checkpoint(self.model.state_dict(), server_round)

        return aggregated_parameters, metrics

    # Delegate other methods to FedAvg
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results using standard FedAvg with error handling."""
        if not results:
            logger.warning(f"No evaluation results received in round {server_round}")
            return None, {}

        # Check if any results have examples
        total_examples = sum([res.num_examples for _, res in results if res.num_examples > 0])
        if total_examples == 0:
            logger.warning(f"No evaluation examples found in round {server_round}")
            return None, {}

        try:
            return self.fedavg_strategy.aggregate_evaluate(server_round, results, failures)
        except ZeroDivisionError as e:
            logger.error(f"Division by zero in evaluation aggregation at round {server_round}: {e}")
            return None, {}
        except Exception as e:
            logger.error(f"Error in evaluation aggregation at round {server_round}: {e}")
            return None, {}

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters using standard FedAvg."""
        return self.fedavg_strategy.evaluate(server_round, parameters)

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying FedAvg strategy."""
        return getattr(self.fedavg_strategy, name)


class DifferentialPrivacyClient(ClientStrategyBase):
    """Client-side Differential Privacy strategy using Opacus PrivacyEngine.

    This implementation follows the official Opacus approach:
    1. Sets up PrivacyEngine.make_private() once per FL round
    2. Uses the DP-enabled components for training
    3. Tracks privacy budget using RDP analysis
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        target_epsilon: float = 8.0,
        target_delta: Optional[float] = None,
        **kwargs,
    ):
        """Initialize Differential Privacy client strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use for computation
            scheduler: Learning rate scheduler (optional)
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
            target_epsilon: Target epsilon for privacy budget
            target_delta: Target delta for privacy budget
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, model, optimizer, criterion, device, scheduler, **kwargs)

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # Client ID validation
        if not hasattr(config.fl, "client_id") or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section."
            )
        self.client_id = config.fl.client_id

        # Privacy engine - initialize but don't set up until training
        self.privacy_engine = None
        self.dp_model = None
        self.dp_optimizer = None
        self.dp_train_loader = None

        # Store original components for restoration
        self.original_model = model
        self.original_optimizer = optimizer

        # Privacy tracking
        self.privacy_spent = {"epsilon": 0.0, "delta": 0.0}
        self.current_round = 0

        # Track if privacy engine was set up this round
        self._privacy_engine_setup_this_round = False

        logger.info(f"DP Client initialized: noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}")

        # Add GPU monitoring info
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"Device: {device}")
            logger.info(f"GPU memory before DP setup: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "differential_privacy"

    def _compute_target_delta(self, dataset_size: int) -> float:
        """Compute target delta if not provided."""
        if self.target_delta is not None:
            return self.target_delta
        # Standard practice: delta should be much smaller than 1/dataset_size
        return 1.0 / (dataset_size * 10)

    def _cleanup_privacy_engine(self):
        """Clean up privacy engine and restore original components."""
        if self.privacy_engine is not None:
            logger.info(f"Client {self.client_id}: Cleaning up privacy engine...")

            # Clear CUDA cache to free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"Client {self.client_id}: Cleared CUDA cache")

            # Reset to original components
            self.model = self.original_model
            self.optimizer = self.original_optimizer

            # Clear DP components
            self.privacy_engine = None
            self.dp_model = None
            self.dp_optimizer = None
            self.dp_train_loader = None

            self._privacy_engine_setup_this_round = False

            logger.info(f"Client {self.client_id}: Privacy engine cleanup completed")

    def _create_opacus_compatible_dataloader(self, train_loader: DataLoader) -> DataLoader:
        """Create an Opacus-compatible dataloader from MONAI dataloader.

        Opacus requires specific dataloader properties that MONAI dataloaders might not satisfy.
        """
        try:
            logger.info(f"Client {self.client_id}: Creating Opacus-compatible dataloader...")

            # Get the dataset from the original loader
            dataset = train_loader.dataset

            # Custom collate function to convert MONAI MetaTensors to regular PyTorch tensors
            def opacus_safe_collate(batch):
                """Custom collate function that converts MONAI MetaTensors to regular PyTorch tensors."""
                import torch
                from torch.utils.data.dataloader import default_collate

                # First, convert any MetaTensors to regular tensors
                converted_batch = []
                for item in batch:
                    converted_item = {}
                    for key, value in item.items():
                        if hasattr(value, "as_tensor"):  # MONAI MetaTensor has as_tensor() method
                            # Convert MetaTensor to regular PyTorch tensor
                            converted_item[key] = value.as_tensor().clone().detach()
                        elif isinstance(value, torch.Tensor):
                            # Regular tensor - ensure it's detached and contiguous
                            converted_item[key] = value.clone().detach().contiguous()
                        else:
                            # Non-tensor data (like metadata)
                            converted_item[key] = value
                    converted_batch.append(converted_item)

                try:
                    # Use default collate with converted tensors
                    collated = default_collate(converted_batch)

                    # Ensure proper dtypes for key tensors
                    if isinstance(collated, dict):
                        for key, value in collated.items():
                            if isinstance(value, torch.Tensor):
                                # Ensure proper dtype - convert to float32 for images, long for labels
                                if key == "image" and value.dtype not in [torch.float32, torch.float16]:
                                    collated[key] = value.float()
                                elif key == "label" and value.dtype not in [torch.long, torch.int64]:
                                    collated[key] = value.long()
                                # Ensure tensor is contiguous
                                collated[key] = collated[key].contiguous()

                    return collated

                except Exception as e:
                    logger.error(f"Client {self.client_id}: Error in default collate: {e}")
                    # Fallback to manual collation with converted tensors
                    if len(converted_batch) == 0:
                        return {}

                    # Manual safe collation
                    result = {}
                    for key in converted_batch[0].keys():
                        try:
                            values = [item[key] for item in converted_batch]
                            if key == "image":
                                # Stack image tensors with float32 dtype
                                tensors = [v.float().contiguous() if isinstance(v, torch.Tensor) else v for v in values]
                                if all(isinstance(t, torch.Tensor) for t in tensors):
                                    result[key] = torch.stack(tensors)
                                else:
                                    result[key] = values
                            elif key == "label":
                                # Stack label tensors with long dtype
                                tensors = [v.long().contiguous() if isinstance(v, torch.Tensor) else v for v in values]
                                if all(isinstance(t, torch.Tensor) for t in tensors):
                                    result[key] = torch.stack(tensors)
                                else:
                                    result[key] = values
                            else:
                                # Try to stack if all are tensors, otherwise keep as list
                                if all(isinstance(v, torch.Tensor) for v in values):
                                    result[key] = torch.stack(values)
                                else:
                                    result[key] = values
                        except Exception as stack_e:
                            logger.error(f"Client {self.client_id}: Error stacking {key}: {stack_e}")
                            # If stacking fails, keep as list
                            result[key] = [item[key] for item in converted_batch]

                    return result

            # Create a new DataLoader with Opacus-compatible settings
            compatible_loader = DataLoader(
                dataset=dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,  # Opacus works better with shuffled data
                num_workers=0,  # Opacus can have issues with multiprocessing
                pin_memory=False,  # Disable pin_memory for Opacus compatibility
                drop_last=True,  # Opacus requires consistent batch sizes
                persistent_workers=False,  # Disable persistent workers
                multiprocessing_context=None,  # No multiprocessing context
                prefetch_factor=None,  # No prefetching
                collate_fn=opacus_safe_collate,  # Use our custom collate function
            )

            logger.info(f"Client {self.client_id}: Created compatible dataloader with {len(compatible_loader)} batches")
            return compatible_loader

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error creating compatible dataloader: {str(e)}")
            # Fall back to original loader
            logger.warning(f"Client {self.client_id}: Falling back to original dataloader")
            return train_loader

    def prepare_for_round(self, server_params: Parameters, round_config: Dict[str, Any]):
        """Prepare the client for a new training round with DP setup."""
        try:
            logger.info(f"Client {self.client_id}: Starting prepare_for_round...")

            # Clean up any existing privacy engine first
            self._cleanup_privacy_engine()

            # Ensure model is on correct device
            self.model.to(self.device)
            self.original_model = self.model

            # Convert parameters and update model
            logger.info(f"Client {self.client_id}: Converting server parameters...")
            param_arrays = safe_parameters_to_ndarrays(server_params)
            logger.info(f"Client {self.client_id}: Setting model parameters...")
            set_params(self.model, param_arrays)
            logger.info(f"Client {self.client_id}: Model parameters updated successfully")

            # Update round number and DP parameters from server
            self.current_round = round_config.get("server_round", self.current_round + 1)
            logger.info(f"Client {self.client_id}: Updated round number to {self.current_round}")

            # Update DP parameters if provided by server
            self.noise_multiplier = round_config.get("noise_multiplier", self.noise_multiplier)
            self.max_grad_norm = round_config.get("max_grad_norm", self.max_grad_norm)
            self.target_epsilon = round_config.get("target_epsilon", self.target_epsilon)
            self.target_delta = round_config.get("target_delta", self.target_delta)
            logger.info(
                f"Client {self.client_id}: DP parameters updated - noise_multiplier={self.noise_multiplier}, "
                f"max_grad_norm={self.max_grad_norm}"
            )

            # Reset optimizer state
            logger.info(f"Client {self.client_id}: Resetting optimizer state...")
            self.optimizer.zero_grad()

            logger.info(f"Client {self.client_id} prepared for FL round {self.current_round}")

            # GPU monitoring
            if torch.cuda.is_available():
                logger.info(
                    f"Client {self.client_id}: GPU memory after prepare_for_round: "
                    f"{torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB"
                )

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error in prepare_for_round: {str(e)}")
            logger.error(f"Client {self.client_id}: Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Client {self.client_id}: Traceback: {traceback.format_exc()}")
            raise e

    def _setup_privacy_engine(self, train_loader: DataLoader):
        """Set up privacy engine once per FL round following official Opacus pattern."""
        try:
            logger.info(f"Client {self.client_id}: Starting _setup_privacy_engine...")

            if self._privacy_engine_setup_this_round:
                logger.info(f"Client {self.client_id}: Privacy engine already set up for this round, skipping")
                return

            # Check if train_loader has data
            dataset_size = len(train_loader.dataset)
            logger.info(f"Client {self.client_id}: Checking dataset size: {dataset_size}")

            if dataset_size == 0:
                logger.error(f"Client {self.client_id}: Train dataloader is empty! Dataset size: {dataset_size}")
                raise ValueError("Cannot set up privacy engine with empty training dataset")

            # Create Opacus-compatible dataloader
            logger.info(f"Client {self.client_id}: Creating Opacus-compatible dataloader...")
            compatible_loader = self._create_opacus_compatible_dataloader(train_loader)

            # Compute target delta
            target_delta = self._compute_target_delta(dataset_size)
            logger.info(f"Client {self.client_id}: Computed target delta: {target_delta}")

            # Ensure model is on correct device
            self.model = self.model.to(self.device)
            device_before = next(self.model.parameters()).device
            logger.info(f"Client {self.client_id}: Model device before make_private: {device_before}")

            # CRITICAL: Ensure model is in training mode for Opacus
            self.model.train()
            logger.info(f"Client {self.client_id}: Model set to training mode for Opacus")

            # GPU monitoring before setup
            if torch.cuda.is_available():
                logger.info(
                    f"Client {self.client_id}: GPU memory before privacy engine setup: "
                    f"{torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB"
                )

            # CRITICAL: Fix model to replace incompatible modules (e.g., BatchNorm3d -> GroupNorm)
            logger.info(f"Client {self.client_id}: Fixing model for DP compatibility...")
            from opacus.validators import ModuleValidator

            # Check if model needs fixing
            errors = ModuleValidator.validate(self.model, strict=False)
            if errors:
                logger.info(
                    f"Client {self.client_id}: Found {len(errors)} incompatible modules, fixing automatically..."
                )
                logger.info(f"Client {self.client_id}: Errors: {[str(e) for e in errors]}")

                # Fix the model by replacing incompatible modules
                self.model = ModuleValidator.fix(self.model)
                logger.info(f"Client {self.client_id}: Model fixed successfully for DP compatibility")

                # Ensure fixed model is on correct device
                self.model = self.model.to(self.device)
                logger.info(f"Client {self.client_id}: Fixed model moved back to device: {self.device}")

                # CRITICAL: Recreate optimizer with new model parameters after fixing
                logger.info(f"Client {self.client_id}: Recreating optimizer for fixed model...")
                optimizer_class = type(self.optimizer)
                optimizer_state = self.optimizer.state_dict()

                # Create new optimizer with same parameters but for the fixed model
                self.optimizer = optimizer_class(
                    self.model.parameters(),
                    **{k: v for k, v in self.optimizer.param_groups[0].items() if k != "params"},
                )

                # Try to restore optimizer state if possible (might fail due to parameter changes)
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                    logger.info(f"Client {self.client_id}: Optimizer state restored successfully")
                except Exception as opt_e:
                    logger.warning(f"Client {self.client_id}: Could not restore optimizer state: {opt_e}")
                    logger.info(f"Client {self.client_id}: Continuing with fresh optimizer state")

            else:
                logger.info(f"Client {self.client_id}: Model is already DP-compatible, no fixes needed")

            # Initialize privacy engine with secure_mode=False for better compatibility
            logger.info(f"Client {self.client_id}: Creating PrivacyEngine with secure_mode=False...")
            self.privacy_engine = PrivacyEngine(secure_mode=False)

            # Make model, optimizer, and data loader private
            logger.info(f"Client {self.client_id}: Calling privacy_engine.make_private...")
            self.dp_model, self.dp_optimizer, self.dp_train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=compatible_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            logger.info(f"Client {self.client_id}: make_private completed successfully")

            # CRITICAL: Ensure model stays on correct device after make_private
            self.dp_model = self.dp_model.to(self.device)
            device_after = next(self.dp_model.parameters()).device
            logger.info(f"Client {self.client_id}: Model device after make_private and device fix: {device_after}")

            if device_after != self.device:
                logger.error(f"Client {self.client_id}: Device mismatch! Expected {self.device}, got {device_after}")
                raise RuntimeError(f"Failed to keep model on device {self.device} after Opacus make_private")

            # Update model reference for training
            self.model = self.dp_model
            self.optimizer = self.dp_optimizer

            # Mark as set up for this round
            self._privacy_engine_setup_this_round = True

            # GPU monitoring after setup
            if torch.cuda.is_available():
                logger.info(
                    f"Client {self.client_id}: GPU memory after privacy engine setup: "
                    f"{torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB"
                )

            logger.info(f"Client {self.client_id}: Privacy engine set up successfully:")
            logger.info(f"  - noise_multiplier: {self.noise_multiplier}")
            logger.info(f"  - max_grad_norm: {self.max_grad_norm}")
            logger.info(f"  - DP dataloader size: {len(self.dp_train_loader)}")
            logger.info(f"  - DP dataloader batch size: {self.dp_train_loader.batch_size}")
            logger.info(f"  - Model device: {next(self.model.parameters()).device}")

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error in _setup_privacy_engine: {str(e)}")
            logger.error(f"Client {self.client_id}: Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Client {self.client_id}: Traceback: {traceback.format_exc()}")
            raise e

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int, **kwargs) -> Tuple[float, float]:
        """Train the model for one epoch using Differential Privacy with Opacus."""
        try:
            logger.info(f"DP Client {self.client_id}: Starting train_epoch {epoch + 1}/{total_epochs}")
            logger.info(f"Train dataloader: {len(train_loader)} batches, dataset size: {len(train_loader.dataset)}")

            # Set up privacy engine once per FL round
            logger.info(f"DP Client {self.client_id}: Setting up privacy engine...")
            self._setup_privacy_engine(train_loader)
            logger.info(f"DP Client {self.client_id}: Privacy engine setup completed")

            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Use the DP-enabled dataloader with BatchMemoryManager for better GPU utilization
            logger.info(f"Starting training loop with DP dataloader: {len(self.dp_train_loader)} batches")

            # Use BatchMemoryManager for efficient memory usage during DP training
            with BatchMemoryManager(
                data_loader=self.dp_train_loader,
                max_physical_batch_size=self.config.training.batch_size,
                optimizer=self.dp_optimizer,
            ) as memory_safe_data_loader:
                for batch_idx, batch in enumerate(memory_safe_data_loader):
                    try:
                        # Debug first batch
                        if batch_idx == 0:
                            logger.info(f"Processing first batch: batch_idx={batch_idx}")
                            if torch.cuda.is_available():
                                logger.info(
                                    f"GPU memory during first batch: "
                                    f"{torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB"
                                )

                        # Ensure tensors are on correct device
                        images = batch["image"].to(self.device, non_blocking=True)
                        labels = batch["label"].to(self.device, non_blocking=True)

                        # Debug batch info
                        if batch_idx == 0:
                            logger.info(f"Batch shapes - images: {images.shape}, labels: {labels.shape}")
                            logger.info(f"Images device: {images.device}, Labels device: {labels.device}")
                            logger.info(f"Model device: {next(self.model.parameters()).device}")

                        # Zero gradients
                        self.optimizer.zero_grad()

                        # Forward pass
                        if self.config.training.mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(images)
                                loss = self.criterion(outputs, labels)
                        else:
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)

                        # Debug first batch forward pass
                        if batch_idx == 0:
                            logger.info(
                                f"Forward pass complete - outputs shape: {outputs.shape}, loss: {loss.item():.4f}"
                            )

                        # Backward pass (Opacus handles DP automatically)
                        loss.backward()

                        # Optimizer step (includes DP noise and clipping)
                        self.optimizer.step()

                        # Track metrics
                        total_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()

                        # Log progress every 10 batches
                        if (batch_idx + 1) % 10 == 0:
                            logger.info(f"Batch {batch_idx + 1}/{len(memory_safe_data_loader)}: loss={loss.item():.4f}")
                            if torch.cuda.is_available():
                                logger.info(f"GPU memory: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")

                    except Exception as batch_e:
                        logger.error(f"DP Client {self.client_id}: Error in batch {batch_idx}: {str(batch_e)}")
                        logger.error(f"DP Client {self.client_id}: Batch exception type: {type(batch_e).__name__}")
                        import traceback

                        logger.error(f"DP Client {self.client_id}: Batch traceback: {traceback.format_exc()}")
                        raise batch_e

            logger.info(
                f"Training epoch complete: processed {total_samples} samples in {len(self.dp_train_loader)} batches"
            )

            # Get privacy budget spent
            if self.privacy_engine is not None:
                try:
                    epsilon = self.privacy_engine.get_epsilon(
                        delta=self._compute_target_delta(len(train_loader.dataset))
                    )
                    self.privacy_spent = {
                        "epsilon": epsilon,
                        "delta": self._compute_target_delta(len(train_loader.dataset)),
                    }

                    logger.info(f"Privacy budget spent: ε={epsilon:.3f}, δ={self.privacy_spent['delta']:.2e}")
                except Exception as privacy_e:
                    logger.error(f"DP Client {self.client_id}: Error getting privacy budget: {str(privacy_e)}")
                    # Continue without privacy budget info

            # Compute metrics
            avg_loss = total_loss / len(self.dp_train_loader) if len(self.dp_train_loader) > 0 else 0.0
            avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

            logger.info(f"Epoch {epoch + 1} results: avg_loss={avg_loss:.4f}, avg_accuracy={avg_accuracy:.2f}%")

            # Step scheduler if needed (only on last local epoch)
            if self.scheduler is not None and epoch == total_epochs - 1:
                try:
                    current_lr_before = self.optimizer.param_groups[0]["lr"]

                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(avg_loss)
                    else:
                        self.scheduler.step()

                    current_lr_after = self.optimizer.param_groups[0]["lr"]
                    if current_lr_before != current_lr_after:
                        logger.info(
                            f"FL Round {self.current_round}: "
                            f"LR changed from {current_lr_before:.8f} to {current_lr_after:.8f}"
                        )
                except Exception as scheduler_e:
                    logger.error(f"DP Client {self.client_id}: Error stepping scheduler: {str(scheduler_e)}")
                    # Continue without scheduler step

            # Final GPU monitoring
            if torch.cuda.is_available():
                logger.info(
                    f"DP Client {self.client_id}: GPU memory after epoch: "
                    f"{torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB"
                )

            logger.info(f"DP Client {self.client_id}: train_epoch completed successfully")
            return avg_loss, avg_accuracy

        except Exception as e:
            logger.error(f"DP Client {self.client_id}: Error in train_epoch: {str(e)}")
            logger.error(f"DP Client {self.client_id}: Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"DP Client {self.client_id}: Traceback: {traceback.format_exc()}")
            raise e

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom Differential Privacy-specific metrics."""
        try:
            metrics = {
                "dp_noise_multiplier": self.noise_multiplier,
                "dp_max_grad_norm": self.max_grad_norm,
                "dp_target_epsilon": self.target_epsilon,
                "dp_epsilon_spent": self.privacy_spent.get("epsilon", 0.0),
                "dp_delta_spent": self.privacy_spent.get("delta", 0.0),
                "dp_budget_exhausted": self.privacy_spent.get("epsilon", 0.0) > self.target_epsilon,
            }

            # Add GPU utilization info if available
            if torch.cuda.is_available():
                metrics.update(
                    {
                        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
                        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
                        "gpu_utilization_active": torch.cuda.is_available()
                        and torch.cuda.memory_allocated(self.device) > 0,
                    }
                )

            logger.info(f"DP Client {self.client_id}: Generated custom metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"DP Client {self.client_id}: Error in get_custom_metrics: {str(e)}")
            # Return minimal metrics to avoid breaking the flow
            return {
                "dp_noise_multiplier": self.noise_multiplier,
                "dp_max_grad_norm": self.max_grad_norm,
                "dp_target_epsilon": self.target_epsilon,
            }

    def get_parameters(self):
        """Get model parameters in the format expected by Flower."""
        try:
            from adni_flwr.task import get_params

            return get_params(self.model)
        except Exception as e:
            logger.error(f"DP Client {self.client_id}: Error getting parameters: {str(e)}")
            # Fall back to getting parameters from original model
            return get_params(self.original_model)

    def get_differential_private_parameters(self):
        """Get DP-protected model parameters for FL aggregation.

        This method is called by StrategyAwareClient.fit() to extract parameters
        after DP training. It returns the parameters from the DP-enabled model.

        Returns:
            Model parameters as numpy arrays
        """
        try:
            logger.info(f"DP Client {self.client_id}: Extracting DP-protected parameters for FL aggregation")

            from adni_flwr.task import get_params

            # Use the DP model if available, otherwise fall back to original model
            if self.dp_model is not None:
                logger.info(f"DP Client {self.client_id}: Using DP model for parameter extraction")
                params = get_params(self.dp_model)
            else:
                logger.warning(f"DP Client {self.client_id}: DP model not available, using original model")
                params = get_params(self.model)

            logger.info(f"DP Client {self.client_id}: Successfully extracted {len(params)} parameter arrays")
            return params

        except Exception as e:
            logger.error(f"DP Client {self.client_id}: Error extracting DP parameters: {str(e)}")
            logger.error(f"DP Client {self.client_id}: Exception type: {type(e).__name__}")

            # Fall back to original model parameters
            try:
                logger.warning(f"DP Client {self.client_id}: Falling back to original model parameters")
                return get_params(self.original_model)
            except Exception as fallback_e:
                logger.error(f"DP Client {self.client_id}: Fallback parameter extraction failed: {str(fallback_e)}")
                raise fallback_e
