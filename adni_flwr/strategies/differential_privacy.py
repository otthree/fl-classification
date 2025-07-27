"""Differential Privacy (DP) strategy implementation for Federated Learning using Flower's LocalDpMod."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common import EvaluateIns, FitIns, FitRes, Parameters, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from loguru import logger
from torch.utils.data import DataLoader

from adni_classification.config.config import Config
from adni_flwr.task import get_params, safe_parameters_to_ndarrays, set_params

from .base import ClientStrategyBase, FLStrategyBase


class DifferentialPrivacyStrategy(FLStrategyBase):
    """Server-side Differential Privacy strategy for Federated Learning with comprehensive WandB logging.

    This strategy implements differential privacy in federated learning by working with clients
    that use Flower's LocalDpMod for standardized local differential privacy.

    Note: This server strategy works with clients using LocalDpMod for privacy protection.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize Differential Privacy strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            **kwargs: Additional DP parameters
        """
        super().__init__(config, model, wandb_logger, **kwargs)

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

        # Initialize standard FedAvg strategy for basic aggregation
        self.fedavg_strategy = FedAvg(**fedavg_params)

        # Initialize additional components for enhanced functionality
        self.current_round = 0

        # Load server-side validation dataset for global evaluation
        self._load_server_validation_data()

        logger.info("DifferentialPrivacyStrategy initialized with LocalDpMod support")
        logger.info("DP parameters are configured via LocalDpMod in client_app.py")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "differential_privacy_localdpmod"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        return {
            "uses_localdpmod": True,
            "fraction_fit": self.fedavg_strategy.fraction_fit,
            "fraction_evaluate": self.fedavg_strategy.fraction_evaluate,
            "min_fit_clients": self.fedavg_strategy.min_fit_clients,
            "min_evaluate_clients": self.fedavg_strategy.min_evaluate_clients,
            "min_available_clients": self.fedavg_strategy.min_available_clients,
        }

    def get_strategy_specific_metrics(self) -> Dict[str, Any]:
        """Return strategy-specific metrics for Differential Privacy.

        Returns:
            Dictionary of DP specific metrics for logging.
        """
        return {
            "uses_localdpmod": True,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results from clients using LocalDpMod.

        Since LocalDpMod handles differential privacy on the client side,
        the server can perform standard aggregation.
        """
        self.current_round = server_round

        # Log client training metrics
        if self.wandb_logger:
            for _client_proxy, fit_res in results:
                client_metrics = fit_res.metrics
                if client_metrics:
                    client_id = fit_res.metrics.get("client_id", "unknown")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(metrics_to_log, prefix=f"client_{client_id}/fit", step=server_round)

        # Use standard FedAvg aggregation since LocalDpMod handles privacy on client side
        aggregated_parameters, metrics = self.fedavg_strategy.aggregate_fit(server_round, results, failures)

        if aggregated_parameters is None:
            return None, {}

        # Update server model with aggregated parameters
        param_arrays = safe_parameters_to_ndarrays(aggregated_parameters)
        set_params(self.model, param_arrays)

        # Add DP-specific information to metrics
        dp_metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "uses_localdpmod": True,
        }
        metrics.update(dp_metrics)

        # Log aggregated fit metrics with DP-specific information
        if self.wandb_logger and metrics:
            self.wandb_logger.log_metrics(metrics, prefix="server", step=server_round)

        # Print server model's current metrics
        print(f"Server model metrics after round {server_round} (using LocalDpMod for DP):")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")

        # Save frequency checkpoint
        if (
            self.config.training.checkpoint.save_regular
            and server_round % self.config.training.checkpoint.save_frequency == 0
        ):
            self._save_checkpoint(self.model.state_dict(), server_round)

        return aggregated_parameters, metrics

    # Implement required Strategy abstract methods
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        print("DifferentialPrivacyStrategy: Initializing parameters from server model")
        ndarrays = get_params(self.model)
        print(f"DifferentialPrivacyStrategy: Sending {len(ndarrays)} parameter arrays to clients")
        print(f"DifferentialPrivacyStrategy: First few parameter shapes: {[arr.shape for arr in ndarrays[:5]]}")

        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get the base configuration from the parent class
        client_instructions = self.fedavg_strategy.configure_fit(server_round, parameters, client_manager)

        # Add server_round and DP configuration to the config for each client
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            # Add server_round and DP information to the existing config
            config = fit_ins.config.copy() if fit_ins.config else {}
            config["server_round"] = server_round
            config["uses_localdpmod"] = True

            # Create new FitIns with updated config
            updated_fit_ins = FitIns(parameters=fit_ins.parameters, config=config)
            updated_instructions.append((client_proxy, updated_fit_ins))

        # Add WandB run ID to instructions
        return self.add_wandb_config_to_instructions(updated_instructions)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Get the base configuration from the parent class
        client_instructions = self.fedavg_strategy.configure_evaluate(server_round, parameters, client_manager)

        # Add server_round to the config for each client
        updated_instructions = []
        for client_proxy, evaluate_ins in client_instructions:
            # Add server_round to the existing config
            config = evaluate_ins.config.copy() if evaluate_ins.config else {}
            config["server_round"] = server_round

            # Create new EvaluateIns with updated config
            updated_evaluate_ins = EvaluateIns(parameters=evaluate_ins.parameters, config=config)
            updated_instructions.append((client_proxy, updated_evaluate_ins))

        # Add WandB run ID to instructions
        return self.add_wandb_config_to_instructions(updated_instructions)

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters."""
        return self.fedavg_strategy.evaluate(server_round, parameters)

    # Delegate other attributes to FedAvg
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying FedAvg strategy."""
        return getattr(self.fedavg_strategy, name)


class DifferentialPrivacyClient(ClientStrategyBase):
    """Client-side Differential Privacy strategy for Federated Learning using Flower's LocalDpMod.

    This client strategy now uses Flower's official LocalDpMod for standardized local differential privacy.
    The LocalDpMod should be added to the ClientApp's mods list.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
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
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, model, optimizer, criterion, device, scheduler, **kwargs)

        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(config.fl, "client_id") or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = config.fl.client_id
        self.current_round = 0

        # Training parameters
        self.mixed_precision = config.training.mixed_precision
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        logger.info("DP Client initialized with LocalDpMod support")
        logger.info("DP parameters are configured via LocalDpMod in client_app.py")
        logger.info("Note: LocalDpMod should be added to ClientApp's mods list for DP functionality")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "differential_privacy_localdpmod"

    def prepare_for_round(self, server_params: Parameters, round_config: Dict[str, Any]):
        """Prepare the client for a new training round.

        Args:
            server_params: Parameters from server
            round_config: Configuration for this round
        """
        # Convert parameters to numpy arrays safely
        param_arrays = safe_parameters_to_ndarrays(server_params)

        # Update model with server parameters
        set_params(self.model, param_arrays)

        # Update round number
        self.current_round = round_config.get("server_round", self.current_round + 1)

        # Store current FL round information
        self.current_fl_round = round_config.get("server_round", 1)

        # Reset optimizer state
        self.optimizer.zero_grad()

        logger.info(f"Client prepared for round {self.current_round} with LocalDpMod DP parameters")

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int, **kwargs) -> Tuple[float, float]:
        """Train the model for one epoch with differential privacy via LocalDpMod.

        Note: LocalDpMod handles all differential privacy operations including gradient clipping and noise addition.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            **kwargs: Additional training parameters

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Mixed precision training
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Unscale gradients before optimizer step
                    self.scaler.unscale_(self.optimizer)

                    # Note: All DP operations are handled by LocalDpMod
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Note: All DP operations are handled by LocalDpMod
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

        # Note: DP operations (including gradient clipping) are handled by LocalDpMod
        logger.info("DP Training completed - differential privacy handled by LocalDpMod")

        # Step the scheduler only once per FL round (after the last local epoch)
        if self.scheduler is not None and epoch == total_epochs - 1:  # Only on last local epoch
            current_lr_before = self.optimizer.param_groups[0]["lr"]

            # Handle ReduceLROnPlateau scheduler which requires validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)  # Use training loss as proxy
            else:
                self.scheduler.step()

            current_lr_after = self.optimizer.param_groups[0]["lr"]
            if current_lr_before != current_lr_after:
                logger.info(
                    f"FL Round {getattr(self, 'current_fl_round', '?')}: "
                    f"LR changed from {current_lr_before:.8f} to {current_lr_after:.8f}"
                )

        return avg_loss, avg_accuracy

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom Differential Privacy-specific metrics.

        Returns:
            Dictionary of custom DP metrics
        """
        # Return DP-specific metrics for WandB tracking
        return {
            "uses_localdpmod": True,
        }

    def clip_gradients(self) -> float:
        """Clip gradients according to DP-SGD protocol.

        Returns:
            Actual clipping norm applied
        """
        if self.clipping_norm <= 0.0:
            return 0.0
        # Compute total gradient norm
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        # Apply clipping if necessary
        if total_norm > self.clipping_norm:
            clipping_factor = self.clipping_norm / total_norm
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(clipping_factor)
            return self.clipping_norm
        return total_norm
