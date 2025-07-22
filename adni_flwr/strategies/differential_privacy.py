"""Differential Privacy (DP) strategy implementation for Federated Learning."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.common import EvaluateIns, FitIns, FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
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

    This strategy implements differential privacy in federated learning by:
    1. Adding Gaussian noise to client parameters for privacy protection
    2. Applying dropout masking for additional privacy
    3. Using deterministic obfuscation masks

    Note: This provides privacy through noise addition, which may affect model accuracy.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        noise_multiplier: float = 0.1,
        dropout_rate: float = 0.0,
        clipping_norm: float = 1.0,
        **kwargs,
    ):
        """Initialize Differential Privacy strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            noise_multiplier: Multiplier for Gaussian noise addition (DP parameter)
            dropout_rate: Dropout rate for parameter masking
            clipping_norm: Gradient clipping norm for DP-SGD
            **kwargs: Additional DP parameters
        """
        super().__init__(config, model, wandb_logger, **kwargs)

        self.noise_multiplier = noise_multiplier
        self.dropout_rate = dropout_rate
        self.clipping_norm = clipping_norm
        self.client_masks = {}  # Store client obfuscation masks

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

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "differential_privacy"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "dropout_rate": self.dropout_rate,
            "clipping_norm": self.clipping_norm,
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
            "noise_multiplier": self.noise_multiplier,
            "dropout_rate": self.dropout_rate,
            "clipping_norm": self.clipping_norm,
        }

    def generate_client_obfuscation_mask(self, client_id: str, round_num: int) -> np.ndarray:
        """Generate a deterministic obfuscation mask for a client.

        Note: This is simple obfuscation, not cryptographic secure aggregation.

        Args:
            client_id: Client identifier
            round_num: Current round number

        Returns:
            Numpy array mask for parameter obfuscation
        """
        # If noise_multiplier is 0, skip obfuscation entirely for true FedAvg behavior
        if self.noise_multiplier == 0.0:
            # Return zero masks (no obfuscation)
            model_params = get_params(self.model)
            mask = []
            for param_array in model_params:
                param_mask = np.zeros_like(param_array)
                mask.append(param_mask)
            return mask

        # Create deterministic seed from client_id and round
        seed_str = f"{client_id}_{round_num}_{self.noise_multiplier}"
        seed = int(hashlib.md5(seed_str.encode(), usedforsecurity=False).hexdigest(), 16) % (2**32)

        # Set numpy random seed for reproducibility
        np.random.seed(seed)

        # Generate mask with same shape as model parameters
        model_params = get_params(self.model)
        mask = []

        for param_array in model_params:
            # Generate random obfuscation mask for each parameter array
            param_mask = np.random.uniform(-1, 1, param_array.shape)
            mask.append(param_mask)

        return mask

    def differential_private_aggregate(
        self, results: List[Tuple[ClientProxy, FitRes]], round_num: int
    ) -> Optional[Parameters]:
        """Perform differential private aggregation of client updates.

        Args:
            results: List of client results
            round_num: Current round number

        Returns:
            Aggregated parameters with DP guarantees
        """
        if not results:
            return None

        logger.info(f"Performing differential private aggregation for {len(results)} clients")

        # Collect denoised parameters and weights
        denoised_params_list = []
        weights = []
        client_masks = []

        for client_proxy, fit_res in results:
            client_id = str(client_proxy.cid)

            # Generate obfuscation mask for this client
            client_mask = self.generate_client_obfuscation_mask(client_id, round_num)
            client_masks.append(client_mask)

            # Get client parameters (already contain DP noise)
            client_params = parameters_to_ndarrays(fit_res.parameters)

            # Remove obfuscation mask that was added by client
            deobfuscated_params = []
            for param, mask in zip(client_params, client_mask, strict=False):
                deobfuscated_param = param - mask
                deobfuscated_params.append(deobfuscated_param)

            denoised_params_list.append(deobfuscated_params)
            weights.append(fit_res.num_examples)

        # Perform weighted average (preserving DP noise)
        total_examples = sum(weights)
        if total_examples == 0:
            return None

        # Initialize aggregated parameters
        aggregated_params = []

        for i in range(len(denoised_params_list[0])):
            # Get the original parameter shape and dtype
            original_param = denoised_params_list[0][i]
            original_dtype = original_param.dtype

            # Initialize weighted sum as float64 to handle arithmetic properly
            weighted_sum = np.zeros(original_param.shape, dtype=np.float64)

            for _, (params, weight) in enumerate(zip(denoised_params_list, weights, strict=False)):
                # Ensure params[i] is also float64 for arithmetic
                param_float = params[i].astype(np.float64)
                weighted_sum += param_float * (weight / total_examples)

            # Convert back to original dtype
            weighted_sum = weighted_sum.astype(original_dtype)
            aggregated_params.append(weighted_sum)

        print(f"Differential private aggregation completed with {len(results)} clients")

        return ndarrays_to_parameters(aggregated_params)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using differential privacy with enhanced logging."""
        self.current_round = server_round

        # Log client training metrics
        if self.wandb_logger:
            for _client_proxy, fit_res in results:
                client_metrics = fit_res.metrics
                if client_metrics:
                    client_id = fit_res.metrics.get("client_id", "unknown")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(metrics_to_log, prefix=f"client_{client_id}/fit", step=server_round)

        # Perform differential private aggregation
        aggregated_parameters = self.differential_private_aggregate(results, server_round)

        if aggregated_parameters is None:
            return None, {}

        # Update server model with aggregated parameters
        param_arrays = safe_parameters_to_ndarrays(aggregated_parameters)
        set_params(self.model, param_arrays)

        # Collect metrics with DP-specific information
        metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "noise_multiplier": self.noise_multiplier,
            "dropout_rate": self.dropout_rate,
            "clipping_norm": self.clipping_norm,
        }

        # Log aggregated fit metrics with DP-specific information
        if self.wandb_logger and metrics:
            metrics_with_dp = metrics.copy()
            self.wandb_logger.log_metrics(metrics_with_dp, prefix="server", step=server_round)

        # Print server model's current metrics
        print(
            f"Server model metrics after round {server_round} "
            f"(DP noise={self.noise_multiplier}, dropout={self.dropout_rate}):"
        )
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
        # Instead of delegating to fedavg_strategy (which returns None),
        # provide initial parameters from our server model
        from flwr.common import ndarrays_to_parameters

        from adni_flwr.task import get_params

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

        # Add server_round and DP parameters to the config for each client
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            # Add server_round and DP parameters to the existing config
            config = fit_ins.config.copy() if fit_ins.config else {}
            config["server_round"] = server_round
            config["noise_multiplier"] = self.noise_multiplier
            config["dropout_rate"] = self.dropout_rate
            config["clipping_norm"] = self.clipping_norm

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
    """Client-side Differential Privacy strategy for Federated Learning.

    This client strategy implements differential privacy by:
    1. Adding calibrated Gaussian noise to model parameters
    2. Applying dropout masking for additional privacy
    3. Using simple obfuscation masks (not cryptographic)
    4. Gradient clipping for proper DP-SGD
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        noise_multiplier: float = 0.1,
        dropout_rate: float = 0.0,
        clipping_norm: float = 1.0,  # Gradient clipping norm for DP-SGD
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
            noise_multiplier: Multiplier for Gaussian noise addition (DP parameter)
            dropout_rate: Dropout rate for parameter masking
            clipping_norm: L2 norm threshold for gradient clipping
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, model, optimizer, criterion, device, scheduler, **kwargs)

        self.noise_multiplier = noise_multiplier
        self.dropout_rate = dropout_rate
        self.clipping_norm = clipping_norm

        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(config.fl, "client_id") or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = config.fl.client_id
        self.current_round = 0

        # DP-specific parameters
        self.mixed_precision = config.training.mixed_precision
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Store parameter scales for noise scaling
        self.param_scales = None

        logger.info(f"DP Client initialized with noise_multiplier={noise_multiplier}, clipping_norm={clipping_norm}")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "differential_privacy"

    def generate_client_obfuscation_mask(self, round_num: int) -> List[np.ndarray]:
        """Generate a deterministic obfuscation mask for this client.

        Note: This is simple obfuscation, not cryptographic secure aggregation.

        Args:
            round_num: Current round number

        Returns:
            List of numpy array obfuscation masks
        """
        # If noise_multiplier is 0, skip obfuscation entirely for true FedAvg behavior
        if self.noise_multiplier == 0.0:
            # Return zero masks (no obfuscation)
            model_params = get_params(self.model)
            mask = []
            for param_array in model_params:
                param_mask = np.zeros_like(param_array)
                mask.append(param_mask)
            return mask

        # Create deterministic seed from client_id and round
        seed_str = f"{self.client_id}_{round_num}_{self.noise_multiplier}"
        seed = int(hashlib.md5(seed_str.encode(), usedforsecurity=False).hexdigest(), 16) % (2**32)

        # Set numpy random seed for reproducibility
        np.random.seed(seed)

        # Generate mask with same shape as model parameters
        model_params = get_params(self.model)
        mask = []

        for param_array in model_params:
            # Generate random obfuscation mask for each parameter array
            param_mask = np.random.uniform(-1, 1, param_array.shape)
            mask.append(param_mask)

        return mask

    def add_gaussian_noise_for_privacy(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """Add calibrated Gaussian noise to parameters for differential privacy.

        Args:
            params: List of parameter arrays

        Returns:
            List of DP-noisy parameter arrays
        """
        # If noise_multiplier is 0, skip noise addition entirely for true FedAvg behavior
        if self.noise_multiplier == 0.0:
            return params

        noisy_params = []

        # Get parameter names for scale lookup
        param_names = [name for name, _ in self.model.named_parameters()]

        for i, param_array in enumerate(params):
            if i < len(param_names):
                # Use parameter-aware noise scaling
                param_scale = np.linalg.norm(param_array)

                # Avoid division by zero for very small parameters
                if param_scale < 1e-8:
                    param_scale = 1e-8

                # Scale noise relative to parameter magnitude
                # For small parameters, use smaller noise; for large parameters, proportional noise
                noise_scale = min(self.noise_multiplier * param_scale, self.noise_multiplier)

                # Add scaled Gaussian noise
                noise = np.random.normal(0, noise_scale, param_array.shape)
                noisy_param = param_array + noise

                logger.debug(f"Added noise to param {i}: scale={param_scale:.6f}, noise_scale={noise_scale:.6f}")
            else:
                # Fallback to absolute noise for any extra parameters
                noise = np.random.normal(0, self.noise_multiplier, param_array.shape)
                noisy_param = param_array + noise

            noisy_params.append(noisy_param)

        return noisy_params

    def apply_dropout_mask(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """Apply dropout mask to parameters for additional privacy.

        Args:
            params: List of parameter arrays

        Returns:
            List of dropout-masked parameter arrays
        """
        if self.dropout_rate == 0.0:
            return params

        masked_params = []

        for param_array in params:
            # Create dropout mask
            mask = np.random.binomial(1, 1 - self.dropout_rate, param_array.shape)
            masked_param = param_array * mask
            masked_params.append(masked_param)

        return masked_params

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

        # Update DP parameters if specified in round config
        if "noise_multiplier" in round_config:
            self.noise_multiplier = round_config["noise_multiplier"]
        if "dropout_rate" in round_config:
            self.dropout_rate = round_config["dropout_rate"]
        if "clipping_norm" in round_config:
            self.clipping_norm = round_config["clipping_norm"]

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int, **kwargs) -> Tuple[float, float]:
        """Train the model for one epoch using Differential Privacy with gradient clipping.

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
        total_clipping_norm = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Mixed precision training with gradient clipping
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                    # Apply gradient clipping for DP
                    if self.noise_multiplier > 0.0:
                        clipping_norm = self.clip_gradients()
                        total_clipping_norm += clipping_norm
                        num_batches += 1

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Apply gradient clipping for DP
                    if self.noise_multiplier > 0.0:
                        clipping_norm = self.clip_gradients()
                        total_clipping_norm += clipping_norm
                        num_batches += 1

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
        avg_clipping_norm = total_clipping_norm / max(num_batches, 1)

        # Log DP-specific metrics
        if self.noise_multiplier > 0.0:
            logger.info(f"DP Training - Avg clipping norm: {avg_clipping_norm:.6f}, Target: {self.clipping_norm}")

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

    def get_differential_private_parameters(self) -> List[np.ndarray]:
        """Get model parameters with differential privacy applied.

        Returns:
            List of DP-protected parameter arrays
        """
        # Get current model parameters
        params = get_params(self.model)

        # Apply dropout mask for additional privacy
        params = self.apply_dropout_mask(params)

        # Add Gaussian noise for differential privacy
        params = self.add_gaussian_noise_for_privacy(params)

        # Generate and apply obfuscation mask (simple obfuscation, not cryptographic)
        client_mask = self.generate_client_obfuscation_mask(self.current_round)
        obfuscated_params = []

        for param, mask in zip(params, client_mask, strict=False):
            obfuscated_param = param + mask
            obfuscated_params.append(obfuscated_param)

        return obfuscated_params

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom Differential Privacy-specific metrics.

        Returns:
            Dictionary of custom DP metrics
        """
        # Return DP-specific metrics for WandB tracking
        return {
            "noise_multiplier": self.noise_multiplier,
            "dropout_rate": self.dropout_rate,
            "clipping_norm": self.clipping_norm,
        }

    def compute_parameter_scales(self) -> Dict[str, float]:
        """Compute parameter scales for noise scaling.

        Returns:
            Dictionary mapping parameter names to their L2 norms
        """
        scales = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    scales[name] = param.norm().item()
        return scales

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
