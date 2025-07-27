"""Federated Learning WandB experiment tracking utilities."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger

from adni_classification.config.config import Config

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseFLWandbLogger(ABC):
    """Base WandB logger for Federated Learning with common functionality."""

    def __init__(self, config: Config):
        """Initialize base WandB logger.

        Args:
            config: Configuration object containing WandB settings
        """
        self.config = config
        self.wandb_enabled = WANDB_AVAILABLE and config.wandb.use_wandb
        self.run = None
        self.run_id = None

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
            step: Optional step number
        """
        if not self.wandb_enabled or not self.run:
            return

        try:
            # Filter out non-loggable metrics (like JSON strings, complex objects)
            loggable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, bool)):
                    loggable_metrics[k] = v
                elif hasattr(v, "__float__"):  # Handle numpy scalars
                    try:
                        loggable_metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values

            if not loggable_metrics:
                return

            # Add prefix to metric names if provided
            if prefix:
                prefixed_metrics = {f"{prefix}/{k}": v for k, v in loggable_metrics.items()}
            else:
                prefixed_metrics = loggable_metrics

            # Log the metrics
            if step is not None:
                wandb.log(prefixed_metrics, step=step)
            else:
                wandb.log(prefixed_metrics)

        except Exception as e:
            logger.error(f"Error logging metrics to WandB: {e}")

    def finish(self) -> None:
        """Finish WandB run."""
        if self.wandb_enabled and self.run:
            try:
                wandb.finish()
                logger.info("WandB run finished successfully")
            except Exception as e:
                logger.error(f"Error finishing WandB run: {e}")

    def get_run_id(self) -> Optional[str]:
        """Get the current WandB run ID.

        Returns:
            Current run ID or None if not available
        """
        return self.run_id

    @abstractmethod
    def init_wandb(self, **kwargs) -> Optional[str]:
        """Initialize WandB run. Implementation varies by logger type.

        Returns:
            Run ID if successful, None otherwise
        """
        pass


class FLServerWandbLogger(BaseFLWandbLogger):
    """Server-side WandB logger for Federated Learning."""

    def __init__(self, config: Config):
        """Initialize server WandB logger.

        Args:
            config: Configuration object containing WandB settings
        """
        super().__init__(config)
        logger.info(f"Server WandB logging: {'enabled' if self.wandb_enabled else 'disabled'}")

    def init_wandb(self, enable_shared_mode: bool = True) -> Optional[str]:
        """Initialize WandB run as primary node.

        Args:
            enable_shared_mode: Whether to enable shared mode for distributed training.
                              When True, server acts as primary node and clients can join the same run.

        Returns:
            Run ID if successful, None otherwise
        """
        if not self.wandb_enabled:
            return None

        try:
            # Prepare WandB settings for distributed training
            wandb_settings = None
            if enable_shared_mode:
                # Server is the primary node in distributed training
                wandb_settings = wandb.Settings(
                    mode="shared",
                    x_primary=True,
                    x_label="server",  # Unique label for server node
                )
                logger.info("Initializing WandB in shared mode as primary node (server)")

            # Initialize wandb with full configuration
            self.run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.run_name,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=self.config.to_dict(),
                settings=wandb_settings,
            )

            if self.run:
                self.run_id = self.run.id
                logger.success(f"WandB run initialized successfully. Run ID: {self.run_id}")
                if enable_shared_mode:
                    logger.info(f"Clients can join this run using run ID: {self.run_id}")
                return self.run_id

        except Exception as e:
            logger.error(f"Error initializing WandB: {e}")
            self.wandb_enabled = False
            return None


class FLClientWandbLogger(BaseFLWandbLogger):
    """Client-side WandB logger for Federated Learning."""

    def __init__(self, config: Config, client_id: int):
        """Initialize client WandB logger.

        Args:
            config: Configuration object containing WandB settings
            client_id: Unique identifier for this client
        """
        super().__init__(config)
        self.client_id = client_id
        self.server_run_id = None
        logger.info(f"Client {client_id} WandB logging: {'enabled' if self.wandb_enabled else 'disabled'}")

    def init_wandb(self, run_id: Optional[str] = None) -> bool:
        """Initialize WandB run in shared mode to join server's run.

        Args:
            run_id: Server's WandB run ID to join. If None, will try environment variable.

        Returns:
            True if successfully joined server's run, False otherwise
        """
        if not self.wandb_enabled:
            logger.info(f"Client {self.client_id}: WandB disabled")
            return False

        # Get server's run ID from parameter, fallback to environment variable
        self.server_run_id = run_id or os.environ.get("WANDB_FL_RUN_ID")
        if not self.server_run_id:
            logger.info(
                f"Client {self.client_id}: No WandB run ID provided. Will attempt to join when received from server."
            )
            return False

        return self._connect_to_wandb()

    def join_wandb_run(self, run_id: str) -> bool:
        """Join an existing WandB run with the provided run ID.

        Args:
            run_id: Server's WandB run ID to join

        Returns:
            True if successfully joined, False otherwise
        """
        if not self.wandb_enabled:
            return False

        if self.run:
            logger.info(f"Client {self.client_id}: Already connected to WandB run")
            return True

        self.server_run_id = run_id
        return self._connect_to_wandb()

    def log_config(self) -> None:
        """Log client configuration to WandB once."""
        if not self.wandb_enabled or not self.run:
            return

        # Use environment variable to persistently track config logging across client reinitializations
        env_flag = f"WANDB_CLIENT_{self.client_id}_CONFIG_LOGGED"
        if os.environ.get(env_flag) == "true":
            return

        self._log_config_internal(env_flag)

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
        """Log metrics to WandB with client-specific error handling.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
            step: Optional step number
        """
        if not self.wandb_enabled or not self.run:
            return

        try:
            # Filter out non-loggable metrics (like JSON strings, complex objects)
            loggable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, bool)):
                    loggable_metrics[k] = v
                elif hasattr(v, "__float__"):  # Handle numpy scalars
                    try:
                        loggable_metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values

            if not loggable_metrics:
                return

            # Add prefix to metric names if provided
            if prefix:
                prefixed_metrics = {f"{prefix}/{k}": v for k, v in loggable_metrics.items()}
            else:
                prefixed_metrics = loggable_metrics

            # Log the metrics
            if step is not None:
                wandb.log(prefixed_metrics, step=step)
            else:
                wandb.log(prefixed_metrics)

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error logging metrics to WandB: {e}")

    def finish(self) -> None:
        """Finish client's participation in WandB run.

        Note: This doesn't finish the actual run since x_update_finish_state=False,
        only this client's participation.
        """
        if self.wandb_enabled and self.run:
            try:
                wandb.finish()
                logger.info(f"Client {self.client_id}: Finished WandB logging")
            except Exception as e:
                logger.error(f"Client {self.client_id}: Error finishing WandB: {e}")

    def _connect_to_wandb(self) -> bool:
        """Internal method to connect to WandB using the stored run ID.

        Returns:
            True if successfully connected, False otherwise
        """
        if not self.server_run_id:
            return False

        try:
            # Set up environment variable flag for config logging
            env_flag = f"WANDB_CLIENT_{self.client_id}_CONFIG_LOGGED"
            config_already_logged = os.environ.get(env_flag) == "true"

            # Client joins the server's run as a worker node
            wandb_settings = wandb.Settings(
                mode="shared",
                x_primary=False,  # Client is a worker node
                x_label=f"client_{self.client_id}",  # Unique label for this client
                x_update_finish_state=False,  # Prevent clients from finishing the run prematurely
            )

            logger.info(f"Client {self.client_id}: Joining WandB run {self.server_run_id} in shared mode")

            # Join the server's run
            self.run = wandb.init(
                id=self.server_run_id,  # Use server's run ID
                settings=wandb_settings,
                resume="allow",  # Allow resuming the run
            )

            if self.run:
                self.run_id = self.run.id
                logger.success(f"Client {self.client_id}: Successfully joined WandB run {self.server_run_id}")
                # Log client configuration once after successful connection (only if not already logged)
                if not config_already_logged:
                    self._log_config_internal(env_flag)
                return True
            else:
                logger.error(f"Client {self.client_id}: Failed to join WandB run")
                self.wandb_enabled = False
                return False

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error joining WandB run: {e}")
            self.wandb_enabled = False
            return False

    def _log_config_internal(self, env_flag: str) -> None:
        """Internal method to log configuration without environment variable checks.

        Args:
            env_flag: Environment variable flag to set after successful logging
        """
        try:
            # Use the existing to_dict method from Config class
            config_dict = self.config.to_dict()

            # Add client-specific prefix
            prefixed_config = {f"client_{self.client_id}": config_dict}

            # Log the configuration
            wandb.config.update(prefixed_config)

            # Set persistent flag to prevent duplicate logging
            os.environ[env_flag] = "true"
            logger.info(f"Client {self.client_id}: Configuration logged to WandB")

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error logging config to WandB: {e}")


# Backwards compatibility aliases
FLWandbLogger = FLServerWandbLogger  # For server_app.py compatibility
