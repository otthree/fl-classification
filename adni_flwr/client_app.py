"""Client application for ADNI Federated Learning."""

import os
from typing import Any, Dict, List, Optional

import torch
from flwr.client import ClientApp
from flwr.common import Context
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.strategies import StrategyAwareClient, StrategyFactory
from adni_flwr.task import create_criterion, load_data, load_model

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from flwr.client.mod import secaggplus_mod

    SECAGGPLUS_MOD_AVAILABLE = True
except ImportError:
    SECAGGPLUS_MOD_AVAILABLE = False
    secaggplus_mod = None

try:
    from flwr.client.mod import LocalDpMod

    LOCALDPMOD_AVAILABLE = True
except ImportError:
    LOCALDPMOD_AVAILABLE = False
    LocalDpMod = None


class FLClientWandbLogger:
    """Client-side WandB logger for Federated Learning distributed training."""

    def __init__(self, config: Config, client_id: int):
        """Initialize client WandB logger."""
        self.config = config
        self.client_id = client_id
        self.wandb_enabled = WANDB_AVAILABLE and config.wandb.use_wandb
        self.run = None
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

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "", step: Optional[int] = None):
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
            logger.error(f"Client {self.client_id}: Error logging metrics to WandB: {e}")

    def finish(self):
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

    def log_config(self):
        """Log client configuration to WandB once."""
        if not self.wandb_enabled or not self.run:
            return

        # Use environment variable to persistently track config logging across client reinitializations
        env_flag = f"WANDB_CLIENT_{self.client_id}_CONFIG_LOGGED"
        if os.environ.get(env_flag) == "true":
            return

        self._log_config_internal(env_flag)

    def _connect_to_wandb(self) -> bool:
        """Internal method to connect to WandB using the stored run ID."""
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

    def _log_config_internal(self, env_flag: str):
        """Internal method to log configuration without environment variable checks."""
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


def client_fn(context: Context):
    """Client factory function.

    Args:
        context: Context containing client configuration

    Returns:
        An instance of NumPyClient
    """
    # Determine which GPU to use if available
    gpu_idx = context.node_config.get("gpu-id", 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    # Get partition ID to determine which config file to use
    partition_id = context.node_config.get("partition-id", 0)

    # Get client config files from app config
    client_config_files = context.run_config.get("client-config-files", "")
    if isinstance(client_config_files, str):
        client_config_files = [s.strip() for s in client_config_files.split(",") if s.strip()]

    # Ensure we have enough config files for all partitions
    if partition_id >= len(client_config_files):
        raise ValueError(
            f"Partition ID {partition_id} is out of range for {len(client_config_files)} client config files"
        )

    # Get the specific config file for this client
    config_path = client_config_files[partition_id]
    config = Config.from_yaml(config_path)

    # Get client ID from config
    client_id = getattr(config.fl, "client_id", partition_id)

    # Initialize client-side WandB logger for distributed training
    client_wandb_logger = FLClientWandbLogger(config, client_id)
    client_wandb_logger.init_wandb()

    # Determine which strategy to use - FAIL FAST if not specified
    if not hasattr(config.fl, "strategy") or not config.fl.strategy:
        raise ValueError(
            f"ERROR: 'strategy' not specified in client config {config_path}. "
            f"You must explicitly set 'strategy' in the FL config section. "
            f"Available strategies: fedavg, fedprox, secagg, secagg+, differential_privacy. "
            f"This prevents dangerous implicit defaults that could cause strategy mismatch between clients and server."
        )

    strategy_name = config.fl.strategy
    logger.info(
        f"Initializing client {client_id} with {strategy_name} strategy, config: {config_path} on device: {device}"
    )

    # Validate strategy requirements
    _validate_strategy_requirements(strategy_name)

    # Use new strategy system (only path supported)
    logger.info(f"Using new strategy system with {strategy_name} strategy")

    # Load model and create optimizer/criterion
    model = load_model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Get total FL rounds from FL config
    total_fl_rounds = config.fl.num_rounds
    logger.info(f"Total FL rounds: {total_fl_rounds}")

    # Load data to create criterion
    train_loader, _ = load_data(config, batch_size=config.training.batch_size)
    criterion = create_criterion(config, train_loader.dataset, device)

    # Create client strategy WITHOUT scheduler initially
    # The scheduler will be managed via Context state
    client_strategy = StrategyFactory.create_client_strategy(
        strategy_name=strategy_name,
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=None,  # No scheduler passed - will be managed by Context
    )

    # Create strategy-aware client with Context for scheduler management
    client = StrategyAwareClient(
        config=config,
        device=device,
        client_strategy=client_strategy,
        context=context,
        total_fl_rounds=total_fl_rounds,
        wandb_logger=client_wandb_logger,
    )

    logger.info(f"Client {partition_id} initialized with Context-based scheduler management")

    return client.to_client()


def _validate_strategy_requirements(strategy_name: str):
    """Validate that required dependencies are available for the strategy.

    Args:
        strategy_name: Name of the strategy to validate

    Raises:
        ValueError: If required dependencies are not available
    """
    strategy_lower = strategy_name.lower()

    if strategy_lower in ["secagg+", "secaggplus"]:
        logger.info("🔒 SecAgg+ strategy detected on client")
        if not SECAGGPLUS_MOD_AVAILABLE:
            raise ValueError(
                "SecAgg+ strategy selected but secaggplus_mod is not available. "
                "Please ensure you have the correct Flower version with SecAgg+ support."
            )
        logger.success("✅ SecAgg+ mod is available")

    elif strategy_lower == "differential_privacy":
        logger.info("🔒 Differential Privacy strategy detected on client")
        if not LOCALDPMOD_AVAILABLE:
            raise ValueError(
                "Differential Privacy strategy selected but LocalDpMod is not available. "
                "Please ensure you have the correct Flower version with LocalDpMod support."
            )
        logger.success("✅ LocalDpMod is available")


def create_mods_for_strategy(strategy_type: str, config: Optional[Config] = None) -> List[Any]:
    """Create the appropriate mods list based on strategy type.

    Args:
        strategy_type: Type of strategy ("regular", "secagg+", "differential_privacy")
        config: Configuration object (required for differential_privacy)

    Returns:
        List of mods for the ClientApp

    Raises:
        ValueError: If required dependencies are not available or config is missing
    """
    if strategy_type == "secagg+":
        if not SECAGGPLUS_MOD_AVAILABLE:
            raise ValueError("SecAgg+ strategy requires secaggplus_mod which is not available")
        return [secaggplus_mod]

    elif strategy_type == "differential_privacy":
        if not LOCALDPMOD_AVAILABLE:
            raise ValueError("Differential Privacy strategy requires LocalDpMod which is not available")

        if config is None:
            # Use default parameters for initialization
            logger.warning("No config provided for LocalDpMod, using default parameters")
            return [LocalDpMod(clipping_norm=1.0, sensitivity=1.0, epsilon=1.0, delta=1e-5)]

        # Create LocalDpMod with config parameters
        return [create_local_dp_mod(config)]

    else:
        # Regular strategy - no mods
        return []


def create_local_dp_mod(config: Config) -> LocalDpMod:
    """Create LocalDpMod instance with parameters from config.

    Args:
        config: Configuration object containing DP parameters

    Returns:
        LocalDpMod instance configured for differential privacy
    """
    if not LOCALDPMOD_AVAILABLE:
        raise ValueError("LocalDpMod is not available")

    # Get DP parameters from config or use defaults
    clipping_norm = getattr(config.fl, "clipping_norm", 1.0)

    # Calculate sensitivity (typically equal to clipping norm for gradient updates)
    sensitivity = clipping_norm

    # Get privacy parameters from config
    epsilon = getattr(config.fl, "epsilon", 1.0)
    delta = getattr(config.fl, "delta", 1e-5)

    logger.info("Creating LocalDpMod with parameters:")
    logger.info(f"  clipping_norm: {clipping_norm}")
    logger.info(f"  sensitivity: {sensitivity}")
    logger.info(f"  epsilon: {epsilon}")
    logger.info(f"  delta: {delta}")

    return LocalDpMod(clipping_norm, sensitivity, epsilon, delta)


def determine_strategy_from_config() -> str:
    """Determine which strategy is being used by checking available config files."""
    try:
        # This is a heuristic to determine strategy during app initialization
        # We'll try to read the strategy from environment or use default client_fn
        import sys

        # Check command line arguments for config files
        config_files = []
        for arg in sys.argv:
            if arg.endswith(".yaml") and "client" in arg and os.path.exists(arg):
                config_files.append(arg)

        # If we found config files, check strategy
        for config_file in config_files:
            try:
                config = Config.from_yaml(config_file)
                if hasattr(config.fl, "strategy"):
                    strategy = config.fl.strategy.lower()
                    if strategy in ["secagg+", "secaggplus"]:
                        logger.info(f"🔒 SecAgg+ detected in config: {config_file}")
                        return "secagg+"
                    elif strategy == "differential_privacy":
                        logger.info(f"🔒 Differential Privacy detected in config: {config_file}")
                        return "differential_privacy"
            except Exception:
                continue

        return "regular"
    except Exception:
        # If we can't determine, default to regular client
        return "regular"


def create_client_app(strategy_type: str = None, config: Optional[Config] = None) -> ClientApp:
    """Create a ClientApp with appropriate mods based on strategy type.

    Args:
        strategy_type: Type of strategy to create app for. If None, auto-detect from config files.
        config: Configuration object (optional, used for strategy-specific parameters)

    Returns:
        ClientApp instance with appropriate mods
    """
    if strategy_type is None:
        strategy_type = determine_strategy_from_config()

    try:
        # Create mods for the strategy
        mods = create_mods_for_strategy(strategy_type, config)

        # Log the app creation
        if strategy_type == "secagg+":
            logger.info("🔒 Creating SecAgg+ client app with secaggplus_mod")
        elif strategy_type == "differential_privacy":
            logger.info("🔒 Creating Differential Privacy client app with LocalDpMod")
        else:
            logger.info("📊 Creating regular client app")

        return ClientApp(client_fn=client_fn, mods=mods)

    except ValueError as e:
        logger.error(f"❌ {e}, falling back to regular client")
        return ClientApp(client_fn=client_fn)


# Create the main app instance with auto-detection
app = create_client_app()

# Create specialized app instances for explicit use
regular_app = create_client_app("regular")

# Create SecAgg+ app if available
if SECAGGPLUS_MOD_AVAILABLE:
    secagg_plus_app = create_client_app("secagg+")
else:
    secagg_plus_app = None

# Create Differential Privacy app if available
if LOCALDPMOD_AVAILABLE:
    differential_privacy_app = create_client_app("differential_privacy")
else:
    differential_privacy_app = None
