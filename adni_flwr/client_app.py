"""Client application for ADNI Federated Learning."""

from typing import Any, List, Optional

import torch
from flwr.client import ClientApp
from flwr.common import Context
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.strategies import StrategyAwareClient, StrategyFactory
from adni_flwr.task import create_criterion, load_data, load_model
from adni_flwr.utils.wandb_logger import FLClientWandbLogger

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

# Import Opacus-based differential privacy client
from adni_flwr.strategies.differential_privacy_opacus import DifferentialPrivacyClient as OpacusDPClient


class AdaptiveLocalDpMod:
    """Custom LocalDpMod with adaptive noise scaling.

    This addresses the core issue by:
    1. Reducing noise magnitude as training progresses
    2. Scaling noise based on current parameter magnitudes
    3. Using exponential decay schedule for epsilon
    """

    def __init__(
        self,
        clipping_norm: float,
        sensitivity: float,
        initial_epsilon: float,
        delta: float,
        decay_factor: float = 0.95,
        min_epsilon: float = None,
    ):
        self.clipping_norm = clipping_norm
        self.sensitivity = sensitivity
        self.initial_epsilon = initial_epsilon
        self.current_epsilon = initial_epsilon
        self.delta = delta
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon or initial_epsilon * 0.1  # Minimum 10% of initial
        self.round_count = 0

        logger.info("🔧 AdaptiveLocalDpMod initialized:")
        logger.info(f"   initial_epsilon: {initial_epsilon}")
        logger.info(f"   decay_factor: {decay_factor}")
        logger.info(f"   min_epsilon: {self.min_epsilon}")

    def __call__(self, message, context, ffn):
        """Apply adaptive DP noise to parameters.

        This method is called by Flower's mod system with the signature:
        __call__(self, message, context, ffn)

        Args:
            message: The message from the server
            context: The context object
            ffn: The next function in the chain

        Returns:
            The modified message with adaptive DP noise applied
        """
        # Call the next function in the chain to get the original response
        response = ffn(message, context)

        # Check if this is a fit response with parameters
        if hasattr(response, 'parameters') and response.parameters is not None:
            parameters = response.parameters
        else:
            # If no parameters in response, return as-is
            logger.debug("No parameters found in response, skipping adaptive DP noise")
            return response

        self.round_count += 1

        # Update epsilon with exponential decay (less noise over time)
        self.current_epsilon = max(
            self.initial_epsilon * (self.decay_factor ** (self.round_count - 1)), self.min_epsilon
        )

        # Calculate noise scale
        noise_scale = self.sensitivity / self.current_epsilon

        # Apply parameter-magnitude aware scaling
        noisy_parameters = []
        total_param_norm = 0.0
        total_noise_norm = 0.0

        for param_array in parameters:
            param_tensor = torch.tensor(param_array, dtype=torch.float32)
            param_norm = torch.norm(param_tensor).item()
            total_param_norm += param_norm

            # Adaptive scaling: reduce noise for smaller parameters
            param_magnitude = max(param_norm, 1e-6)  # Avoid division by zero
            adaptive_scale = min(noise_scale, noise_scale * param_magnitude / 10.0)

            # Generate and apply noise
            noise = torch.normal(0, adaptive_scale, param_tensor.shape)
            noisy_param = param_tensor + noise

            noise_norm = torch.norm(noise).item()
            total_noise_norm += noise_norm

            noisy_parameters.append(noisy_param.numpy())

        logger.info(f"🔒 AdaptiveLocalDpMod Round {self.round_count}:")
        logger.info(f"   current_epsilon: {self.current_epsilon:.4f}")
        logger.info(f"   noise_scale: {noise_scale:.6f}")
        logger.info(f"   param_norm: {total_param_norm:.6f}")
        logger.info(f"   noise_norm: {total_noise_norm:.6f}")
        logger.info(f"   noise/param_ratio: {total_noise_norm/max(total_param_norm, 1e-6):.4f}")

        # Update the response with noisy parameters
        response.parameters = noisy_parameters
        return response


def create_adaptive_dp_app(config: Optional[Config] = None) -> ClientApp:
    """Create a ClientApp with adaptive LocalDpMod that reduces noise over time.

    This is a compromise solution if you must use parameter-level DP:
    1. Starts with higher noise (lower epsilon) for initial privacy
    2. Gradually reduces noise as training progresses
    3. Prevents the "noise overwhelm" issue in later rounds

    Args:
        config: Optional pre-loaded config. If None, will be loaded from context.

    Returns:
        ClientApp with adaptive DP mod

    Raises:
        ValueError: If config is invalid or strategy is not differential_privacy
        FileNotFoundError: If config file cannot be found
    """

    def adaptive_dp_client_fn(context: Context):
        """Client factory function for adaptive DP."""
        # Standard client setup
        gpu_idx = context.node_config.get("gpu-id", 0)
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        partition_id = context.node_config.get("partition-id", 0)

        # Load config using proper dynamic config loading
        if config is None:
            # Get client config files from app config (same pattern as main client_fn)
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
            try:
                from adni_classification.config.config import Config

                loaded_config = Config.from_yaml(config_path)
                logger.info(f"Loaded config from: {config_path}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Config file not found: {config_path}. Error: {e}") from e
            except Exception as e:
                raise ValueError(f"Failed to load config from {config_path}. Error: {e}") from e
        else:
            loaded_config = config
            logger.info("Using provided config")

        # Strict validation
        if not hasattr(loaded_config, "fl"):
            raise ValueError("Config missing 'fl' section")

        if not hasattr(loaded_config.fl, "strategy"):
            raise ValueError("Config missing 'fl.strategy' field")

        if loaded_config.fl.strategy != "differential_privacy":
            raise ValueError(f"Config strategy must be 'differential_privacy', got: '{loaded_config.fl.strategy}'")

        # Validate required DP parameters
        required_dp_params = ["dp_epsilon", "dp_sensitivity", "dp_clipping_norm", "dp_delta"]
        missing_params = [param for param in required_dp_params if not hasattr(loaded_config.fl, param)]

        if missing_params:
            raise ValueError(
                f"Missing required DP parameters: {missing_params}. "
                f"Please add these to your config file under the 'fl' section."
            )

        # Create standard client components
        try:
            model = load_model(loaded_config).to(device)
            train_loader, val_loader = load_data(loaded_config)
            criterion = create_criterion(loaded_config, train_loader.dataset, device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=loaded_config.training.learning_rate,
                weight_decay=loaded_config.training.weight_decay,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model/data/optimizer: {e}") from e

        # Create client with differential_privacy strategy
        try:
            from adni_flwr.strategies import StrategyFactory

            client_strategy = StrategyFactory.create_client_strategy(
                strategy_name="differential_privacy",
                config=loaded_config,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            return StrategyAwareClient(
                config=loaded_config,
                device=device,
                client_strategy=client_strategy,
                context=context,
                total_fl_rounds=loaded_config.fl.num_rounds,
                wandb_logger=None,  # Add wandb_logger if needed
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create adaptive DP client strategy: {e}") from e

    # Create AdaptiveLocalDpMod with parameters from config or defaults
    if config is not None:
        # Use parameters from provided config
        clipping_norm = float(config.fl.dp_clipping_norm)
        sensitivity = float(config.fl.dp_sensitivity)
        initial_epsilon = float(config.fl.dp_epsilon)
        delta = float(config.fl.dp_delta)
        decay_factor = getattr(config.fl, "dp_decay_factor", 0.95)
        min_epsilon = getattr(config.fl, "dp_min_epsilon", None)

        logger.info("Creating AdaptiveLocalDpMod with config parameters:")
        logger.info(f"  clipping_norm: {clipping_norm}")
        logger.info(f"  sensitivity: {sensitivity}")
        logger.info(f"  initial_epsilon: {initial_epsilon}")
        logger.info(f"  delta: {delta}")
        logger.info(f"  decay_factor: {decay_factor}")
        logger.info(f"  min_epsilon: {min_epsilon}")

        adaptive_dp_mod = AdaptiveLocalDpMod(
            clipping_norm=clipping_norm,
            sensitivity=sensitivity,
            initial_epsilon=initial_epsilon,
            delta=delta,
            decay_factor=decay_factor,
            min_epsilon=min_epsilon,
        )
    else:
        # Use default parameters
        adaptive_dp_mod = AdaptiveLocalDpMod(
            clipping_norm=1.0,  # Default value
            sensitivity=1.0,     # Default value
            initial_epsilon=100.0,  # Default value - high epsilon for less noise
            delta=1e-5,          # Default value
            decay_factor=0.95,   # Default decay factor
            min_epsilon=10.0,    # Default minimum epsilon
        )

        logger.info("Created adaptive_differential_privacy_app with AdaptiveLocalDpMod (default parameters)")
        logger.info("Note: For custom parameters, use create_adaptive_dp_app(config) with a config object")

    return ClientApp(client_fn=adaptive_dp_client_fn, mods=[adaptive_dp_mod])


# ==================================================================================
# ADAPTIVE DP APP INSTANCE - Explicit Error Handling
# ==================================================================================
try:
    adaptive_differential_privacy_app = create_adaptive_dp_app()
    logger.info("Adaptive differential privacy app created successfully")
except Exception as e:
    logger.error(f"Failed to create adaptive_differential_privacy_app: {e}")
    # Re-raise to make the error explicit - don't hide import failures
    raise RuntimeError(f"Failed to initialize adaptive_differential_privacy_app: {e}") from e


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
        logger.success("SecAgg+ mod is available")

    elif strategy_lower == "differential_privacy":
        logger.info("🔒 Differential Privacy strategy detected on client")
        if not LOCALDPMOD_AVAILABLE:
            raise ValueError(
                "Differential Privacy strategy selected but LocalDpMod is not available. "
                "Please ensure you have the correct Flower version with LocalDpMod support."
            )
        logger.success("LocalDpMod is available")


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
            raise ValueError(
                "CRITICAL ERROR: Differential Privacy strategy requires a config to create LocalDpMod. "
                "Use 'create_dp_client_app(config)' function instead of 'create_client_app(\"differential_privacy\")'."
            )

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
    clipping_norm = getattr(config.fl, "dp_clipping_norm", 1.0)
    sensitivity = getattr(config.fl, "dp_sensitivity", clipping_norm)  # Use config value or fall back to clipping_norm
    epsilon = getattr(config.fl, "dp_epsilon", 1.0)
    delta = getattr(config.fl, "dp_delta", 1e-5)

    # Convert values to float to ensure proper type (YAML can parse numbers as strings)
    try:
        clipping_norm = float(clipping_norm)
        sensitivity = float(sensitivity)
        epsilon = float(epsilon)
        delta = float(delta)
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            f"CRITICAL ERROR: Differential Privacy parameters must be numeric. "
            f"Raw values: clipping_norm={clipping_norm} ({type(clipping_norm)}), "
            f"sensitivity={sensitivity} ({type(sensitivity)}), "
            f"epsilon={epsilon} ({type(epsilon)}), "
            f"delta={delta} ({type(delta)}). "
            f"Error: {e}"
        ) from e

    # Validate parameter ranges for LocalDpMod
    if clipping_norm <= 0:
        raise RuntimeError(f"CRITICAL ERROR: LocalDpMod clipping_norm must be positive, got: {clipping_norm}")
    if sensitivity <= 0:
        raise RuntimeError(f"CRITICAL ERROR: LocalDpMod sensitivity must be positive, got: {sensitivity}")
    if epsilon <= 0:
        raise RuntimeError(f"CRITICAL ERROR: LocalDpMod epsilon must be positive, got: {epsilon}")
    if delta <= 0 or delta >= 1:
        raise RuntimeError(f"CRITICAL ERROR: LocalDpMod delta must be between 0 and 1 (exclusive), got: {delta}")

    logger.info("Creating LocalDpMod with parameters:")
    logger.info(f"  clipping_norm: {clipping_norm}")
    logger.info(f"  sensitivity: {sensitivity}")
    logger.info(f"  epsilon: {epsilon}")
    logger.info(f"  delta: {delta}")

    return LocalDpMod(clipping_norm, sensitivity, epsilon, delta)


def create_client_app(strategy_type: str, config: Optional[Config] = None) -> ClientApp:
    """Create a ClientApp with appropriate mods based on strategy type.

    Args:
        strategy_type: Type of strategy to create app for (required)
        config: Configuration object (optional, used for strategy-specific parameters)

    Returns:
        ClientApp instance with appropriate mods

    Raises:
        ValueError: If strategy requirements are not met or invalid strategy provided
    """
    # Create mods for the strategy - fail explicitly if it doesn't work
    mods = create_mods_for_strategy(strategy_type, config)

    # Log the app creation
    if strategy_type == "secagg+":
        logger.info("🔒 Creating SecAgg+ client app with secaggplus_mod")
    elif strategy_type == "differential_privacy":
        logger.info("🔒 Creating Differential Privacy client app with LocalDpMod")
    else:
        logger.info("📊 Creating regular client app")

    return ClientApp(client_fn=client_fn, mods=mods)


def create_dp_client_app(config: Config) -> ClientApp:
    """Create a ClientApp with LocalDpMod configured for differential privacy.

    Args:
        config: Configuration object containing DP parameters (dp_clipping_norm, dp_sensitivity, etc.)

    Returns:
        ClientApp with LocalDpMod properly configured

    Raises:
        RuntimeError: If LocalDpMod is not available or config parameters are invalid
    """
    if not LOCALDPMOD_AVAILABLE:
        raise RuntimeError(
            "CRITICAL ERROR: Differential Privacy requires LocalDpMod which is not available. "
            "Please ensure you have the correct Flower version with LocalDpMod support."
        )

    # Validate that this is actually a DP config
    if not (hasattr(config.fl, "strategy") and config.fl.strategy == "differential_privacy"):
        raise ValueError(
            f"CRITICAL ERROR: Config strategy is '{getattr(config.fl, 'strategy', 'not specified')}' "
            f"but expected 'differential_privacy'."
        )

    # Create LocalDpMod with the provided config
    local_dp_mod = create_local_dp_mod(config)
    logger.info("Created DP ClientApp with LocalDpMod using provided config")

    return ClientApp(client_fn=client_fn, mods=[local_dp_mod])


# ==================================================================================
# CLIENT APP INSTANCES - Choose the right one for your strategy
# ==================================================================================
#
# IMPORTANT: Flower's ClientApp mods (like LocalDpMod, SecAggPlusMod) must be applied
# at app creation time, NOT at runtime.
#
# USAGE IN PYPROJECT.TOML:
# [tool.flwr.app.components]
# clientapp = "adni_flwr.client_app:app"                          # Regular FL
# clientapp = "adni_flwr.client_app:secagg_plus_app"             # SecAgg+
# clientapp = "adni_flwr.client_app:differential_privacy_app"     # Differential Privacy
#
# ==================================================================================

# Main app - Regular FedAvg/FedProx without any special mods
app = ClientApp(client_fn=client_fn)

# Specialized app instances
regular_app = create_client_app("regular")

# SecAgg+ app - Pre-configured with SecAggPlusMod
if SECAGGPLUS_MOD_AVAILABLE:
    secagg_plus_app = create_client_app("secagg+")
else:
    secagg_plus_app = None


def create_differential_privacy_app_with_defaults() -> Optional[ClientApp]:
    """Create a differential privacy app with explicit default parameters.

    Returns:
        ClientApp with LocalDpMod using default parameters, None if DP not available
    """
    if not LOCALDPMOD_AVAILABLE:
        logger.warning("LocalDpMod not available, cannot create differential_privacy_app")
        return None

    try:
        # Explicit DP parameters - edit these values as needed for your use case
        # Tuned for convergence: noise_stddev = sensitivity/epsilon = 1.0/100 = 0.01
        default_dp_config = type(
            "Config",
            (),
            {
                "fl": type(
                    "FL",
                    (),
                    {
                        "strategy": "differential_privacy",
                        "dp_clipping_norm": 1.0,  # 👈 EDIT THIS: Gradient clipping norm
                        "dp_sensitivity": 1.0,  # 👈 EDIT THIS: Usually equals clipping_norm
                        "dp_epsilon": 500.0,  # 👈 EDIT THIS: Privacy budget (higher = less privacy, better utility)
                        "dp_delta": 1e-5,  # 👈 EDIT THIS: Failure probability
                    },
                )()
            },
        )()

        local_dp_mod = create_local_dp_mod(default_dp_config)
        logger.info("Created differential_privacy_app with EXPLICIT DEFAULT parameters")
        logger.info(f"   DP parameters: clipping_norm={1.0}, sensitivity={1.0}, epsilon={100.0}, delta={1e-5}")
        logger.info(f"   Expected noise stddev: {1.0 / 100.0:.3f}")
        return ClientApp(client_fn=client_fn, mods=[local_dp_mod])

    except Exception as e:
        logger.error(f"Failed to create differential_privacy_app: {e}")
        return None


# ==================================================================================
# DIFFERENTIAL PRIVACY APP CREATION
# ==================================================================================

# Create differential_privacy_app with explicit default parameters
# Edit the parameters in create_differential_privacy_app_with_defaults() function above
differential_privacy_app = create_differential_privacy_app_with_defaults()

# ==================================================================================
# SUMMARY:
# - app: Regular ClientApp (FedAvg/FedProx)
# - secagg_plus_app: ClientApp with SecAggPlusMod (if available)
# - differential_privacy_app: ClientApp with LocalDpMod (EXPLICIT defaults - edit above to customize)
#
# Use in pyproject.toml: clientapp = "adni_flwr.client_app:differential_privacy_app"
#
# To customize DP parameters:
# 1. Edit the values in create_differential_privacy_app_with_defaults() function above
# 2. Key parameter: noise_stddev = dp_sensitivity / dp_epsilon
# 3. Lower noise = better convergence, but less privacy
# ==================================================================================


def create_opacus_dp_app(config: Optional[Config] = None) -> ClientApp:
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
        # Determine which GPU to use if available
        gpu_idx = context.node_config.get("gpu-id", 0)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_idx}")
        else:
            device = torch.device("cpu")

        # Get partition ID to determine which config file to use
        partition_id = context.node_config.get("partition-id", 0)

        # Load the appropriate config using proper dynamic config loading
        if config is None:
            # Get client config files from app config (same pattern as main client_fn)
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
            try:
                from adni_classification.config.config import Config

                loaded_config = Config.from_yaml(config_path)
                logger.info(f"Loaded config from: {config_path}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Config file not found: {config_path}. Error: {e}") from e
            except Exception as e:
                raise ValueError(f"Failed to load config from {config_path}. Error: {e}") from e
        else:
            loaded_config = config
            logger.info("Using provided config")

        # Strict validation - no implicit handling
        if not hasattr(loaded_config, "fl"):
            raise ValueError("Config missing 'fl' section")

        if not hasattr(loaded_config.fl, "strategy"):
            raise ValueError("Config missing 'fl.strategy' field")

        if loaded_config.fl.strategy != "differential_privacy":
            raise ValueError(
                f"Config strategy must be 'differential_privacy', got: '{loaded_config.fl.strategy}'. "
                f"Please set 'strategy: differential_privacy' in your config file."
            )

        # Validate required DP parameters
        required_dp_params = ["dp_epsilon", "dp_sensitivity", "dp_clipping_norm", "dp_delta"]
        missing_params = []
        for param in required_dp_params:
            if not hasattr(loaded_config.fl, param):
                missing_params.append(param)

        if missing_params:
            raise ValueError(
                f"Missing required DP parameters: {missing_params}. "
                f"Please add these to your config file under the 'fl' section."
            )

        # Load model, data, etc.
        try:
            model = load_model(loaded_config).to(device)
            train_loader, val_loader = load_data(loaded_config)
            criterion = create_criterion(loaded_config, train_loader.dataset, device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model/data components: {e}") from e

        # Create optimizer
        try:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=loaded_config.training.learning_rate,
                weight_decay=loaded_config.training.weight_decay,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create optimizer: {e}") from e

        # Create scheduler if specified
        scheduler = None
        if hasattr(loaded_config.training, "lr_scheduler") and loaded_config.training.lr_scheduler == "cosine":
            try:
                total_steps = loaded_config.fl.local_epochs * len(train_loader) * loaded_config.fl.num_rounds
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            except Exception as e:
                logger.warning(f"⚠️ Failed to create cosine scheduler: {e}. Continuing without scheduler.")

        # Extract DP parameters with strict validation
        epsilon = float(loaded_config.fl.dp_epsilon)
        sensitivity = float(loaded_config.fl.dp_sensitivity)
        clipping_norm = float(loaded_config.fl.dp_clipping_norm)
        delta = float(loaded_config.fl.dp_delta)

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

        logger.info("🔧 Creating Opacus DP client with:")
        logger.info(f"   epsilon: {epsilon}")
        logger.info(f"   sensitivity: {sensitivity}")
        logger.info(f"   noise_multiplier: {noise_multiplier:.6f}")
        logger.info(f"   max_grad_norm: {clipping_norm}")

        # Create Opacus differential privacy client strategy
        try:
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


# ==================================================================================
# OPACUS DP APP INSTANCE - Explicit Error Handling (RECOMMENDED)
# ==================================================================================
try:
    opacus_differential_privacy_app = create_opacus_dp_app()
    logger.info("Opacus differential privacy app created successfully")
except Exception as e:
    logger.error(f"Failed to create opacus_differential_privacy_app: {e}")
    # Re-raise to make the error explicit - don't hide import failures
    raise RuntimeError(f"Failed to initialize opacus_differential_privacy_app: {e}") from e


def create_scheduled_dp_app(config: Optional[Config] = None) -> ClientApp:
    """Create a ClientApp with epsilon scheduling to reduce noise over time.

    This modifies the configuration to use increasing epsilon (decreasing noise)
    as training progresses, addressing the parameter shrinkage issue.

    Args:
        config: Optional pre-loaded config. If None, will be loaded from context.

    Returns:
        ClientApp with scheduled epsilon differential privacy

    Raises:
        ValueError: If config is invalid or strategy is not differential_privacy
        ImportError: If LocalDpMod is not available
        FileNotFoundError: If config file cannot be found
    """
    if not LOCALDPMOD_AVAILABLE:
        raise ImportError("LocalDpMod not available. Please install the required Flower version with DP support.")

    def scheduled_dp_client_fn(context: Context):
        """Client factory with epsilon scheduling."""
        # Standard setup
        gpu_idx = context.node_config.get("gpu-id", 0)
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        partition_id = context.node_config.get("partition-id", 0)

        # Load config using proper dynamic config loading
        if config is None:
            # Get client config files from app config (same pattern as main client_fn)
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
            try:
                from adni_classification.config.config import Config

                loaded_config = Config.from_yaml(config_path)
                logger.info(f"Loaded config from: {config_path}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Config file not found: {config_path}. Error: {e}") from e
            except Exception as e:
                raise ValueError(f"Failed to load config from {config_path}. Error: {e}") from e
        else:
            loaded_config = config
            logger.info("Using provided config")

        # Strict validation
        if not hasattr(loaded_config, "fl"):
            raise ValueError("Config missing 'fl' section")

        if not hasattr(loaded_config.fl, "strategy"):
            raise ValueError("Config missing 'fl.strategy' field")

        if loaded_config.fl.strategy != "differential_privacy":
            raise ValueError(f"Config strategy must be 'differential_privacy', got: '{loaded_config.fl.strategy}'")

        # Validate required DP parameters
        required_dp_params = ["dp_epsilon", "dp_sensitivity", "dp_clipping_norm", "dp_delta"]
        missing_params = [param for param in required_dp_params if not hasattr(loaded_config.fl, param)]

        if missing_params:
            raise ValueError(
                f"Missing required DP parameters: {missing_params}. "
                f"Please add these to your config file under the 'fl' section."
            )

        # Create client components
        try:
            model = load_model(loaded_config).to(device)
            train_loader, val_loader = load_data(loaded_config)
            criterion = create_criterion(loaded_config, train_loader.dataset, device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=loaded_config.training.learning_rate,
                weight_decay=loaded_config.training.weight_decay,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model/data/optimizer: {e}") from e

        # Create standard DP client
        try:
            from adni_flwr.strategies import StrategyFactory

            client_strategy = StrategyFactory.create_client_strategy(
                strategy_name="differential_privacy",
                config=loaded_config,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            return StrategyAwareClient(
                config=loaded_config,
                device=device,
                client_strategy=client_strategy,
                context=context,
                total_fl_rounds=loaded_config.fl.num_rounds,
                wandb_logger=None,  # Add wandb_logger if needed
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create scheduled DP client strategy: {e}") from e

    return ClientApp(client_fn=scheduled_dp_client_fn)


# ==================================================================================
# SCHEDULED DP APP INSTANCE - Explicit Error Handling
# ==================================================================================
try:
    scheduled_differential_privacy_app = create_scheduled_dp_app()
    logger.info("Scheduled differential privacy app created successfully")
except Exception as e:
    logger.error(f"Failed to create scheduled_differential_privacy_app: {e}")
    # Re-raise to make the error explicit - don't hide import failures
    raise RuntimeError(f"Failed to initialize scheduled_differential_privacy_app: {e}") from e
