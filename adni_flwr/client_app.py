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
            raise ValueError(
                "❌ CRITICAL ERROR: Differential Privacy strategy requires a config to create LocalDpMod. "
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
            f"❌ CRITICAL ERROR: Differential Privacy parameters must be numeric. "
            f"Raw values: clipping_norm={clipping_norm} ({type(clipping_norm)}), "
            f"sensitivity={sensitivity} ({type(sensitivity)}), "
            f"epsilon={epsilon} ({type(epsilon)}), "
            f"delta={delta} ({type(delta)}). "
            f"Error: {e}"
        ) from e

    # Validate parameter ranges for LocalDpMod
    if clipping_norm <= 0:
        raise RuntimeError(f"❌ CRITICAL ERROR: LocalDpMod clipping_norm must be positive, got: {clipping_norm}")
    if sensitivity <= 0:
        raise RuntimeError(f"❌ CRITICAL ERROR: LocalDpMod sensitivity must be positive, got: {sensitivity}")
    if epsilon <= 0:
        raise RuntimeError(f"❌ CRITICAL ERROR: LocalDpMod epsilon must be positive, got: {epsilon}")
    if delta <= 0 or delta >= 1:
        raise RuntimeError(f"❌ CRITICAL ERROR: LocalDpMod delta must be between 0 and 1 (exclusive), got: {delta}")

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
            "❌ CRITICAL ERROR: Differential Privacy requires LocalDpMod which is not available. "
            "Please ensure you have the correct Flower version with LocalDpMod support."
        )

    # Validate that this is actually a DP config
    if not (hasattr(config.fl, "strategy") and config.fl.strategy == "differential_privacy"):
        raise ValueError(
            f"❌ CRITICAL ERROR: Config strategy is '{getattr(config.fl, 'strategy', 'not specified')}' "
            f"but expected 'differential_privacy'."
        )

    # Create LocalDpMod with the provided config
    local_dp_mod = create_local_dp_mod(config)
    logger.info("✅ Created DP ClientApp with LocalDpMod using provided config")

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
                        "dp_epsilon": 100.0,  # 👈 EDIT THIS: Privacy budget (higher = less privacy, better utility)
                        "dp_delta": 1e-5,  # 👈 EDIT THIS: Failure probability
                    },
                )()
            },
        )()

        local_dp_mod = create_local_dp_mod(default_dp_config)
        logger.info("✅ Created differential_privacy_app with EXPLICIT DEFAULT parameters")
        logger.info(f"   DP parameters: clipping_norm={1.0}, sensitivity={1.0}, epsilon={100.0}, delta={1e-5}")
        logger.info(f"   Expected noise stddev: {1.0/100.0:.3f}")
        return ClientApp(client_fn=client_fn, mods=[local_dp_mod])

    except Exception as e:
        logger.error(f"❌ Failed to create differential_privacy_app: {e}")
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
