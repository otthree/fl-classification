"""Client application for ADNI Federated Learning - Refactored Version."""

from typing import Optional

from flwr.client import ClientApp
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.apps.client_apps import (
    ClientAppFactory,
    adaptive_differential_privacy_client_app,
    differential_privacy_client_app,
    opacus_differential_privacy_client_app,
    regular_client_app,
    secagg_plus_client_app,
)

# =============================================================================
# MAIN CLIENT APP - Regular FedAvg/FedProx without any special mods
# =============================================================================
app = regular_client_app

# =============================================================================
# SPECIALIZED CLIENT APP INSTANCES
# =============================================================================

# Regular app
regular_app = regular_client_app

# SecAgg+ app (may be None if not available)
secagg_plus_app = secagg_plus_client_app

# Differential Privacy apps
differential_privacy_app = differential_privacy_client_app
adaptive_differential_privacy_app = adaptive_differential_privacy_client_app
opacus_differential_privacy_app = opacus_differential_privacy_client_app

# =============================================================================
# FACTORY FUNCTIONS FOR CUSTOM APP CREATION
# =============================================================================


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
    if strategy_type == "regular":
        return ClientAppFactory.create_regular_client_app()
    elif strategy_type == "secagg+":
        return ClientAppFactory.create_secagg_plus_client_app()
    elif strategy_type == "differential_privacy":
        if config is None:
            raise ValueError(
                "CRITICAL ERROR: Differential Privacy strategy requires a config to create LocalDpMod. "
                "Use 'create_dp_client_app(config)' function instead."
            )
        return ClientAppFactory.create_dp_client_app(config)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def create_dp_client_app(config: Config) -> ClientApp:
    """Create a ClientApp with LocalDpMod configured for differential privacy.

    Args:
        config: Configuration object containing DP parameters

    Returns:
        ClientApp with LocalDpMod properly configured

    Raises:
        RuntimeError: If LocalDpMod is not available or config parameters are invalid
    """
    return ClientAppFactory.create_dp_client_app(config)


def create_adaptive_dp_app(config: Optional[Config] = None) -> ClientApp:
    """Create a ClientApp with adaptive LocalDpMod that reduces noise over time.

    Args:
        config: Optional pre-loaded config. If None, will be loaded from context.

    Returns:
        ClientApp with adaptive DP mod

    Raises:
        ValueError: If config is invalid or strategy is not differential_privacy
        FileNotFoundError: If config file cannot be found
    """
    return ClientAppFactory.create_adaptive_dp_client_app(config)


def create_opacus_dp_app(config: Optional[Config] = None) -> ClientApp:
    """Create a ClientApp using Opacus-based gradient-level DP.

    Args:
        config: Optional pre-loaded config. If None, will be loaded from context.

    Returns:
        ClientApp with Opacus DP strategy

    Raises:
        ValueError: If config is invalid or strategy is not differential_privacy
        FileNotFoundError: If config file cannot be found
        RuntimeError: If differential privacy client creation fails
    """
    return ClientAppFactory.create_opacus_dp_client_app(config)


# =============================================================================
# SUMMARY OF AVAILABLE APPS
# =============================================================================
# Usage in pyproject.toml:
# [tool.flwr.app.components]
# clientapp = "adni_flwr.client_app:app"                              # Regular FL
# clientapp = "adni_flwr.client_app:secagg_plus_app"                 # SecAgg+
# clientapp = "adni_flwr.client_app:differential_privacy_app"         # DP with LocalDpMod
# clientapp = "adni_flwr.client_app:adaptive_differential_privacy_app" # Adaptive DP
# clientapp = "adni_flwr.client_app:opacus_differential_privacy_app"   # Opacus DP
# =============================================================================

logger.info("✅ Client app module loaded successfully with all variants")
logger.info("📝 Available client apps: regular, secagg+, differential_privacy, adaptive_dp, opacus_dp")
