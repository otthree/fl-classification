"""Server application for ADNI Federated Learning - Refactored Version."""

from flwr.server import ServerApp
from loguru import logger

from adni_flwr.apps.server_apps import (
    ServerAppFactory,
    auto_detecting_server_app,
    regular_server_app,
    secagg_plus_server_app,
)

# =============================================================================
# MAIN SERVER APP - Regular FedAvg/FedProx/SecAgg (NOT SecAgg+)
# =============================================================================
app = regular_server_app

# =============================================================================
# SPECIALIZED SERVER APP INSTANCES
# =============================================================================

# Regular server app for FedAvg, FedProx, SecAgg strategies
regular_app = regular_server_app

# SecAgg+ server app with workflow-based execution
secagg_plus_app = secagg_plus_server_app

# Auto-detecting server app that chooses the right execution pattern
auto_app = auto_detecting_server_app

# =============================================================================
# FACTORY FUNCTIONS FOR CUSTOM APP CREATION
# =============================================================================


def create_server_app(strategy_type: str) -> ServerApp:
    """Create a ServerApp based on strategy type.

    Args:
        strategy_type: Type of strategy ("regular", "secagg+", "auto")

    Returns:
        ServerApp instance

    Raises:
        ValueError: If invalid strategy type provided
    """
    if strategy_type == "regular":
        return ServerAppFactory.create_regular_server_app()
    elif strategy_type == "secagg+":
        return ServerAppFactory.create_secagg_plus_server_app()
    elif strategy_type == "auto":
        return ServerAppFactory.create_auto_detecting_server_app()
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def create_regular_server_app() -> ServerApp:
    """Create a regular server app for FedAvg/FedProx/SecAgg strategies.

    Returns:
        ServerApp with regular server function
    """
    return ServerAppFactory.create_regular_server_app()


def create_secagg_plus_server_app() -> ServerApp:
    """Create a SecAgg+ server app with workflow-based execution.

    Returns:
        ServerApp with SecAgg+ main function
    """
    return ServerAppFactory.create_secagg_plus_server_app()


def create_auto_detecting_server_app() -> ServerApp:
    """Create an auto-detecting server app.

    Returns:
        ServerApp that automatically detects and executes the appropriate strategy
    """
    return ServerAppFactory.create_auto_detecting_server_app()


# =============================================================================
# SUMMARY OF AVAILABLE APPS
# =============================================================================
# Usage in pyproject.toml:
# [tool.flwr.app.components]
# serverapp = "adni_flwr.server_app:app"                    # Regular FL (FedAvg/FedProx/SecAgg)
# serverapp = "adni_flwr.server_app:secagg_plus_app"       # SecAgg+ with workflow
# serverapp = "adni_flwr.server_app:auto_app"              # Auto-detecting
#
# Supported Strategies:
# - Regular execution: fedavg, fedprox, secagg
# - Workflow execution: secagg+ (requires dedicated app or auto_app)
# - Auto-detection: Automatically chooses execution pattern based on config
# =============================================================================

logger.info("✅ Server app module loaded successfully with all variants")
logger.info("📝 Available server apps: regular, secagg+, auto_detecting")
