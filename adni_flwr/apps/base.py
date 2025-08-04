"""Base application factory patterns shared by client and server apps."""

from typing import Optional, Union

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.common import ConfigLoader, StrategyDetector
from adni_flwr.utils import AppUtils, DeviceManager


class BaseAppFactory:
    """Base class for app factories with common patterns."""

    @staticmethod
    def _log_app_creation(app_type: str, strategy_name: str, additional_info: str = "") -> None:
        """Log app creation with consistent formatting.

        Args:
            app_type: Type of app being created (e.g., "Client", "Server")
            strategy_name: Strategy name
            additional_info: Additional information to log
        """
        if strategy_name.lower() in ["secagg+", "secaggplus"]:
            logger.info(f"🔒 Creating {app_type} app with SecAgg+ strategy {additional_info}")
        elif strategy_name.lower() == "differential_privacy":
            logger.info(f"🔒 Creating {app_type} app with Differential Privacy strategy {additional_info}")
        else:
            logger.info(f"📊 Creating {app_type} app with {strategy_name} strategy {additional_info}")

    @staticmethod
    def _validate_strategy_config(config: Config, expected_strategy: Optional[str] = None) -> str:
        """Validate strategy configuration with optional specific strategy check.

        Args:
            config: Configuration object
            expected_strategy: Expected strategy name (optional)

        Returns:
            Validated strategy name

        Raises:
            ValueError: If strategy validation fails
        """
        ConfigLoader.validate_fl_section(config, "config")
        strategy_name = config.fl.strategy

        if expected_strategy and strategy_name != expected_strategy:
            raise ValueError(f"Expected strategy '{expected_strategy}', got '{strategy_name}'")

        StrategyDetector.validate_strategy_requirements(strategy_name)
        return strategy_name

    @staticmethod
    def _handle_app_creation_error(error: Exception, app_type: str, strategy_name: str) -> None:
        """Handle app creation errors with consistent logging and re-raising.

        Args:
            error: Exception that occurred
            app_type: Type of app being created
            strategy_name: Strategy name

        Raises:
            RuntimeError: Always re-raises with additional context
        """
        logger.error(f"❌ Failed to create {app_type} app with {strategy_name} strategy: {error}")
        raise RuntimeError(f"Failed to initialize {app_type.lower()}_{strategy_name}_app: {error}") from error


class AppCreationMixin:
    """Mixin providing common app creation utilities."""

    @staticmethod
    def create_app_with_error_handling(
        app_creation_func: callable, app_name: str, *args, **kwargs
    ) -> Union[ClientApp, ServerApp, None]:
        """Create app with consistent error handling.

        Args:
            app_creation_func: Function to create the app
            app_name: Name of the app for logging
            *args: Arguments for app creation function
            **kwargs: Keyword arguments for app creation function

        Returns:
            Created app or None if creation fails

        Raises:
            RuntimeError: If app creation fails and should be re-raised
        """
        try:
            app = app_creation_func(*args, **kwargs)
            logger.success(f"✅ {app_name} created successfully")
            return app
        except Exception as e:
            logger.error(f"❌ Failed to create {app_name}: {e}")
            # Re-raise to make the error explicit - don't hide initialization failures
            raise RuntimeError(f"Failed to initialize {app_name}: {e}") from e


class ClientAppFactoryMixin(AppCreationMixin):
    """Mixin providing client-specific app creation utilities."""

    @staticmethod
    def _create_client_context_components(context: Context, config: Optional[Config] = None):
        """Create common client components from context.

        Args:
            context: Flower context
            config: Optional pre-loaded config

        Returns:
            Tuple of (device, partition_id, config, client_id)
        """
        # Get device and partition ID
        device = DeviceManager.get_device(context)
        partition_id = DeviceManager.get_partition_id(context)

        # Load config if not provided
        if config is None:
            config = ConfigLoader.load_client_config(context, partition_id)

        # Get client ID
        client_id = AppUtils.get_client_id(config, partition_id)

        return device, partition_id, config, client_id


class ServerAppFactoryMixin(AppCreationMixin):
    """Mixin providing server-specific app creation utilities."""

    @staticmethod
    def _create_server_context_components(context: Context, config: Optional[Config] = None):
        """Create common server components from context.

        Args:
            context: Flower context
            config: Optional pre-loaded config

        Returns:
            Tuple of (config,)
        """
        # Load config if not provided
        if config is None:
            config = ConfigLoader.load_server_config(context)

        return (config,)


class DifferentialPrivacyMixin:
    """Mixin providing differential privacy utilities."""

    @staticmethod
    def _validate_dp_config(config: Config, config_description: str) -> None:
        """Validate differential privacy configuration.

        Args:
            config: Configuration object
            config_description: Description for error messages

        Raises:
            ValueError: If DP configuration is invalid
        """
        ConfigLoader.validate_dp_config(config, config_description)

    @staticmethod
    def _create_dp_default_config() -> Config:
        """Create a default config object for DP with explicit parameters.

        Returns:
            Config object with default DP parameters
        """
        # Create explicit default DP parameters
        default_dp_config = type(
            "Config",
            (),
            {
                "fl": type(
                    "FL",
                    (),
                    {
                        "strategy": "differential_privacy",
                        "dp_clipping_norm": 1.0,  # Gradient clipping norm
                        "dp_sensitivity": 1.0,  # Usually equals clipping_norm
                        "dp_epsilon": 50.0,  # Privacy budget (higher = less privacy, better utility)
                        "dp_delta": 1e-5,  # Failure probability
                        "dp_decay_factor": 0.95,  # For adaptive DP
                        "dp_min_epsilon": 10.0,  # Minimum epsilon for adaptive DP
                    },
                )()
            },
        )()

        logger.info("🔧 Created default DP config with explicit parameters:")
        logger.info(f"   dp_clipping_norm: {default_dp_config.fl.dp_clipping_norm}")
        logger.info(f"   dp_sensitivity: {default_dp_config.fl.dp_sensitivity}")
        logger.info(f"   dp_epsilon: {default_dp_config.fl.dp_epsilon}")
        logger.info(f"   dp_delta: {default_dp_config.fl.dp_delta}")
        logger.info(f"   dp_decay_factor: {default_dp_config.fl.dp_decay_factor}")
        logger.info(f"   dp_min_epsilon: {default_dp_config.fl.dp_min_epsilon}")

        return default_dp_config
