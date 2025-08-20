"""Client application factories for all supported FL strategies."""

from typing import Optional

from flwr.client import ClientApp
from flwr.common import Context
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.apps.base import BaseAppFactory, ClientAppFactoryMixin, DifferentialPrivacyMixin
from adni_flwr.apps.differential_privacy import AdaptiveLocalDpMod, OpacusDPClientFactory
from adni_flwr.common import StrategyDetector
from adni_flwr.strategies import StrategyAwareClient, StrategyFactory
from adni_flwr.utils import AppUtils
from adni_flwr.utils.logging_config import setup_fl_logging
from adni_flwr.utils.wandb_logger import FLClientWandbLogger


class ClientAppFactory(BaseAppFactory, ClientAppFactoryMixin, DifferentialPrivacyMixin):
    """Factory for creating client applications with different FL strategies."""

    @staticmethod
    def create_regular_client_app() -> ClientApp:
        """Create a regular client app for FedAvg/FedProx strategies.

        Returns:
            ClientApp with regular client function
        """
        BaseAppFactory._log_app_creation("Client", "regular")
        return ClientApp(client_fn=ClientAppFactory._create_client_fn())

    @staticmethod
    def create_secagg_plus_client_app() -> ClientApp:
        """Create a SecAgg+ client app with appropriate mods.

        Returns:
            ClientApp with SecAgg+ mods

        Raises:
            ValueError: If SecAgg+ requirements are not met
        """
        # Validate requirements
        StrategyDetector.validate_strategy_requirements("secagg+")
        mods = StrategyDetector.get_strategy_mods("secagg+")

        BaseAppFactory._log_app_creation("Client", "secagg+", "with secaggplus_mod")
        return ClientApp(client_fn=ClientAppFactory._create_client_fn(), mods=mods)

    @staticmethod
    def create_dp_client_app(config: Config) -> ClientApp:
        """Create a differential privacy client app with LocalDpMod.

        Args:
            config: Configuration object containing DP parameters

        Returns:
            ClientApp with LocalDpMod properly configured

        Raises:
            ValueError: If DP configuration is invalid
            RuntimeError: If LocalDpMod is not available
        """
        # Validate DP configuration
        ClientAppFactory._validate_dp_config(config, "provided config")

        # Get DP mods
        mods = StrategyDetector.get_strategy_mods("differential_privacy", config)

        BaseAppFactory._log_app_creation("Client", "differential_privacy", "with LocalDpMod")
        return ClientApp(client_fn=ClientAppFactory._create_client_fn(), mods=mods)

    @staticmethod
    def create_adaptive_dp_client_app(config: Optional[Config] = None) -> ClientApp:
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
            try:
                # Create common client components
                device, partition_id, loaded_config, client_id = ClientAppFactory._create_client_context_components(
                    context, config
                )

                # Setup FL logging with client-specific configuration
                setup_fl_logging(client_id=client_id)

                # Validate DP configuration
                ClientAppFactory._validate_dp_config(loaded_config, f"client partition {partition_id} config")

                # Create standard client components
                model, optimizer, criterion, train_loader, val_loader = AppUtils.create_model_components(
                    loaded_config, device
                )

                # Create client strategy
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
            adaptive_dp_mod = ClientAppFactory._create_adaptive_dp_mod_from_config(config)
        else:
            adaptive_dp_mod = ClientAppFactory._create_adaptive_dp_mod_with_defaults()

        BaseAppFactory._log_app_creation("Client", "adaptive_differential_privacy", "with AdaptiveLocalDpMod")
        return ClientApp(client_fn=adaptive_dp_client_fn, mods=[adaptive_dp_mod])

    @staticmethod
    def create_opacus_dp_client_app(config: Optional[Config] = None) -> ClientApp:
        """Create Opacus-based differential privacy client app.

        Args:
            config: Optional pre-loaded config. If None, will be loaded from context.

        Returns:
            ClientApp with Opacus DP strategy

        Raises:
            ValueError: If config is invalid
            RuntimeError: If client creation fails
        """
        BaseAppFactory._log_app_creation("Client", "opacus_differential_privacy", "with gradient-level DP")
        return OpacusDPClientFactory.create_opacus_dp_app(config)

    @staticmethod
    def _create_client_fn():
        """Create the main client factory function.

        Returns:
            Client factory function
        """

        def client_fn(context: Context):
            """Client factory function.

            Args:
                context: Context containing client configuration

            Returns:
                An instance of NumPyClient
            """
            try:
                # Create common client components
                device, partition_id, config, client_id = ClientAppFactory._create_client_context_components(context)

                # Setup FL logging with client-specific configuration
                setup_fl_logging(client_id=client_id)

                # Validate strategy configuration
                strategy_name = ClientAppFactory._validate_strategy_config(config)

                # Initialize client-side WandB logger
                client_wandb_logger = FLClientWandbLogger(config, client_id)
                client_wandb_logger.init_wandb()

                # Log initialization
                AppUtils.log_strategy_initialization(strategy_name, client_id, f"partition {partition_id}", device)
                AppUtils.log_fl_rounds(config.fl.num_rounds)

                # Create model components
                model, optimizer, criterion, train_loader, val_loader = AppUtils.create_model_components(config, device)

                # Create client strategy
                client_strategy = StrategyFactory.create_client_strategy(
                    strategy_name=strategy_name,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    scheduler=None,  # Scheduler managed via Context
                )

                # Create strategy-aware client
                client = StrategyAwareClient(
                    config=config,
                    device=device,
                    client_strategy=client_strategy,
                    context=context,
                    total_fl_rounds=config.fl.num_rounds,
                    wandb_logger=client_wandb_logger,
                )

                logger.info(f"✅ Client {partition_id} initialized with Context-based scheduler management")
                return client.to_client()

            except Exception as e:
                raise RuntimeError(f"Failed to create client: {e}") from e

        return client_fn

    @staticmethod
    def _create_adaptive_dp_mod_from_config(config: Config) -> AdaptiveLocalDpMod:
        """Create AdaptiveLocalDpMod from config parameters.

        Args:
            config: Configuration with DP parameters

        Returns:
            AdaptiveLocalDpMod instance
        """
        clipping_norm = float(config.fl.dp_clipping_norm)
        sensitivity = float(config.fl.dp_sensitivity)
        initial_epsilon = float(config.fl.dp_epsilon)
        delta = float(config.fl.dp_delta)
        decay_factor = getattr(config.fl, "dp_decay_factor", 0.95)
        min_epsilon = getattr(config.fl, "dp_min_epsilon", None)
        use_gaussian_mechanism = getattr(config.fl, "dp_use_gaussian_mechanism", True)

        logger.info("🔧 Creating AdaptiveLocalDpMod with config parameters:")
        logger.info(f"   clipping_norm: {clipping_norm}")
        logger.info(f"   sensitivity: {sensitivity}")
        logger.info(f"   initial_epsilon: {initial_epsilon}")
        logger.info(f"   delta: {delta}")
        logger.info(f"   decay_factor: {decay_factor}")
        logger.info(f"   min_epsilon: {min_epsilon}")
        logger.info(f"   use_gaussian_mechanism: {use_gaussian_mechanism}")

        return AdaptiveLocalDpMod(
            clipping_norm=clipping_norm,
            sensitivity=sensitivity,
            initial_epsilon=initial_epsilon,
            delta=delta,
            decay_factor=decay_factor,
            min_epsilon=min_epsilon,
            use_gaussian_mechanism=use_gaussian_mechanism,
        )

    @staticmethod
    def _create_adaptive_dp_mod_with_defaults() -> AdaptiveLocalDpMod:
        """Create AdaptiveLocalDpMod with default parameters.

        Returns:
            AdaptiveLocalDpMod instance with default parameters
        """
        adaptive_dp_mod = AdaptiveLocalDpMod(
            clipping_norm=1.0,  # Default value
            sensitivity=1.0,  # Default value
            initial_epsilon=1000.0,  # Default value - high epsilon for less noise
            delta=1e-5,  # Default value
            decay_factor=0.95,  # Default decay factor
            min_epsilon=10.0,  # Default minimum epsilon
            use_gaussian_mechanism=False,  # Default value
        )

        logger.info("🔧 Created AdaptiveLocalDpMod with default parameters:")
        logger.info(f"   clipping_norm: {adaptive_dp_mod.clipping_norm}")
        logger.info(f"   sensitivity: {adaptive_dp_mod.sensitivity}")
        logger.info(f"   initial_epsilon: {adaptive_dp_mod.initial_epsilon}")
        logger.info(f"   delta: {adaptive_dp_mod.delta}")
        logger.info(f"   decay_factor: {adaptive_dp_mod.decay_factor}")
        logger.info(f"   min_epsilon: {adaptive_dp_mod.min_epsilon}")
        return adaptive_dp_mod


# =============================================================================
# APP INSTANCES WITH ERROR HANDLING
# =============================================================================


def _create_app_with_error_handling(creation_func, app_name: str, *args, **kwargs):
    """Create app with consistent error handling."""
    try:
        app = creation_func(*args, **kwargs)
        logger.success(f"✅ {app_name} created successfully")
        return app
    except Exception as e:
        logger.error(f"❌ Failed to create {app_name}: {e}")
        # Re-raise to make the error explicit - don't hide import failures
        raise RuntimeError(f"Failed to initialize {app_name}: {e}") from e


# Create client app instances
regular_client_app = _create_app_with_error_handling(ClientAppFactory.create_regular_client_app, "regular_client_app")

# SecAgg+ app (conditional creation)
secagg_plus_client_app = None
try:
    secagg_plus_client_app = _create_app_with_error_handling(
        ClientAppFactory.create_secagg_plus_client_app, "secagg_plus_client_app"
    )
except Exception:
    logger.warning("⚠️ SecAgg+ client app not available - secaggplus_mod not found")

# Differential privacy apps
differential_privacy_client_app = None
try:
    # Create with default parameters
    default_dp_config = ClientAppFactory._create_dp_default_config()
    differential_privacy_client_app = _create_app_with_error_handling(
        ClientAppFactory.create_dp_client_app, "differential_privacy_client_app", default_dp_config
    )
except Exception:
    logger.warning("⚠️ Differential privacy client app not available - LocalDpMod not found")

# Adaptive DP app
adaptive_differential_privacy_client_app = _create_app_with_error_handling(
    ClientAppFactory.create_adaptive_dp_client_app, "adaptive_differential_privacy_client_app"
)

# Opacus DP app
opacus_differential_privacy_client_app = _create_app_with_error_handling(
    ClientAppFactory.create_opacus_dp_client_app, "opacus_differential_privacy_client_app"
)
