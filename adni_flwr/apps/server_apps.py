"""Server application factories for all supported FL strategies."""

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.workflow import DefaultWorkflow
from loguru import logger

from adni_classification.config.config import Config
from adni_flwr.apps.base import BaseAppFactory, ServerAppFactoryMixin
from adni_flwr.common import StrategyDetector
from adni_flwr.server_fn import safe_weighted_average
from adni_flwr.strategies import StrategyFactory
from adni_flwr.task import debug_model_architecture, get_params, load_model
from adni_flwr.utils.wandb_logger import FLServerWandbLogger


class ServerAppFactory(BaseAppFactory, ServerAppFactoryMixin):
    """Factory for creating server applications with different FL strategies."""

    @staticmethod
    def create_regular_server_app() -> ServerApp:
        """Create a regular server app for FedAvg/FedProx/SecAgg strategies.

        Returns:
            ServerApp with regular server function
        """
        BaseAppFactory._log_app_creation("Server", "regular")
        return ServerApp(server_fn=ServerAppFactory._create_server_fn())

    @staticmethod
    def create_secagg_plus_server_app() -> ServerApp:
        """Create a SecAgg+ server app with workflow-based execution.

        Returns:
            ServerApp with SecAgg+ main function
        """
        secagg_plus_app = ServerApp()

        @secagg_plus_app.main()
        def main(grid: Grid, context: Context):
            """Main entry point for SecAgg+ server app using proper Flower workflow pattern."""
            logger.info("🔒 SecAgg+ Server - Starting workflow-based execution")
            ServerAppFactory._execute_secagg_plus_workflow(grid, context)

        BaseAppFactory._log_app_creation("Server", "secagg+", "with workflow execution")
        return secagg_plus_app

    @staticmethod
    def create_auto_detecting_server_app() -> ServerApp:
        """Create an auto-detecting server app that chooses the right execution pattern.

        Returns:
            ServerApp that automatically detects and executes the appropriate strategy
        """
        auto_app = ServerApp()

        @auto_app.main()
        def auto_main(grid: Grid, context: Context):
            """Auto-detecting main entry point that chooses the right execution pattern."""
            try:
                # Load config
                (config,) = ServerAppFactory._create_server_context_components(context)

                # Validate and get strategy
                strategy_name = ServerAppFactory._validate_strategy_config(config)
                logger.info(f"🔍 Auto-detected strategy: {strategy_name}")

                if StrategyDetector.is_secagg_plus_strategy(strategy_name):
                    logger.info("🔒 Using SecAgg+ workflow execution")
                    ServerAppFactory._execute_secagg_plus_workflow(grid, context, config)
                else:
                    logger.error("📊 Regular strategies require the standard ServerApp pattern")
                    logger.error("Use: flower-server-app adni_flwr.server_app:app")
                    raise ValueError(
                        f"Regular strategy '{strategy_name}' cannot be used with auto_app.\n"
                        f"Regular strategies (fedavg, fedprox, secagg) require the standard ServerApp pattern.\n"
                        f"Use: flower-server-app adni_flwr.server_app:app\n"
                        f"Only SecAgg+ can be auto-detected due to its workflow requirements."
                    )

            except Exception as e:
                import traceback

                logger.error(f"❌ Auto-detection failed: {e}")
                logger.error(traceback.format_exc())
                raise

        BaseAppFactory._log_app_creation("Server", "auto_detecting", "with strategy detection")
        return auto_app

    @staticmethod
    def _create_server_fn():
        """Create the main server factory function.

        Returns:
            Server factory function
        """

        def server_fn(context: Context):
            """Server factory function that auto-detects strategy and uses appropriate execution mode.

            Args:
                context: Context containing server configuration

            Returns:
                Server components for the FL application (regular strategies only)

            Raises:
                ValueError: If SecAgg+ strategy is detected (requires workflow execution)
            """
            try:
                # Load config
                (config,) = ServerAppFactory._create_server_context_components(context)

                # Validate strategy and check for SecAgg+
                strategy_name = ServerAppFactory._validate_strategy_config(config)
                logger.info(f"🔧 Using FL strategy: {strategy_name}")

                # Check for SecAgg+ and provide helpful error
                if StrategyDetector.is_secagg_plus_strategy(strategy_name):
                    raise ValueError(
                        "🔒 SecAgg+ strategy detected!\n"
                        "SecAgg+ requires workflow-based execution using @app.main() pattern.\n"
                        "Your config specifies SecAgg+ but you're using the regular ServerApp pattern.\n"
                        "\n"
                        "Solutions:\n"
                        "1. Use dedicated SecAgg+ app: flower-server-app adni_flwr.server_app:secagg_plus_app\n"
                        "2. Or use auto-detecting app: flower-server-app adni_flwr.server_app:auto_app\n"
                        "\n"
                        "The regular 'app' cannot handle SecAgg+ due to Flower framework limitations.\n"
                        "SecAgg+ requires Grid parameter that's only available in @app.main() context."
                    )

                # Create server components
                logger.info(f"📊 Using regular strategy execution for: {strategy_name}")
                return ServerAppFactory._create_server_components(config, strategy_name)

            except Exception as e:
                import traceback

                logger.error(f"❌ Error in server_fn: {e}")
                logger.error(traceback.format_exc())
                raise

        return server_fn

    @staticmethod
    def _create_server_components(config: Config, strategy_name: str) -> ServerAppComponents:
        """Create server components (strategy, config, etc.).

        Args:
            config: Configuration object
            strategy_name: Name of the FL strategy

        Returns:
            ServerAppComponents with strategy and config

        Raises:
            Exception: If component creation fails
        """
        # Create WandB logger
        wandb_logger = FLServerWandbLogger(config)
        enable_shared_mode = getattr(config.wandb, "enable_shared_mode", True)
        run_id = wandb_logger.init_wandb(enable_shared_mode=enable_shared_mode)

        # Log WandB status
        if run_id:
            logger.info(f"✅ WandB run ID {run_id} will be shared with clients through FL communication")
        else:
            logger.warning("⚠️ Failed to get WandB run ID for sharing with clients")

        # Initialize and setup model
        model = load_model(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Debug server model
        debug_model_architecture(model, "Server Model (after initialization)")

        # Convert initial model parameters
        ndarrays = get_params(model)
        ndarrays_to_parameters(ndarrays)

        # Get FL parameters
        num_rounds = config.fl.num_rounds

        # Validate strategy configuration
        StrategyFactory.validate_strategy_config(strategy_name, config)

        # Create strategy with fallback to safe weighted average
        try:
            strategy = StrategyFactory.create_server_strategy(
                strategy_name=strategy_name,
                config=config,
                model=model,
                wandb_logger=wandb_logger,
                evaluate_metrics_aggregation_fn=ServerAppFactory._weighted_average,
                fit_metrics_aggregation_fn=ServerAppFactory._weighted_average,
            )
        except Exception as e:
            logger.error(f"❌ Error creating strategy with weighted_average: {e}")
            logger.info("🔄 Falling back to safe_weighted_average...")

            strategy = StrategyFactory.create_server_strategy(
                strategy_name=strategy_name,
                config=config,
                model=model,
                wandb_logger=wandb_logger,
                evaluate_metrics_aggregation_fn=safe_weighted_average,
                fit_metrics_aggregation_fn=safe_weighted_average,
            )

        # Create server configuration
        server_config = ServerConfig(num_rounds=num_rounds)

        logger.info(f"✅ Server components created for {strategy_name} strategy")
        return ServerAppComponents(strategy=strategy, config=server_config)

    @staticmethod
    def _execute_secagg_plus_workflow(grid: Grid, context: Context, config: Config = None):
        """Execute SecAgg+ workflow.

        Args:
            grid: Flower Grid for workflow execution
            context: Context containing server configuration
            config: Optional pre-loaded config
        """
        logger.info("🔒 SecAgg+ Server - Starting workflow-based execution")

        try:
            # Load config if not provided
            if config is None:
                (config,) = ServerAppFactory._create_server_context_components(context)

            # Validate it's actually SecAgg+
            strategy_name = ServerAppFactory._validate_strategy_config(config, "secagg+")

            # Create WandB logger
            wandb_logger = FLServerWandbLogger(config)
            enable_shared_mode = getattr(config.wandb, "enable_shared_mode", True)
            wandb_logger.init_wandb(enable_shared_mode=enable_shared_mode)

            # Initialize model
            model = load_model(config)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Debug server model
            debug_model_architecture(model, "SecAgg+ Server Model (after initialization)")

            # Get FL parameters
            num_rounds = config.fl.num_rounds
            logger.info(f"🔒 SecAgg+ Server - Using strategy: {strategy_name}")

            # Validate strategy configuration
            StrategyFactory.validate_strategy_config(strategy_name, config)

            # Create SecAgg+ strategy
            strategy = StrategyFactory.create_server_strategy(
                strategy_name=strategy_name,
                config=config,
                model=model,
                wandb_logger=wandb_logger,
                evaluate_metrics_aggregation_fn=ServerAppFactory._weighted_average,
                fit_metrics_aggregation_fn=ServerAppFactory._weighted_average,
            )

            # Get the SecAgg+ workflow
            if not hasattr(strategy, "get_secagg_workflow"):
                raise ValueError("SecAgg+ strategy does not have get_secagg_workflow method")

            secagg_workflow = strategy.get_secagg_workflow()
            logger.success(f"✅ Retrieved SecAgg+ workflow: {type(secagg_workflow).__name__}")

            # Create the main workflow with SecAgg+ fit workflow
            workflow = DefaultWorkflow(fit_workflow=secagg_workflow)

            # Create legacy context for workflow execution
            legacy_context = LegacyContext(
                context=context,
                config=ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )

            # Execute the SecAgg+ workflow
            logger.info("🚀 Executing SecAgg+ workflow...")
            workflow(grid, legacy_context)
            logger.success("✅ SecAgg+ workflow completed successfully!")

        except Exception as e:
            logger.error(f"❌ SecAgg+ server execution failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @staticmethod
    def _weighted_average(metrics):
        """Compute weighted average of metrics with error handling.

        Args:
            metrics: List of (num_examples, metrics) tuples

        Returns:
            Aggregated metrics dictionary
        """
        from collections.abc import Mapping

        try:
            if not metrics:
                logger.warning("weighted_average received empty metrics list")
                return {}

            logger.info(f"📊 Weighted average received {len(metrics)} metrics")

            filtered_metrics = [
                (num_examples, dict(m)) for num_examples, m in metrics if isinstance(m, (dict, Mapping)) and m
            ]
            if not filtered_metrics:
                logger.warning("📊 Weighted average filtered metrics list is empty after filtering")
                return {}

            # Get all metric names that are present in at least one client
            all_metric_names = set()
            for _, m in filtered_metrics:
                all_metric_names.update(name for name in m.keys() if isinstance(name, str))

            logger.info(f"📊 Metric names present: {all_metric_names}")

            acc_metrics = {}
            for name in all_metric_names:
                # Get all clients that reported this metric
                client_data = [(num_examples, m.get(name)) for num_examples, m in filtered_metrics if name in m]

                # Skip empty data
                if not client_data:
                    continue

                # Sample value to determine type
                sample_value = client_data[0][1]

                # Handle scalar metrics (compute weighted average)
                if all(isinstance(value, (int, float)) for _, value in client_data):
                    try:
                        # For training_time, use simple average instead of weighted average
                        if name == "training_time":
                            values = [float(value) for _, value in client_data]
                            acc_metrics[name] = sum(values) / len(values) if values else 0.0
                        else:
                            weighted_values = [float(value) * num_examples for num_examples, value in client_data]
                            total_examples = sum(num_examples for num_examples, _ in client_data)
                            if total_examples > 0:
                                acc_metrics[name] = sum(weighted_values) / total_examples
                    except Exception as e:
                        logger.error(f"❌ Error processing scalar metric '{name}': {e}")

                # Pass through string metrics (like JSON-encoded lists)
                elif name in ["predictions_json", "labels_json", "sample_info", "client_id"] and isinstance(
                    sample_value, str
                ):
                    # Use the first client's value (arbitrary choice)
                    acc_metrics[name] = sample_value
                    logger.info(f"📊 Passing through string metric '{name}' with length {len(sample_value)}")

                # Other scalar values (like num_classes)
                elif name == "num_classes" and isinstance(sample_value, (int, float)):
                    # Use the first client's value
                    acc_metrics[name] = sample_value

                # Error for other types that shouldn't be here
                else:
                    logger.warning(
                        f"⚠️ Skipping metric '{name}' with type {type(sample_value).__name__} "
                        f"- not supported for aggregation"
                    )

            logger.info(f"📊 Aggregated metrics keys: {list(acc_metrics.keys())}")
            return acc_metrics

        except Exception as e:
            import traceback

            logger.error(f"❌ Error in weighted_average: {e}")
            logger.error(traceback.format_exc())
            return {}


# =============================================================================
# SERVER APP INSTANCES WITH ERROR HANDLING
# =============================================================================


def _create_server_app_with_error_handling(creation_func, app_name: str, *args, **kwargs):
    """Create server app with consistent error handling."""
    try:
        app = creation_func(*args, **kwargs)
        logger.success(f"✅ {app_name} created successfully")
        return app
    except Exception as e:
        logger.error(f"❌ Failed to create {app_name}: {e}")
        # Re-raise to make the error explicit - don't hide import failures
        raise RuntimeError(f"Failed to initialize {app_name}: {e}") from e


# Create server app instances
regular_server_app = _create_server_app_with_error_handling(
    ServerAppFactory.create_regular_server_app, "regular_server_app"
)

secagg_plus_server_app = _create_server_app_with_error_handling(
    ServerAppFactory.create_secagg_plus_server_app, "secagg_plus_server_app"
)

auto_detecting_server_app = _create_server_app_with_error_handling(
    ServerAppFactory.create_auto_detecting_server_app, "auto_detecting_server_app"
)
