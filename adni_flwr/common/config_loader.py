"""Unified configuration loading for client and server applications."""

import os

from flwr.common import Context
from loguru import logger

from adni_classification.config.config import Config


class ConfigLoader:
    """Centralized configuration loading with consistent error handling."""

    @staticmethod
    def load_client_config(context: Context, partition_id: int) -> Config:
        """Load client configuration from context.

        Args:
            context: Flower context containing configuration
            partition_id: Client partition ID to determine which config file to use

        Returns:
            Loaded Config object

        Raises:
            ValueError: If config files are missing or partition_id is invalid
            FileNotFoundError: If config file doesn't exist
            Exception: If config file is invalid or can't be parsed
        """
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
        return ConfigLoader._load_config_from_path(config_path, f"client partition {partition_id}")

    @staticmethod
    def load_server_config(context: Context) -> Config:
        """Load server configuration from context.

        Args:
            context: Flower context containing configuration

        Returns:
            Loaded Config object

        Raises:
            ValueError: If server config file is missing
            FileNotFoundError: If config file doesn't exist
            Exception: If config file is invalid or can't be parsed
        """
        # Get server config file from app config
        server_config_file = context.run_config.get("server-config-file")
        if not server_config_file:
            raise ValueError("Server config file not specified in run config")

        return ConfigLoader._load_config_from_path(server_config_file, "server")

    @staticmethod
    def load_config_from_path(config_path: str, description: str = "config") -> Config:
        """Load configuration from a specific file path.

        Args:
            config_path: Path to configuration file
            description: Description for error messages

        Returns:
            Loaded Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            Exception: If config file is invalid or can't be parsed
        """
        return ConfigLoader._load_config_from_path(config_path, description)

    @staticmethod
    def _load_config_from_path(config_path: str, description: str) -> Config:
        """Internal method to load config with consistent error handling.

        Args:
            config_path: Path to configuration file
            description: Description for error messages

        Returns:
            Loaded Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            Exception: If config file is invalid or can't be parsed
        """
        # Validate file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{description.title()} config file not found: {config_path}")

        try:
            config = Config.from_yaml(config_path)
            logger.info(f"✅ Loaded {description} config from: {config_path}")
            return config
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{description.title()} config file not found: {config_path}. Error: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load {description} config from {config_path}. Error: {e}") from e

    @staticmethod
    def validate_fl_section(config: Config, config_description: str = "config") -> None:
        """Validate that config has required FL section.

        Args:
            config: Configuration object to validate
            config_description: Description for error messages

        Raises:
            ValueError: If FL section is missing or invalid
        """
        if not hasattr(config, "fl"):
            raise ValueError(f"{config_description.title()} missing 'fl' section")

        if not hasattr(config.fl, "strategy"):
            raise ValueError(f"{config_description.title()} missing 'fl.strategy' field")

        if not config.fl.strategy:
            raise ValueError(
                f"ERROR: 'strategy' not specified in {config_description}. "
                f"You must explicitly set 'strategy' in the FL config section. "
                f"Available strategies: fedavg, fedprox, secagg, secagg+, differential_privacy. "
                f"This prevents dangerous implicit defaults that could cause strategy mismatch "
                f"between clients and server."
            )

    @staticmethod
    def validate_dp_config(config: Config, config_description: str = "config") -> None:
        """Validate differential privacy configuration parameters.

        Args:
            config: Configuration object to validate
            config_description: Description for error messages

        Raises:
            ValueError: If DP configuration is invalid or missing required parameters
        """
        ConfigLoader.validate_fl_section(config, config_description)

        if config.fl.strategy != "differential_privacy":
            raise ValueError(
                f"{config_description.title()} strategy must be 'differential_privacy', " f"got: '{config.fl.strategy}'"
            )

        # Validate required DP parameters
        required_dp_params = ["dp_epsilon", "dp_sensitivity", "dp_clipping_norm", "dp_delta"]
        missing_params = [param for param in required_dp_params if not hasattr(config.fl, param)]

        if missing_params:
            raise ValueError(
                f"Missing required DP parameters in {config_description}: {missing_params}. "
                f"Please add these to your config file under the 'fl' section."
            )

        # Validate parameter types and ranges
        try:
            clipping_norm = float(config.fl.dp_clipping_norm)
            sensitivity = float(config.fl.dp_sensitivity)
            epsilon = float(config.fl.dp_epsilon)
            delta = float(config.fl.dp_delta)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"DP parameters in {config_description} must be numeric. "
                f"Raw values: clipping_norm={config.fl.dp_clipping_norm}, "
                f"sensitivity={config.fl.dp_sensitivity}, "
                f"epsilon={config.fl.dp_epsilon}, "
                f"delta={config.fl.dp_delta}. "
                f"Error: {e}"
            ) from e

        # Validate optional boolean parameter
        if hasattr(config.fl, "dp_use_gaussian_mechanism"):
            use_gaussian = config.fl.dp_use_gaussian_mechanism
            if not isinstance(use_gaussian, bool):
                raise ValueError(
                    f"dp_use_gaussian_mechanism in {config_description} must be boolean, "
                    f"got: {use_gaussian} (type: {type(use_gaussian)})"
                )

        # Validate parameter ranges
        if clipping_norm <= 0:
            raise ValueError(f"dp_clipping_norm must be positive, got: {clipping_norm}")
        if sensitivity <= 0:
            raise ValueError(f"dp_sensitivity must be positive, got: {sensitivity}")
        if epsilon <= 0:
            raise ValueError(f"dp_epsilon must be positive, got: {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"dp_delta must be between 0 and 1 (exclusive), got: {delta}")

        logger.info(f"✅ DP config validation passed for {config_description}")
