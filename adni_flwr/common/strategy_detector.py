"""Strategy detection and validation for federated learning."""

from typing import Any, List, Optional

from loguru import logger

from adni_classification.config.config import Config


class StrategyDetector:
    """Handles strategy detection, validation, and module availability checks."""

    # Module availability checks
    _SECAGGPLUS_MOD_AVAILABLE = None
    _LOCALDPMOD_AVAILABLE = None
    _secaggplus_mod = None
    _LocalDpMod = None

    @classmethod
    def _check_secaggplus_availability(cls) -> bool:
        """Check if SecAgg+ mod is available."""
        if cls._SECAGGPLUS_MOD_AVAILABLE is None:
            try:
                from flwr.client.mod import secaggplus_mod

                cls._SECAGGPLUS_MOD_AVAILABLE = True
                cls._secaggplus_mod = secaggplus_mod
            except ImportError:
                cls._SECAGGPLUS_MOD_AVAILABLE = False
                cls._secaggplus_mod = None
        return cls._SECAGGPLUS_MOD_AVAILABLE

    @classmethod
    def _check_localdpmod_availability(cls) -> bool:
        """Check if LocalDpMod is available."""
        if cls._LOCALDPMOD_AVAILABLE is None:
            try:
                from flwr.client.mod import LocalDpMod

                cls._LOCALDPMOD_AVAILABLE = True
                cls._LocalDpMod = LocalDpMod
            except ImportError:
                cls._LOCALDPMOD_AVAILABLE = False
                cls._LocalDpMod = None
        return cls._LOCALDPMOD_AVAILABLE

    @staticmethod
    def validate_strategy_requirements(strategy_name: str) -> None:
        """Validate that required dependencies are available for the strategy.

        Args:
            strategy_name: Name of the strategy to validate

        Raises:
            ValueError: If required dependencies are not available
        """
        strategy_lower = strategy_name.lower()

        if strategy_lower in ["secagg+", "secaggplus"]:
            logger.info("🔒 SecAgg+ strategy detected")
            if not StrategyDetector._check_secaggplus_availability():
                raise ValueError(
                    "SecAgg+ strategy selected but secaggplus_mod is not available. "
                    "Please ensure you have the correct Flower version with SecAgg+ support."
                )
            logger.success("✅ SecAgg+ mod is available")

        elif strategy_lower == "differential_privacy":
            logger.info("🔒 Differential Privacy strategy detected")
            if not StrategyDetector._check_localdpmod_availability():
                raise ValueError(
                    "Differential Privacy strategy selected but LocalDpMod is not available. "
                    "Please ensure you have the correct Flower version with LocalDpMod support."
                )
            logger.success("✅ LocalDpMod is available")

        elif strategy_lower in ["fedavg", "fedprox", "secagg"]:
            logger.info(f"📊 Regular strategy detected: {strategy_name}")
            # No special requirements for regular strategies

        else:
            logger.warning(f"⚠️ Unknown strategy: {strategy_name}")

    @staticmethod
    def get_strategy_mods(strategy_name: str, config: Optional[Config] = None) -> List[Any]:
        """Get the appropriate mods list based on strategy type.

        Args:
            strategy_name: Name of the strategy
            config: Configuration object (required for differential_privacy)

        Returns:
            List of mods for the ClientApp

        Raises:
            ValueError: If required dependencies are not available or config is missing
        """
        strategy_lower = strategy_name.lower()

        if strategy_lower in ["secagg+", "secaggplus"]:
            if not StrategyDetector._check_secaggplus_availability():
                raise ValueError("SecAgg+ strategy requires secaggplus_mod which is not available")
            return [StrategyDetector._secaggplus_mod]

        elif strategy_lower == "differential_privacy":
            if not StrategyDetector._check_localdpmod_availability():
                raise ValueError("Differential Privacy strategy requires LocalDpMod which is not available")

            if config is None:
                raise ValueError(
                    "CRITICAL ERROR: Differential Privacy strategy requires a config to create LocalDpMod. "
                    "Use a DP-specific app creation function with a config parameter."
                )

            # Create LocalDpMod with config parameters
            return [StrategyDetector.create_local_dp_mod(config)]

        else:
            # Regular strategy - no mods
            return []

    @staticmethod
    def create_local_dp_mod(config: Config):
        """Create LocalDpMod instance with parameters from config.

        Args:
            config: Configuration object containing DP parameters

        Returns:
            LocalDpMod instance configured for differential privacy

        Raises:
            ValueError: If LocalDpMod is not available or config parameters are invalid
        """
        if not StrategyDetector._check_localdpmod_availability():
            raise ValueError("LocalDpMod is not available")

        # Get DP parameters from config or use defaults
        clipping_norm = getattr(config.fl, "dp_clipping_norm", 1.0)
        sensitivity = getattr(config.fl, "dp_sensitivity", clipping_norm)
        epsilon = getattr(config.fl, "dp_epsilon", 200.0)
        delta = getattr(config.fl, "dp_delta", 1e-5)

        # Convert values to float to ensure proper type
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

        logger.info("🔒 Creating LocalDpMod with parameters:")
        logger.info(f"   clipping_norm: {clipping_norm}")
        logger.info(f"   sensitivity: {sensitivity}")
        logger.info(f"   epsilon: {epsilon}")
        logger.info(f"   delta: {delta}")

        return StrategyDetector._LocalDpMod(clipping_norm, sensitivity, epsilon, delta)

    @staticmethod
    def is_secagg_plus_strategy(strategy_name: str) -> bool:
        """Check if strategy is SecAgg+.

        Args:
            strategy_name: Strategy name to check

        Returns:
            True if strategy is SecAgg+
        """
        return strategy_name.lower() in ["secagg+", "secaggplus"]

    @staticmethod
    def is_differential_privacy_strategy(strategy_name: str) -> bool:
        """Check if strategy is differential privacy.

        Args:
            strategy_name: Strategy name to check

        Returns:
            True if strategy is differential privacy
        """
        return strategy_name.lower() == "differential_privacy"

    @staticmethod
    def requires_workflow_execution(strategy_name: str) -> bool:
        """Check if strategy requires workflow-based execution.

        Args:
            strategy_name: Strategy name to check

        Returns:
            True if strategy requires workflow execution (like SecAgg+)
        """
        return StrategyDetector.is_secagg_plus_strategy(strategy_name)
