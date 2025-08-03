"""Adaptive differential privacy implementation with noise scaling."""

import torch
from loguru import logger


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
        """Initialize adaptive differential privacy mod.

        Args:
            clipping_norm: Gradient clipping norm
            sensitivity: Sensitivity parameter for DP
            initial_epsilon: Initial epsilon value (privacy budget)
            delta: Delta parameter for DP
            decay_factor: Exponential decay factor for epsilon
            min_epsilon: Minimum epsilon value (defaults to 10% of initial)
        """
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
        if hasattr(response, "parameters") and response.parameters is not None:
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
