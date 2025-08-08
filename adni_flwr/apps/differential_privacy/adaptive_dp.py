"""Adaptive differential privacy implementation with noise scaling."""

import io
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger


class AdaptiveLocalDpMod:
    """Custom LocalDpMod with adaptive noise scaling.

    This addresses the core issue by:
    1. Reducing noise magnitude as training progresses (epsilon grows over rounds)
    2. Scaling noise based on per-tensor statistics (std-based, capped)
    3. Using an exponential growth schedule for epsilon (inverse of ``decay_factor``)
    """

    def __init__(
        self,
        clipping_norm: float,
        sensitivity: float,
        initial_epsilon: float,
        delta: float,
        decay_factor: float = 0.95,
        min_epsilon: Optional[float] = None,
        max_epsilon: Optional[float] = None,
    ):
        """Initialize adaptive differential privacy mod.

        Args:
            clipping_norm: Gradient clipping norm
            sensitivity: Sensitivity parameter for DP (overridden by ``clipping_norm`` if > 0)
            initial_epsilon: Initial epsilon value (privacy budget)
            delta: Delta parameter for DP
            decay_factor: Exponential decay factor used inversely to grow epsilon per round
            min_epsilon: Minimum epsilon value (defaults to 10% of initial)
            max_epsilon: Optional cap on epsilon growth (no cap if None)
        """
        self.clipping_norm = float(clipping_norm)
        # Derive sensitivity from clipping norm if available, aligning with DP-SGD
        if self.clipping_norm and self.clipping_norm > 0.0:
            self.base_sensitivity = self.clipping_norm
            if sensitivity != self.clipping_norm:
                logger.info(
                    "Using clipping_norm as sensitivity (overriding provided sensitivity): "
                    f"clipping_norm={self.clipping_norm}, provided_sensitivity={sensitivity}"
                )
        else:
            self.base_sensitivity = float(sensitivity)
        self.initial_epsilon = initial_epsilon
        self.current_epsilon = initial_epsilon
        self.delta = delta
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon or initial_epsilon * 0.1  # Minimum 10% of initial
        self.max_epsilon = max_epsilon
        self.round_count = 0

        logger.info("🔧 AdaptiveLocalDpMod initialized:")
        logger.info(f"   initial_epsilon: {initial_epsilon}")
        logger.info(f"   decay_factor: {decay_factor}")
        logger.info(f"   min_epsilon: {self.min_epsilon}")
        if self.max_epsilon is not None:
            logger.info(f"   max_epsilon: {self.max_epsilon}")
        logger.info(f"   base_sensitivity (from clipping_norm if >0): {self.base_sensitivity}")

    @property
    def sensitivity(self) -> float:
        """Backward-compatible accessor.

        Returns the effective sensitivity used for noise scaling.
        """
        return self.base_sensitivity

    def __call__(self, message: Any, context: Any, ffn: Any) -> Any:
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

        # Extract parameters from the response - handle different message types
        parameters = None

        logger.debug(f"🔍 AdaptiveLocalDpMod: Response type: {type(response)}")
        logger.debug(f"🔍 AdaptiveLocalDpMod: Response attributes: {dir(response)}")

        # Check if this is a FitRes with parameters
        if hasattr(response, "parameters") and response.parameters is not None:
            parameters = response.parameters
            logger.debug("✅ Found parameters in response.parameters")
        # Check if this is a message with fit_res containing parameters
        elif (
            hasattr(response, "fit_res")
            and hasattr(response.fit_res, "parameters")
            and response.fit_res.parameters is not None
        ):
            parameters = response.fit_res.parameters
            logger.debug("✅ Found parameters in response.fit_res.parameters")
        # Check if this is a tuple return from NumPyClient (parameters, num_examples, metrics)
        elif isinstance(response, tuple) and len(response) >= 1:
            # First element should be parameters
            potential_params = response[0]
            if isinstance(potential_params, (list, tuple)) and len(potential_params) > 0:
                parameters = potential_params
                logger.debug("✅ Found parameters in tuple response[0]")
                # Check if response has a content attribute with parameters
        elif hasattr(response, "content") and response.content is not None:
            content = response.content
            logger.debug(f"🔍 Content type: {type(content)}")
            logger.debug(f"🔍 Content attributes: {dir(content)}")

            # Check if content is a RecordDict with parameters_records
            if hasattr(content, "parameters_records") and content.parameters_records is not None:
                # Extract parameters from RecordDict
                param_records = content.parameters_records
                logger.debug(f"🔍 Found parameters_records: {param_records}")

                # Check if it's a dict-like structure with fitres.parameters
                if hasattr(param_records, "get"):
                    fitres_params = param_records.get("fitres.parameters")
                    if fitres_params is not None:
                        # Extract the parameter arrays from the fitres.parameters dict
                        param_list = []
                        param_keys = sorted([k for k in fitres_params.keys() if k.isdigit()], key=int)
                        logger.debug(f"🔍 Parameter keys: {param_keys}")

                        for key in param_keys:
                            array_obj = fitres_params[key]
                            logger.debug(f"🔍 Processing parameter {key}: {type(array_obj)}")
                            logger.debug(
                                f"🔍 Array attributes: {[attr for attr in dir(array_obj) if not attr.startswith('_')]}"
                            )
                            if hasattr(array_obj, "__dict__"):
                                logger.debug(f"🔍 Array dict: {array_obj.__dict__}")
                            if hasattr(array_obj, "data"):
                                # Convert the Array object to numpy array
                                try:
                                    # First try to load as numpy's native format (from np.save)
                                    if array_obj.data.startswith(b"\x93NUMPY"):
                                        # This is numpy's native serialization format
                                        buffer = io.BytesIO(array_obj.data)
                                        param_array = np.load(buffer, allow_pickle=False)
                                        logger.debug(
                                            f"✅ Loaded parameter {key} using numpy format, shape={param_array.shape}"
                                        )
                                    else:
                                        # Fallback: try raw bytes with explicit dtype and shape
                                        try:
                                            param_array = np.frombuffer(array_obj.data, dtype=np.float32)
                                            # Reshape according to the Array's shape if available
                                            if hasattr(array_obj, "shape") and array_obj.shape:
                                                # Parse shape from string if it's a string representation
                                                if isinstance(array_obj.shape, str):
                                                    # Extract shape from string like "(8, 1, 3, 3, 3)"
                                                    import ast

                                                    try:
                                                        shape = ast.literal_eval(array_obj.shape)
                                                        param_array = param_array.reshape(shape)
                                                    except Exception:
                                                        logger.warning(f"Could not parse shape: {array_obj.shape}")
                                                else:
                                                    param_array = param_array.reshape(array_obj.shape)
                                            logger.debug(
                                                f"✅ Loaded parameter {key} using frombuffer, shape={param_array.shape}"
                                            )
                                        except Exception as e2:
                                            logger.error(f"❌ Failed frombuffer approach: {e2}")
                                            raise

                                    param_list.append(param_array)
                                except Exception as e:
                                    logger.error(f"❌ Failed to extract parameter {key}: {e}")
                                    # Try alternative approach
                                    try:
                                        param_array = np.frombuffer(array_obj.data, dtype=np.uint8)
                                        logger.debug(f"   Raw data length: {len(param_array)}")
                                    except Exception as e2:
                                        logger.error(f"❌ Could not read raw data: {e2}")

                        if param_list:
                            parameters = param_list
                            logger.debug(f"✅ Found {len(parameters)} parameters in RecordDict")

            # Check if content is a FitRes with parameters attribute
            elif hasattr(content, "parameters") and content.parameters is not None:
                parameters = content.parameters
                logger.debug("✅ Found parameters in response.content.parameters")
            # Check if content has fit_res attribute containing parameters
            elif hasattr(content, "fit_res") and hasattr(content.fit_res, "parameters"):
                parameters = content.fit_res.parameters
                logger.debug("✅ Found parameters in response.content.fit_res.parameters")
            # Check if content is a tuple (parameters, num_examples, metrics)
            elif isinstance(content, tuple) and len(content) >= 1:
                potential_params = content[0]
                if isinstance(potential_params, (list, tuple)) and len(potential_params) > 0:
                    parameters = potential_params
                    logger.debug("✅ Found parameters in response.content tuple")
            # Try to access content as a record/dict-like structure
            elif hasattr(content, "__getitem__") or hasattr(content, "get"):
                try:
                    # Try different possible keys for parameters
                    for key in ["parameters", "fit_res", "result"]:
                        if hasattr(content, "get"):
                            param_candidate = content.get(key)
                        elif hasattr(content, "__getitem__"):
                            try:
                                param_candidate = content[key]
                            except (KeyError, IndexError):
                                param_candidate = None
                        else:
                            param_candidate = None

                        if param_candidate is not None:
                            if hasattr(param_candidate, "parameters"):
                                parameters = param_candidate.parameters
                                logger.debug(f"✅ Found parameters in response.content[{key}].parameters")
                                break
                            elif isinstance(param_candidate, (list, tuple)) and len(param_candidate) > 0:
                                parameters = param_candidate
                                logger.debug(f"✅ Found parameters in response.content[{key}]")
                                break
                except Exception as e:
                    logger.debug(f"   Failed to access content as dict/record: {e}")
            else:
                logger.debug("   Content does not have recognizable parameter structure")

        if parameters is None:
            # If no parameters found, return as-is
            logger.info(f"❌ No parameters found in response of type {type(response)}, skipping adaptive DP noise")
            logger.debug(f"   Response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")

            # Additional debugging for Message objects
            if hasattr(response, "content") and response.content is not None:
                content = response.content
                logger.debug(f"   Content type: {type(content)}")
                logger.debug(f"   Content dir: {[attr for attr in dir(content) if not attr.startswith('_')]}")
                if hasattr(content, "__dict__"):
                    logger.debug(f"   Content dict: {content.__dict__}")

                # Try to inspect the content more deeply
                try:
                    logger.debug(f"   Content str representation: {str(content)}")
                    logger.debug(f"   Content repr: {repr(content)}")

                    # Check if it's a RecordSet or similar Flower structure
                    if hasattr(content, "records"):
                        logger.debug(f"   Content has records: {content.records}")
                    if hasattr(content, "parameters"):
                        logger.debug(f"   Content has parameters: {type(content.parameters)}")
                    if hasattr(content, "fit_res"):
                        logger.debug(f"   Content has fit_res: {type(content.fit_res)}")

                except Exception as e:
                    logger.debug(f"   Error inspecting content: {e}")

            return response

        self.round_count += 1

        # Update epsilon with exponential growth (less noise over time):
        # epsilon_t = initial_epsilon * (1/decay_factor)^(t-1)
        growth_multiplier = (1.0 / max(self.decay_factor, 1e-6)) ** (self.round_count - 1)
        grown_epsilon = self.initial_epsilon * growth_multiplier
        if self.max_epsilon is not None:
            grown_epsilon = min(grown_epsilon, self.max_epsilon)
        # Respect the minimum epsilon lower bound
        self.current_epsilon = max(grown_epsilon, self.min_epsilon)

        # Calculate noise scale
        noise_scale = self.base_sensitivity / self.current_epsilon

        # Two-pass adaptive scaling using per-tensor std, capped in [min_scale, 1.0]
        # This meaningfully reduces noise for low-variance tensors while avoiding amplification beyond base noise.
        param_tensors = [torch.tensor(p, dtype=torch.float32) for p in parameters]
        tensor_stds = [torch.std(t).item() for t in param_tensors]
        mean_std = max(float(np.mean(tensor_stds)) if tensor_stds else 0.0, 1e-12)

        min_scale = 0.1  # Do not reduce below 10% of base noise

        noisy_parameters = []
        total_param_norm = 0.0
        total_noise_norm = 0.0

        for param_tensor, t_std in zip(param_tensors, tensor_stds, strict=False):
            param_norm = torch.norm(param_tensor).item()
            total_param_norm += param_norm

            # Adaptive factor based on relative std of this tensor vs. mean std across tensors
            rel_std = t_std / mean_std
            scale_factor = float(np.clip(rel_std, min_scale, 1.0))
            adaptive_scale = noise_scale * scale_factor

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
        logger.info(f"   noise/param_ratio: {total_noise_norm / max(total_param_norm, 1e-6):.4f}")

        # Update the response with noisy parameters - handle different message types
        if hasattr(response, "parameters"):
            response.parameters = noisy_parameters
        elif hasattr(response, "fit_res") and hasattr(response.fit_res, "parameters"):
            response.fit_res.parameters = noisy_parameters
        elif isinstance(response, tuple) and len(response) >= 1:
            # Replace the first element (parameters) in the tuple
            response = (noisy_parameters, *response[1:])
        elif hasattr(response, "content") and response.content is not None:
            content = response.content
            updated = False

            # Try to update parameters in the same way we found them
            if hasattr(content, "parameters_records") and content.parameters_records is not None:
                # Update parameters in RecordDict
                param_records = content.parameters_records
                if hasattr(param_records, "get"):
                    fitres_params = param_records.get("fitres.parameters")
                    if fitres_params is not None:
                        # Update the parameter arrays in the fitres.parameters dict
                        param_keys = sorted([k for k in fitres_params.keys() if k.isdigit()], key=int)

                        for i, key in enumerate(param_keys):
                            if i < len(noisy_parameters):
                                array_obj = fitres_params[key]
                                if hasattr(array_obj, "data"):
                                    # Convert noisy parameter back to bytes using numpy's safe serialization
                                    noisy_param = noisy_parameters[i].astype(np.float32)

                                    # Use numpy's save method to create proper serialized data
                                    buffer = io.BytesIO()
                                    np.save(buffer, noisy_param, allow_pickle=False)
                                    array_obj.data = buffer.getvalue()

                                    logger.debug(f"✅ Updated parameter {key} in RecordDict using numpy serialization")
                        updated = True
                        logger.debug("✅ Updated all parameters in RecordDict")
            elif hasattr(content, "parameters"):
                content.parameters = noisy_parameters
                logger.debug("✅ Updated parameters in response.content.parameters")
                updated = True
            elif hasattr(content, "fit_res") and hasattr(content.fit_res, "parameters"):
                content.fit_res.parameters = noisy_parameters
                logger.debug("✅ Updated parameters in response.content.fit_res.parameters")
                updated = True
            elif isinstance(content, tuple) and len(content) >= 1:
                # Replace the first element (parameters) in the tuple
                response.content = (noisy_parameters, *content[1:])
                logger.debug("✅ Updated parameters in response.content tuple")
                updated = True
            # Try to update content as a record/dict-like structure
            elif hasattr(content, "__setitem__") or hasattr(content, "set"):
                try:
                    # Try different possible keys for parameters
                    for key in ["parameters", "fit_res", "result"]:
                        if hasattr(content, "get"):
                            param_candidate = content.get(key)
                        elif hasattr(content, "__getitem__"):
                            try:
                                param_candidate = content[key]
                            except (KeyError, IndexError):
                                param_candidate = None
                        else:
                            param_candidate = None

                        if param_candidate is not None:
                            if hasattr(param_candidate, "parameters"):
                                param_candidate.parameters = noisy_parameters
                                logger.debug(f"✅ Updated parameters in response.content[{key}].parameters")
                                updated = True
                                break
                            elif isinstance(param_candidate, (list, tuple)):
                                if hasattr(content, "__setitem__"):
                                    content[key] = noisy_parameters
                                    logger.debug(f"✅ Updated parameters in response.content[{key}]")
                                    updated = True
                                    break
                except Exception as e:
                    logger.debug(f"   Failed to update content as dict/record: {e}")

            if not updated:
                logger.warning(f"Could not update content parameters - content type: {type(content)}")
        else:
            logger.warning("Could not update response with noisy parameters - unknown response format")

        return response
