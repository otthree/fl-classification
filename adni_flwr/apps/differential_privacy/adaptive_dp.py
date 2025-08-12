"""Adaptive differential privacy implementation with noise scaling."""

import io
import os
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

        logger.info(
            f"AdaptiveLocalDpMod initialized | initial_epsilon={initial_epsilon} "
            f"decay_factor={decay_factor} min_epsilon={self.min_epsilon} "
            f"max_epsilon={self.max_epsilon} base_sensitivity={self.base_sensitivity}"
        )

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

        # Check if this is a FitRes with parameters
        if hasattr(response, "parameters") and response.parameters is not None:
            parameters = response.parameters
        # Check if this is a message with fit_res containing parameters
        elif (
            hasattr(response, "fit_res")
            and hasattr(response.fit_res, "parameters")
            and response.fit_res.parameters is not None
        ):
            parameters = response.fit_res.parameters
        # Check if this is a tuple return from NumPyClient (parameters, num_examples, metrics)
        elif isinstance(response, tuple) and len(response) >= 1:
            # First element should be parameters
            potential_params = response[0]
            if isinstance(potential_params, (list, tuple)) and len(potential_params) > 0:
                parameters = potential_params
                # Check if response has a content attribute with parameters
        elif hasattr(response, "content") and response.content is not None:
            content = response.content

            # Check if content is a RecordDict with parameters_records
            if hasattr(content, "parameters_records") and content.parameters_records is not None:
                # Extract parameters from RecordDict
                param_records = content.parameters_records

                # Check if it's a dict-like structure with fitres.parameters
                if hasattr(param_records, "get"):
                    fitres_params = param_records.get("fitres.parameters")
                    if fitres_params is not None:
                        # Extract the parameter arrays from the fitres.parameters dict
                        param_list = []
                        param_keys = sorted([k for k in fitres_params.keys() if k.isdigit()], key=int)

                        for key in param_keys:
                            array_obj = fitres_params[key]
                            if hasattr(array_obj, "data"):
                                # Convert the Array object to numpy array
                                try:
                                    # First try to load as numpy's native format (from np.save)
                                    if array_obj.data.startswith(b"\x93NUMPY"):
                                        # This is numpy's native serialization format
                                        buffer = io.BytesIO(array_obj.data)
                                        param_array = np.load(buffer, allow_pickle=False)

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

                                        except Exception as e2:
                                            logger.error(f"Failed frombuffer approach: {e2}")
                                            raise

                                    param_list.append(param_array)
                                except Exception as e:
                                    logger.error(f"Failed to extract parameter {key}: {e}")
                                    # Try alternative approach
                                    try:
                                        param_array = np.frombuffer(array_obj.data, dtype=np.uint8)
                                    except Exception as e2:
                                        logger.error(f"Could not read raw data: {e2}")

                        if param_list:
                            parameters = param_list

            # Check if content is a FitRes with parameters attribute
            elif hasattr(content, "parameters") and content.parameters is not None:
                parameters = content.parameters
            # Check if content has fit_res attribute containing parameters
            elif hasattr(content, "fit_res") and hasattr(content.fit_res, "parameters"):
                parameters = content.fit_res.parameters
            # Check if content is a tuple (parameters, num_examples, metrics)
            elif isinstance(content, tuple) and len(content) >= 1:
                potential_params = content[0]
                if isinstance(potential_params, (list, tuple)) and len(potential_params) > 0:
                    parameters = potential_params
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
                                logger.debug(f"Found parameters in response.content[{key}].parameters")
                                break
                            elif isinstance(param_candidate, (list, tuple)) and len(param_candidate) > 0:
                                parameters = param_candidate
                                logger.debug(f"Found parameters in response.content[{key}]")
                                break
                except Exception:
                    logger.exception("Failed to access parameters via dict-like content structure")
            else:
                pass

        if parameters is None:
            # If no parameters found, likely not a FitRes (e.g., EvaluateRes). Log benign debug and return.
            message_hint = "unknown"
            try:
                if hasattr(response, "content") and response.content is not None:
                    content = response.content
                    metric_records = getattr(content, "metric_records", None)
                    config_records = getattr(content, "config_records", None)

                    def _has_eval_keys(rec: Any) -> bool:
                        try:
                            return bool(rec) and any("evaluateres" in str(k) for k in rec.keys())
                        except Exception:
                            return False

                    if _has_eval_keys(metric_records) or _has_eval_keys(config_records):
                        message_hint = "evaluate"
                    else:
                        message_hint = "non-fit"
            except Exception:
                logger.error(f"Failed to extract parameters from response: {response}")
                message_hint = "unknown"

            logger.warning(
                "AdaptiveLocalDpMod: no parameters present (hint=%s); skipping adaptive DP noise.",
                message_hint,
            )
            return response

        # Determine FL round index in a stateless manner if possible
        def _extract_round_index(resp: Any, ctx: Any) -> Optional[int]:
            try:
                content = getattr(resp, "content", None)
                # Prefer metric records, e.g., {'fitres.metrics': {'round': N}}
                metric_records = getattr(content, "metric_records", None)
                if isinstance(metric_records, dict):
                    # Direct key
                    fit_metrics = metric_records.get("fitres.metrics")
                    if isinstance(fit_metrics, dict):
                        for k in ("round", "current_round", "server_round"):
                            if k in fit_metrics and isinstance(fit_metrics[k], (int, float)):
                                return int(fit_metrics[k])
                    # Search any metric record
                    for _k, rec in metric_records.items():
                        if isinstance(rec, dict):
                            for k in ("round", "current_round", "server_round"):
                                if k in rec and isinstance(rec[k], (int, float)):
                                    return int(rec[k])

                # Fallback: try config records, e.g., FitIns config
                config_records = getattr(content, "config_records", None)
                if isinstance(config_records, dict):
                    # Some stacks include FitIns/EvaluateIns config in records
                    for _k, rec in config_records.items():
                        if isinstance(rec, dict):
                            for k in ("round", "current_round", "server_round"):
                                if k in rec and isinstance(rec[k], (int, float)):
                                    return int(rec[k])

                # As a last resort, some contexts provide round in ctx
                # Try common attributes conservatively
                for attr_name in ("round", "server_round", "current_round"):
                    try:
                        val = getattr(ctx, attr_name)
                        if isinstance(val, (int, float)):
                            return int(val)
                    except Exception:
                        logger.exception("Failed while inspecting content for parameter hint keys")
            except Exception:
                logger.exception("Unexpected error extracting round index from response/context")
            return None

        def _extract_client_id(resp: Any, ctx: Any) -> Optional[str]:
            try:
                content = getattr(resp, "content", None)
                # Look into metric records
                metric_records = getattr(content, "metric_records", None)
                if isinstance(metric_records, dict):
                    fit_metrics = metric_records.get("fitres.metrics")
                    if isinstance(fit_metrics, dict) and "client_id" in fit_metrics:
                        return str(fit_metrics["client_id"]).strip()
                    for _k, rec in metric_records.items():
                        if isinstance(rec, dict) and "client_id" in rec:
                            return str(rec["client_id"]).strip()
                # Look into config records
                config_records = getattr(content, "config_records", None)
                if isinstance(config_records, dict):
                    for _k, rec in config_records.items():
                        if isinstance(rec, dict) and "client_id" in rec:
                            return str(rec["client_id"]).strip()
            except Exception:
                logger.exception("Unexpected error extracting client id from response/context")
            # Try context attributes
            for attr_name in ("client_id", "partition_id"):
                try:
                    val = getattr(ctx, attr_name)
                    if val is not None:
                        return str(val).strip()
                except Exception:
                    continue
            return None

        def _env_round_key(resp: Any, ctx: Any) -> str:
            client_id = _extract_client_id(resp, ctx) or "UNKNOWN"
            return f"ADAPTIVE_DP_ROUND_{client_id}"

        inferred_round = _extract_round_index(response, context)
        env_key = _env_round_key(response, context)
        # Read existing env-tracked round (process-local persistence across re-inits)
        try:
            env_round_val = int(os.environ.get(env_key, "0"))
        except Exception:
            env_round_val = 0

        if inferred_round is not None and inferred_round >= 1:
            self.round_count = inferred_round
        elif env_round_val >= 1:
            self.round_count = env_round_val + 1
        else:
            # Fall back to internal counter if nothing else is available
            self.round_count += 1

        # Persist the chosen round into env for next instantiation within same process
        try:
            os.environ[env_key] = str(self.round_count)
        except Exception:
            logger.exception("Failed to persist adaptive DP round into environment")

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

        logger.info(
            "AdaptiveLocalDpMod round=%d current_epsilon=%.4f noise_scale=%.6f param_norm=%.6f noise_norm=%.6f "
            "noise_param_ratio=%.4f",
            self.round_count,
            self.current_epsilon,
            noise_scale,
            total_param_norm,
            total_noise_norm,
            total_noise_norm / max(total_param_norm, 1e-6),
        )

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

                        updated = True
            elif hasattr(content, "parameters"):
                content.parameters = noisy_parameters
                logger.debug("Updated parameters in response.content.parameters")
                updated = True
            elif hasattr(content, "fit_res") and hasattr(content.fit_res, "parameters"):
                content.fit_res.parameters = noisy_parameters
                logger.debug("Updated parameters in response.content.fit_res.parameters")
                updated = True
            elif isinstance(content, tuple) and len(content) >= 1:
                # Replace the first element (parameters) in the tuple
                response.content = (noisy_parameters, *content[1:])
                logger.debug("Updated parameters in response.content tuple")
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
                                logger.debug(f"Updated parameters in response.content[{key}].parameters")
                                updated = True
                                break
                            elif isinstance(param_candidate, (list, tuple)):
                                if hasattr(content, "__setitem__"):
                                    content[key] = noisy_parameters
                                    logger.debug(f"Updated parameters in response.content[{key}]")
                                    updated = True
                                    break
                except Exception:
                    logger.exception("Failed to update content as dict/record with noisy parameters")

            if not updated:
                logger.warning(f"Could not update content parameters - content type: {type(content)}")
        else:
            logger.warning("Could not update response with noisy parameters - unknown response format")

        return response
