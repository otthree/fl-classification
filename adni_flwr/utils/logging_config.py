"""Logging configuration for ADNI Federated Learning.

This module provides centralized logging configuration using loguru for production-level logging
across the entire adni_flwr package.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from loguru._logger import Logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    serialize: bool = False,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "gz",
) -> None:
    """Configure loguru logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging is used.
                  For FL applications, this should be None to avoid Ray serialization issues.
        serialize: Whether to serialize logs as JSON (only applies to file logging)
        rotation: When to rotate log files (only applies to file logging)
        retention: How long to keep old log files (only applies to file logging)
        compression: Compression format for rotated logs (only applies to file logging)
    """
    # Check if we're already configured to avoid reconfiguration in Ray workers
    if hasattr(logger, "_ray_configured"):
        return
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Add console handler with colored output (always present)
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if log_file is specified (not recommended for FL/Ray applications)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=serialize,
            backtrace=True,
            diagnose=True,
        )

    # Mark as configured to prevent reconfiguration
    logger._ray_configured = True


def get_logger(name: str) -> Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


def setup_fl_logging(
    client_id: Optional[int] = None,
    log_dir: Optional[Union[str, Path]] = None,
    level: Optional[str] = None,
) -> None:
    """Setup console-only logging for federated learning components.

    This function is Ray-compatible as it only uses console logging, avoiding
    file handler serialization issues. File-based logging is handled by WandB.

    Args:
        client_id: Client ID for client-specific logging (used for log formatting)
        log_dir: Directory for log files (ignored - kept for API compatibility)
        level: Logging level. If None, will use LOG_LEVEL environment variable or default to "INFO"
    """
    # Skip if already configured for this process/worker
    config_key = f"_fl_logging_configured_{client_id}"
    if hasattr(logger, config_key):
        return

    # Get log level from environment variable if not provided
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        logger.warning(f"Invalid log level '{level}', defaulting to 'INFO'. Valid levels: {valid_levels}")
        level = "INFO"

    # Console-only logging setup - Ray-compatible, no file handlers
    try:
        setup_logging(level=level, log_file=None)  # Console only
        # Mark this specific configuration as complete
        setattr(logger, config_key, True)

        # Log the setup completion with client context
        if client_id is not None:
            logger.info(f"Console logging configured for FL Client {client_id} (level: {level})")
        else:
            logger.info(f"Console logging configured for FL Server (level: {level})")

    except Exception as e:
        print(f"ERROR: Console logging setup failed: {e}")
        # Continue without custom logging setup
