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
        serialize: Whether to serialize logs as JSON
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep old log files
        compression: Compression format for rotated logs
    """
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Add console handler with colored output
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

    # Add file handler if log_file is specified
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
    """Setup logging specifically for federated learning components.

    Args:
        client_id: Client ID for client-specific logging
        log_dir: Directory for log files
        level: Logging level. If None, will use LOG_LEVEL environment variable or default to "INFO"
    """
    # Get log level from environment variable if not provided
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        logger.warning(f"Invalid log level '{level}', defaulting to 'INFO'. Valid levels: {valid_levels}")
        level = "INFO"
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)

    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup different log files for server and clients
    if client_id is not None:
        log_file = log_dir / f"client_{client_id}.log"
    else:
        log_file = log_dir / "server.log"

    setup_logging(
        level=level,
        log_file=log_file,
        serialize=False,
        rotation="50 MB",
        retention="14 days",
    )
