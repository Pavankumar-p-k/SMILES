"""Utility functions for logging and setup."""

import logging
import sys
from pathlib import Path
from typing import Any

from config import (
    RANDOM_STATE,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)


def setup_logging(name: str = __name__, log_file: bool = True) -> logging.Logger:
    """
    Configure logging for a module.

    Args:
        name: Logger name
        log_file: Also log to file

    Returns:
        Configured logger
    """
    log_level = getattr(logging, LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)

    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    if not logger_obj.handlers:
        logger_obj.addHandler(ch)

    # File handler
    if log_file:
        log_path = PROJECT_ROOT / "admet.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if not any(isinstance(h, logging.FileHandler) for h in logger_obj.handlers):
            logger_obj.addHandler(fh)

    return logger_obj


def ensure_directories() -> None:
    """Create necessary directories."""
    for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def get_random_state() -> int:
    """Get reproducible random state."""
    return RANDOM_STATE


def print_banner(text: str) -> None:
    """Print formatted banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {text.center(width - 2)}")
    print("=" * width + "\n")


def robust_transform(func, *args, **kwargs) -> Any:
    """
    Robustly execute function with error handling.

    Args:
        func: Function to execute
        *args, **kwargs: Function arguments

    Returns:
        Result or None on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None
