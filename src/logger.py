"""Logging helpers for the UCLA neural network project."""

from __future__ import annotations

import logging
from pathlib import Path

from src import config


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to both file and console."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = Path(config.LOGS_DIR) / "ucla_neural_network.log"

    existing_files = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
    ]
    existing_streams = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]

    formatter = logging.Formatter(LOG_FORMAT)

    if not existing_files:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not existing_streams:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

