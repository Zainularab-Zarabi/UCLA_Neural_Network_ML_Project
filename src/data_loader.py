"""Dataset loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config
from src.logger import get_logger


logger = get_logger(__name__)


def validate_dataset(
    dataset: pd.DataFrame,
    required_columns: Iterable[str] = config.REQUIRED_COLUMNS,
) -> None:
    """Validate that the UCLA admissions dataset has the expected columns."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("The loaded dataset must be a pandas DataFrame.")

    if dataset.empty:
        raise ValueError("The dataset is empty.")

    duplicate_columns = dataset.columns[dataset.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate columns found: {duplicate_columns}")

    required = list(required_columns)
    missing_columns = [column for column in required if column not in dataset.columns]
    if missing_columns:
        message = f"Missing required columns: {missing_columns}"
        logger.error(message)
        raise ValueError(message)

    logger.info("Dataset validation passed with %s rows.", len(dataset))


def load_dataset(file_path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the UCLA admissions CSV and validate its required columns."""
    path = Path(file_path)
    logger.info("Loading dataset from %s", path)

    if not path.exists():
        message = f"Dataset file was not found: {path}"
        logger.error(message)
        raise FileNotFoundError(message)

    try:
        dataset = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        logger.error("Dataset file is empty: %s", path)
        raise ValueError("Dataset file is empty.") from exc
    except pd.errors.ParserError as exc:
        logger.error("Dataset file could not be parsed: %s", path)
        raise ValueError("Dataset file could not be parsed as CSV.") from exc

    validate_dataset(dataset)
    logger.info("Loaded dataset shape: %s", dataset.shape)
    return dataset

