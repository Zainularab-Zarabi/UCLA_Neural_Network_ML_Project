"""Preprocessing functions based on the original UCLA notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src import config
from src.data_loader import validate_dataset
from src.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessedData:
    """Container for split and transformed data used by training."""

    x_train_raw: pd.DataFrame
    x_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    x_train_processed: np.ndarray
    x_test_processed: np.ndarray
    preprocessor: Pipeline
    feature_names: list[str]


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible with common sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)


def build_preprocessing_pipeline() -> Pipeline:
    """Build the notebook-style preprocessing pipeline.

    The original notebook uses dummy variables for University_Rating and
    Research, then applies MinMaxScaler to the full feature matrix.
    """
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(config.NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("columns", column_transformer),
            ("scaler", MinMaxScaler()),
        ]
    )


def prepare_features_and_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the binary admission target."""
    validate_dataset(dataset)
    data = dataset.copy()

    target_values = pd.to_numeric(data[config.TARGET_COLUMN], errors="coerce")
    if target_values.isna().any():
        raise ValueError("Target column contains non-numeric or missing values.")

    target = (target_values >= config.TARGET_THRESHOLD).astype(int)
    features = data.loc[:, list(config.RAW_FEATURES)].copy()

    for column in config.NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if features[list(config.NUMERIC_FEATURES)].isna().any().any():
        logger.warning("Missing numeric feature values found; imputation will be used.")

    logger.info("Prepared %s features and binary target.", len(config.RAW_FEATURES))
    return features, target


def get_transformed_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return feature names after one-hot encoding and scaling."""
    transformer = preprocessor.named_steps["columns"]
    try:
        return [str(name) for name in transformer.get_feature_names_out()]
    except AttributeError:
        logger.warning("Could not read transformed feature names from preprocessor.")
        return []


def preprocess_train_test(dataset: pd.DataFrame) -> PreprocessedData:
    """Split the data and fit preprocessing only on the training set."""
    features, target = prepare_features_and_target(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessing_pipeline()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    feature_names = get_transformed_feature_names(preprocessor)

    logger.info(
        "Preprocessed train shape %s and test shape %s.",
        x_train_processed.shape,
        x_test_processed.shape,
    )

    return PreprocessedData(
        x_train_raw=x_train.reset_index(drop=True),
        x_test_raw=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        x_train_processed=x_train_processed,
        x_test_processed=x_test_processed,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


def save_preprocessor(preprocessor: Pipeline, file_path: Path = config.PREPROCESSOR_PATH) -> None:
    """Save the fitted preprocessing pipeline with joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info("Saved preprocessing pipeline to %s", path)


def load_preprocessor(file_path: Path = config.PREPROCESSOR_PATH) -> Pipeline:
    """Load a fitted preprocessing pipeline."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline was not found: {path}")
    return joblib.load(path)


def validate_user_input(input_values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one admission profile entered by a user."""
    missing = [feature for feature in config.RAW_FEATURES if feature not in input_values]
    if missing:
        raise ValueError(f"Missing prediction inputs: {missing}")

    cleaned: dict[str, Any] = {}

    for feature in config.NUMERIC_FEATURES:
        value = float(input_values[feature])
        minimum, maximum = config.INPUT_RANGES[feature]
        if value < minimum or value > maximum:
            raise ValueError(
                f"{feature} must be between {minimum} and {maximum}."
            )
        cleaned[feature] = value

    university_rating = int(input_values["University_Rating"])
    if university_rating not in range(1, 6):
        raise ValueError("University_Rating must be one of 1, 2, 3, 4, or 5.")
    cleaned["University_Rating"] = university_rating

    research = int(input_values["Research"])
    if research not in (0, 1):
        raise ValueError("Research must be 0 or 1.")
    cleaned["Research"] = research

    return cleaned


def create_input_dataframe(input_values: Mapping[str, Any]) -> pd.DataFrame:
    """Convert validated user input into the model's raw feature format."""
    cleaned = validate_user_input(input_values)
    return pd.DataFrame([cleaned], columns=list(config.RAW_FEATURES))

