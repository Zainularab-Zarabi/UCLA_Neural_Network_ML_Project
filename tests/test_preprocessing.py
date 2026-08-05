"""Fast tests for dataset validation and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import load_dataset, validate_dataset
from src.preprocessing import (
    build_preprocessing_pipeline,
    create_input_dataframe,
    prepare_features_and_target,
)


def tiny_dataset() -> pd.DataFrame:
    """Return a small valid dataset with both target classes."""
    return pd.DataFrame(
        {
            "Serial_No": [1, 2, 3, 4],
            "GRE_Score": [337, 324, 316, 290],
            "TOEFL_Score": [118, 107, 104, 92],
            "University_Rating": [4, 4, 3, 1],
            "SOP": [4.5, 4.0, 3.0, 1.5],
            "LOR": [4.5, 4.5, 3.5, 2.0],
            "CGPA": [9.65, 8.87, 8.0, 7.2],
            "Research": [1, 1, 1, 0],
            "Admit_Chance": [0.92, 0.76, 0.81, 0.50],
        }
    )


def test_load_dataset_reads_project_csv() -> None:
    dataset = load_dataset(config.DATA_PATH)
    assert dataset.shape == (500, 9)
    assert config.TARGET_COLUMN in dataset.columns


def test_missing_file_raises_clear_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("definitely_missing_admission_file.csv"))


def test_required_column_validation_fails_for_missing_target() -> None:
    dataset = tiny_dataset().drop(columns=[config.TARGET_COLUMN])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(dataset)


def test_feature_target_separation_uses_notebook_threshold() -> None:
    features, target = prepare_features_and_target(tiny_dataset())
    assert "Serial_No" not in features.columns
    assert list(features.columns) == list(config.RAW_FEATURES)
    assert target.tolist() == [1, 0, 1, 0]


def test_preprocessor_outputs_numeric_matrix() -> None:
    features, _ = prepare_features_and_target(tiny_dataset())
    preprocessor = build_preprocessing_pipeline()
    processed = preprocessor.fit_transform(features)
    assert processed.shape[0] == len(features)
    assert processed.shape[1] >= len(config.NUMERIC_FEATURES)
    assert np.isfinite(processed).all()


def test_invalid_prediction_input_is_rejected() -> None:
    bad_input = dict(config.DEFAULT_INPUT)
    bad_input["GRE_Score"] = 400
    with pytest.raises(ValueError, match="GRE_Score"):
        create_input_dataframe(bad_input)

