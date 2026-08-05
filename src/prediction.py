"""Prediction helpers for the saved UCLA admission model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import joblib

from src import config
from src.logger import get_logger
from src.preprocessing import create_input_dataframe


logger = get_logger(__name__)


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be completed safely."""


@dataclass(frozen=True)
class PredictionArtifacts:
    """Loaded model and preprocessing objects needed for prediction."""

    model: Any
    preprocessor: Any


def load_prediction_artifacts() -> PredictionArtifacts:
    """Load the saved Keras model and fitted preprocessing pipeline."""
    if not config.KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model was not found: {config.KERAS_MODEL_PATH}. "
            "Run python -m src.model_training first."
        )
    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline was not found: {config.PREPROCESSOR_PATH}. "
            "Run python -m src.model_training first."
        )

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(config.KERAS_MODEL_PATH, compile=False)
        preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    except Exception as exc:
        logger.error("Could not load prediction artifacts: %s", exc)
        raise PredictionError(
            "The saved model files could not be loaded. Please retrain the model."
        ) from exc

    logger.info("Loaded prediction artifacts.")
    return PredictionArtifacts(model=model, preprocessor=preprocessor)


def predict_admission(
    input_values: Mapping[str, Any],
    artifacts: PredictionArtifacts | None = None,
) -> dict[str, Any]:
    """Predict the binary admission class and probability for one applicant."""
    try:
        active_artifacts = artifacts or load_prediction_artifacts()
        input_frame = create_input_dataframe(input_values)
        processed_input = active_artifacts.preprocessor.transform(input_frame)
        probability = float(
            active_artifacts.model.predict(processed_input, verbose=0).ravel()[0]
        )
        predicted_class = int(probability >= 0.5)
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise PredictionError("Prediction could not be completed.") from exc

    return {
        "predicted_class": predicted_class,
        "predicted_label": config.CLASS_LABELS[predicted_class],
        "admission_probability": probability,
    }


