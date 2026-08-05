"""Build and export the neural-network model used by the project."""

from __future__ import annotations

from typing import Any

import numpy as np

from src import config
from src.logger import get_logger


logger = get_logger(__name__)


def set_random_seeds(seed: int = config.RANDOM_STATE) -> None:
    """Set practical random seeds for reproducible neural-network behavior."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        logger.warning("TensorFlow deterministic operations could not be enabled.")


def build_keras_model(
    input_size: int,
    hidden_units: int = config.HIDDEN_LAYER_SIZES[0],
    activation: str = config.FINAL_ACTIVATION,
):
    """Create a Keras model matching the notebook MLP architecture."""
    import tensorflow as tf

    if input_size <= 0:
        raise ValueError("input_size must be a positive integer.")

    set_random_seeds(config.RANDOM_STATE)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,), name="input_features"),
            tf.keras.layers.Dense(
                hidden_units,
                activation=activation,
                name="hidden_layer_1",
            ),
            tf.keras.layers.Dense(
                1,
                activation=config.OUTPUT_ACTIVATION,
                name="admit_probability",
            ),
        ],
        name="ucla_admission_neural_network",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def copy_sklearn_mlp_weights_to_keras(sklearn_mlp: Any, keras_model: Any) -> Any:
    """Copy trained sklearn MLPClassifier weights into the equivalent Keras model."""
    if not hasattr(sklearn_mlp, "coefs_") or not hasattr(sklearn_mlp, "intercepts_"):
        raise ValueError("The sklearn MLPClassifier must be fitted before export.")

    if len(sklearn_mlp.coefs_) != 2 or len(sklearn_mlp.intercepts_) != 2:
        raise ValueError("Only the notebook's one-hidden-layer MLP can be exported.")

    weights = [
        np.asarray(sklearn_mlp.coefs_[0], dtype=np.float32),
        np.asarray(sklearn_mlp.intercepts_[0], dtype=np.float32),
        np.asarray(sklearn_mlp.coefs_[1], dtype=np.float32),
        np.asarray(sklearn_mlp.intercepts_[1], dtype=np.float32),
    ]
    keras_model.set_weights(weights)
    logger.info("Copied sklearn MLP weights into Keras model.")
    return keras_model

