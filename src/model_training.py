"""Train the UCLA admission neural network and save project artifacts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from src import config
from src.data_loader import load_dataset
from src.logger import get_logger
from src.model_builder import build_keras_model, copy_sklearn_mlp_weights_to_keras
from src.model_evaluation import evaluate_binary_classifier
from src.preprocessing import preprocess_train_test, save_preprocessor


logger = get_logger(__name__)


def _train_mlp_classifier(
    x_train,
    y_train,
    activation: str,
) -> MLPClassifier:
    """Train the notebook's MLPClassifier with one hidden layer."""
    model = MLPClassifier(
        hidden_layer_sizes=config.HIDDEN_LAYER_SIZES,
        activation=activation,
        solver="adam",
        batch_size=config.BATCH_SIZE,
        max_iter=config.MAX_ITER,
        random_state=config.RANDOM_STATE,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)

    for warning in caught_warnings:
        if issubclass(warning.category, ConvergenceWarning):
            logger.warning(
                "MLPClassifier reached max_iter=%s before full convergence.",
                config.MAX_ITER,
            )

    return model


def _evaluate_model(model: MLPClassifier, split_data) -> dict[str, Any]:
    """Evaluate one fitted MLPClassifier on train and test data."""
    train_pred = model.predict(split_data.x_train_processed)
    test_pred = model.predict(split_data.x_test_processed)
    test_probability = model.predict_proba(split_data.x_test_processed)[:, 1]

    return {
        "train_accuracy": float(
            evaluate_binary_classifier(split_data.y_train, train_pred)["accuracy"]
        ),
        "test_metrics": evaluate_binary_classifier(
            split_data.y_test,
            test_pred,
            test_probability,
        ),
        "final_loss": float(model.loss_),
        "n_iter": int(model.n_iter_),
    }


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_training() -> dict[str, Any]:
    """Run the complete training workflow and save all required artifacts."""
    logger.info("Starting UCLA neural-network training workflow.")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    split_data = preprocess_train_test(dataset)

    trained_models: dict[str, MLPClassifier] = {}
    model_results: dict[str, Any] = {}

    for activation in (config.REFERENCE_ACTIVATION, config.FINAL_ACTIVATION):
        logger.info("Training MLPClassifier with activation=%s.", activation)
        model = _train_mlp_classifier(
            split_data.x_train_processed,
            split_data.y_train,
            activation=activation,
        )
        trained_models[activation] = model
        model_results[activation] = _evaluate_model(model, split_data)

    selected_activation = config.FINAL_ACTIVATION
    selected_model = trained_models[selected_activation]

    save_preprocessor(split_data.preprocessor, config.PREPROCESSOR_PATH)
    joblib.dump(selected_model, config.SKLEARN_MODEL_PATH)
    logger.info("Saved notebook MLPClassifier to %s", config.SKLEARN_MODEL_PATH)

    keras_model = build_keras_model(
        input_size=split_data.x_train_processed.shape[1],
        activation=selected_activation,
    )
    keras_model = copy_sklearn_mlp_weights_to_keras(selected_model, keras_model)
    keras_model.save(config.KERAS_MODEL_PATH)
    logger.info("Saved equivalent Keras model to %s", config.KERAS_MODEL_PATH)

    history = pd.DataFrame(
        {
            "iteration": range(1, len(selected_model.loss_curve_) + 1),
            "loss": selected_model.loss_curve_,
        }
    )
    history.to_csv(config.TRAINING_HISTORY_PATH, index=False)

    test_predictions = split_data.x_test_raw.copy()
    test_predictions["actual_class"] = split_data.y_test
    test_predictions["predicted_class"] = selected_model.predict(
        split_data.x_test_processed
    )
    test_predictions["predicted_probability"] = selected_model.predict_proba(
        split_data.x_test_processed
    )[:, 1]
    test_predictions.to_csv(config.TEST_PREDICTIONS_PATH, index=False)

    metrics = {
        "dataset": {
            "filename": config.DATA_FILENAME,
            "original_shape": list(dataset.shape),
            "modeling_shape_after_serial_drop": [int(dataset.shape[0]), 8],
            "target_column": config.TARGET_COLUMN,
            "target_threshold": config.TARGET_THRESHOLD,
            "class_counts": {
                str(key): int(value)
                for key, value in (
                    dataset[config.TARGET_COLUMN] >= config.TARGET_THRESHOLD
                ).astype(int).value_counts().sort_index().items()
            },
            "missing_values": {
                column: int(value)
                for column, value in dataset.isna().sum().to_dict().items()
            },
        },
        "preprocessing": {
            "excluded_columns": list(config.EXCLUDED_COLUMNS),
            "numeric_features": list(config.NUMERIC_FEATURES),
            "categorical_features": list(config.CATEGORICAL_FEATURES),
            "transformed_feature_names": split_data.feature_names,
            "scaler": "MinMaxScaler",
            "categorical_encoding": "OneHotEncoder",
        },
        "training_settings": {
            "task": "binary_classification",
            "train_test_split": {
                "test_size": config.TEST_SIZE,
                "random_state": config.RANDOM_STATE,
                "stratify": True,
            },
            "hidden_layer_sizes": list(config.HIDDEN_LAYER_SIZES),
            "batch_size": config.BATCH_SIZE,
            "max_iter": config.MAX_ITER,
            "solver": "adam",
            "selected_activation": selected_activation,
            "loss": "binary_log_loss",
        },
        "models": model_results,
        "selected_model": selected_activation,
        "saved_files": {
            "keras_model": str(config.KERAS_MODEL_PATH.name),
            "sklearn_reference_model": str(config.SKLEARN_MODEL_PATH.name),
            "preprocessing_pipeline": str(config.PREPROCESSOR_PATH.name),
            "training_history": str(config.TRAINING_HISTORY_PATH.name),
            "test_predictions": str(config.TEST_PREDICTIONS_PATH.name),
        },
    }

    _save_json(metrics, config.METRICS_PATH)
    logger.info("Saved metrics to %s", config.METRICS_PATH)
    return metrics


def main() -> None:
    """Entry point for command-line training."""
    try:
        metrics = run_training()
    except Exception:
        logger.exception("Training failed.")
        raise

    selected = metrics["selected_model"]
    selected_metrics = metrics["models"][selected]["test_metrics"]
    print("Training completed.")
    print(f"Selected activation: {selected}")
    print(f"Test accuracy: {selected_metrics['accuracy']:.4f}")
    print(f"Test F1-score: {selected_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()

