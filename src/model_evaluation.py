"""Model evaluation helpers for binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: Any,
    y_predicted: Any,
    y_probability: Any | None = None,
) -> dict[str, Any]:
    """Calculate classification metrics used for the UCLA admission model."""
    y_true_array = np.asarray(y_true)
    y_predicted_array = np.asarray(y_predicted)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_array, y_predicted_array)),
        "precision": float(
            precision_score(y_true_array, y_predicted_array, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, y_predicted_array, zero_division=0)),
        "f1_score": float(f1_score(y_true_array, y_predicted_array, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_array, y_predicted_array).tolist(),
        "classification_report": classification_report(
            y_true_array,
            y_predicted_array,
            zero_division=0,
            output_dict=True,
        ),
    }

    if y_probability is not None and len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_probability))

    return metrics

