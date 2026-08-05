"""Reusable plotting functions for the Streamlit app and notebook."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(target: pd.Series):
    """Plot the binary target distribution."""
    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Class 0", "Class 1"], counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title("Admission Class Distribution")
    ax.set_ylabel("Number of Students")
    ax.set_xlabel("Class")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 2, str(value), ha="center")
    fig.tight_layout()
    return fig


def plot_loss_curve(history: pd.DataFrame):
    """Plot the neural-network loss curve saved during training."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["iteration"], history["loss"], color="#4C78A8")
    ax.set_title("MLPClassifier Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: Iterable[Iterable[int]]):
    """Plot a simple confusion matrix for binary classification."""
    values = pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    fig, ax = plt.subplots(figsize=(4, 3))
    image = ax.imshow(values.values, cmap="Blues")
    ax.set_xticks(range(values.shape[1]), values.columns)
    ax.set_yticks(range(values.shape[0]), values.index)
    ax.set_title("Confusion Matrix")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, values.iloc[row, col], ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_prediction_probabilities(predictions: pd.DataFrame):
    """Plot predicted admission probabilities for the saved test set."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(predictions["predicted_probability"], bins=12, color="#54A24B", edgecolor="white")
    ax.set_title("Predicted Admission Probabilities")
    ax.set_xlabel("Predicted probability for class 1")
    ax.set_ylabel("Number of Test Records")
    fig.tight_layout()
    return fig

