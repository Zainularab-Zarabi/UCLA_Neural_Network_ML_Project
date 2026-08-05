"""Streamlit app for the UCLA neural network admission model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_dataset
from src.prediction import PredictionError, load_prediction_artifacts, predict_admission
from src.preprocessing import prepare_features_and_target
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_prediction_probabilities,
)


st.set_page_config(
    page_title="UCLA Neural Network ML Project",
    page_icon="UCLA",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    """Load the dataset once per app session."""
    return load_dataset()


@st.cache_data(show_spinner=False)
def cached_json(path: str) -> dict[str, Any]:
    """Load a JSON artifact."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def cached_csv(path: str) -> pd.DataFrame:
    """Load a CSV artifact if it exists."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def cached_prediction_artifacts():
    """Load saved model artifacts without retraining."""
    return load_prediction_artifacts()


def metric_row(metrics: dict[str, Any]) -> None:
    """Display the main model metrics."""
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    cols[1].metric("Precision", f"{metrics.get('precision', 0):.2%}")
    cols[2].metric("Recall", f"{metrics.get('recall', 0):.2%}")
    cols[3].metric("F1-score", f"{metrics.get('f1_score', 0):.2%}")
    cols[4].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")


def main() -> None:
    """Render the Streamlit application."""
    st.title("UCLA Neural Network ML Project")
    st.write(
        "This app uses the UCLA admissions dataset to classify whether a student "
        "has a higher admission chance, using the same neural-network workflow "
        "from the original notebook."
    )

    try:
        dataset = cached_dataset()
        features, binary_target = prepare_features_and_target(dataset)
    except Exception as exc:
        st.error(f"The dataset could not be loaded: {exc}")
        return

    metrics = cached_json(str(config.METRICS_PATH))
    history = cached_csv(str(config.TRAINING_HISTORY_PATH))
    predictions = cached_csv(str(config.TEST_PREDICTIONS_PATH))

    st.header("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", dataset.shape[0])
    c2.metric("Columns", dataset.shape[1])
    c3.metric("Target threshold", f">= {config.TARGET_THRESHOLD:.2f}")

    with st.expander("Dataset preview", expanded=True):
        st.dataframe(dataset.head(20), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Column Information")
        column_info = pd.DataFrame(
            {
                "column": dataset.columns,
                "dtype": [str(dtype) for dtype in dataset.dtypes],
                "missing_values": dataset.isna().sum().values,
            }
        )
        st.dataframe(column_info, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Target Distribution")
        st.pyplot(plot_class_distribution(binary_target))

    st.subheader("Descriptive Statistics")
    st.dataframe(dataset.describe().T, use_container_width=True)

    st.header("Model Methodology")
    st.write(
        "The original notebook drops `Serial_No`, converts `Admit_Chance` into "
        "a binary class at the 0.80 threshold, one-hot encodes "
        "`University_Rating` and `Research`, and applies `MinMaxScaler` after "
        "the train/test split to avoid data leakage."
    )
    st.write(
        "The notebook model is an `MLPClassifier` with one hidden layer of "
        "3 neurons, batch size 50, max_iter 200, the Adam solver, and the final "
        "tested activation function `tanh`. The trained weights are exported "
        "into an equivalent Keras model for deployment."
    )

    if metrics:
        selected = metrics.get("selected_model", config.FINAL_ACTIVATION)
        selected_metrics = metrics.get("models", {}).get(selected, {}).get("test_metrics", {})

        st.header("Actual Model Results")
        metric_row(selected_metrics)
        model_rows = []
        for name, result in metrics.get("models", {}).items():
            test_metrics = result.get("test_metrics", {})
            model_rows.append(
                {
                    "activation": name,
                    "train_accuracy": result.get("train_accuracy"),
                    "test_accuracy": test_metrics.get("accuracy"),
                    "test_f1": test_metrics.get("f1_score"),
                    "roc_auc": test_metrics.get("roc_auc"),
                    "final_loss": result.get("final_loss"),
                }
            )
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        col_loss, col_matrix = st.columns(2)
        with col_loss:
            st.subheader("Training Loss")
            if not history.empty:
                st.pyplot(plot_loss_curve(history))
            else:
                st.info("Training history is not available yet.")

        with col_matrix:
            st.subheader("Confusion Matrix")
            matrix = selected_metrics.get("confusion_matrix")
            if matrix:
                st.pyplot(plot_confusion_matrix(matrix))
            else:
                st.info("Confusion matrix is not available yet.")

        if not predictions.empty:
            st.subheader("Test Prediction Probabilities")
            st.pyplot(plot_prediction_probabilities(predictions))
    else:
        st.warning("Model metrics are missing. Run `python -m src.model_training` first.")

    st.header("Try a Prediction")
    st.write("Enter one student profile using the same input features as the notebook.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            gre = st.number_input("GRE score (290-340)", min_value=290, max_value=340, value=317, step=1)
            toefl = st.number_input("TOEFL score (92-120)", min_value=92, max_value=120, value=107, step=1)
            cgpa = st.number_input("CGPA (6.80-9.92)", min_value=6.80, max_value=9.92, value=8.56, step=0.01, format="%.2f")
            research_label = st.selectbox("Research experience", ["Yes", "No"], index=0)
        with col_b:
            university_rating = st.selectbox("University rating", [1, 2, 3, 4, 5], index=2)
            sop = st.number_input("SOP strength (1.0-5.0)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
            lor = st.number_input("LOR strength (1.0-5.0)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)

        submitted = st.form_submit_button("Predict admission class")

    if submitted:
        user_input = {
            "GRE_Score": gre,
            "TOEFL_Score": toefl,
            "SOP": sop,
            "LOR": lor,
            "CGPA": cgpa,
            "University_Rating": university_rating,
            "Research": 1 if research_label == "Yes" else 0,
        }
        try:
            artifacts = cached_prediction_artifacts()
            result = predict_admission(user_input, artifacts)
            st.success(result["predicted_label"])
            st.metric(
                "Predicted probability for class 1",
                f"{result['admission_probability']:.2%}",
            )
        except (ValueError, FileNotFoundError, PredictionError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Something went wrong while making the prediction.")

    st.header("Project Limitations")
    st.warning(
    """
    This application is an educational machine-learning project and should not be
    used to make real university admission decisions.

    The model was trained on only 500 historical records and does not include all
    factors considered by universities. The original admission chance was also
    converted into two classes using a fixed 0.80 threshold.

    A predicted probability represents the model's confidence in its classification.
    It is not the student's actual probability of admission.
    """
)

if __name__ == "__main__":
    main()

