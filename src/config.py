"""Project settings and reusable paths for the UCLA admission model."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_FILENAME = "Admission.csv"
ORIGINAL_NOTEBOOK_FILENAME = "UCLA_Neural_Networks_Solution.ipynb"
CLEANED_NOTEBOOK_FILENAME = "UCLA_Neural_Network_Cleaned.ipynb"

DATA_PATH = DATA_DIR / DATA_FILENAME
ORIGINAL_NOTEBOOK_PATH = NOTEBOOKS_DIR / ORIGINAL_NOTEBOOK_FILENAME
CLEANED_NOTEBOOK_PATH = NOTEBOOKS_DIR / CLEANED_NOTEBOOK_FILENAME

KERAS_MODEL_PATH = MODELS_DIR / "neural_network_model.keras"
SKLEARN_MODEL_PATH = MODELS_DIR / "notebook_mlp_classifier.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.csv"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"

TARGET_COLUMN = "Admit_Chance"
TARGET_THRESHOLD = 0.8
EXCLUDED_COLUMNS = ("Serial_No",)

NUMERIC_FEATURES = ("GRE_Score", "TOEFL_Score", "SOP", "LOR", "CGPA")
CATEGORICAL_FEATURES = ("University_Rating", "Research")
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = EXCLUDED_COLUMNS + RAW_FEATURES + (TARGET_COLUMN,)

RANDOM_STATE = 123
TEST_SIZE = 0.2
BATCH_SIZE = 50
MAX_ITER = 200
HIDDEN_LAYER_SIZES = (3,)
REFERENCE_ACTIVATION = "relu"
FINAL_ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "sigmoid"
LEARNING_RATE = 0.001

CLASS_LABELS = {
    0: "Lower admission chance (< 0.80)",
    1: "Higher admission chance (>= 0.80)",
}

# These ranges are from the provided Admission.csv dataset.
INPUT_RANGES = {
    "GRE_Score": (290, 340),
    "TOEFL_Score": (92, 120),
    "SOP": (1.0, 5.0),
    "LOR": (1.0, 5.0),
    "CGPA": (6.8, 9.92),
    "University_Rating": (1, 5),
    "Research": (0, 1),
}

DEFAULT_INPUT = {
    "GRE_Score": 317,
    "TOEFL_Score": 107,
    "SOP": 3.5,
    "LOR": 3.5,
    "CGPA": 8.56,
    "University_Rating": 3,
    "Research": 1,
}

