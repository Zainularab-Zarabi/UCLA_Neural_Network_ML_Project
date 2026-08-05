## UCLA Neural Network ML Project

This project uses a neural network to predict whether a student belongs to the higher university admission-chance category based on academic application information.

The original Jupyter Notebook was reorganized into reusable Python modules, automated tests, saved model files, logging, error handling, and an interactive Streamlit application.

### Project Links

- [Live Streamlit Application](PASTE_STREAMLIT_LINK_HERE)

### Project Overview

The dataset contains university admission information such as GRE score, TOEFL score, university rating, statement-of-purpose rating, letter-of-recommendation rating, CGPA, research experience, and admission chance.

The original `Admit_Chance` value is continuous. Following the notebook logic, it is converted into a binary classification target:

- Class `1`: `Admit_Chance >= 0.80`
- Class `0`: `Admit_Chance < 0.80`

The model therefore predicts whether a student belongs to the higher admission-chance class.

### Machine-Learning Task

This is a supervised binary classification problem.

The model learns from historical records containing both:

- Student application features
- The converted admission class

After training, it can predict the admission-chance category of a new application.

### Dataset

The project uses:

```text
data/Admission.csv
```

#### Dataset Information

- Rows: 500
- Columns: 9
- Missing values: none
- Duplicate rows: none
- Target column: `Admit_Chance`
- Excluded identifier: `Serial_No`

#### Dataset Columns

- `Serial_No`
- `GRE_Score`
- `TOEFL_Score`
- `University_Rating`
- `SOP`
- `LOR`
- `CGPA`
- `Research`
- `Admit_Chance`

### Target Variable

The original target is:

```text
Admit_Chance
```

It is converted into two classes using the notebook threshold:

```text
Admit_Chance >= 0.80 → Class 1
Admit_Chance < 0.80  → Class 0
```

#### Target Distribution

- Class `0`: 345 records
- Class `1`: 155 records

The positive class is smaller than the negative class, so the dataset has some class imbalance.

### Input Features

The neural network uses:

- `GRE_Score`
- `TOEFL_Score`
- `University_Rating`
- `SOP`
- `LOR`
- `CGPA`
- `Research`

`Serial_No` is excluded because it is only a row identifier and does not provide useful predictive information.

### Data Preprocessing

The preprocessing workflow:

- Loads and validates the dataset
- Removes `Serial_No`
- Converts `Admit_Chance` into a binary class
- Separates input features and target
- Treats `University_Rating` and `Research` as categorical features
- One-hot encodes categorical features
- Scales the complete feature matrix using `MinMaxScaler`
- Fits preprocessing only on the training data
- Applies the saved preprocessing steps to testing and prediction data

Fitting preprocessing only on the training data helps prevent data leakage.

### Why a Neural Network Is Used

The project follows the original notebook, which uses Scikit-learn’s `MLPClassifier`.

A neural network is suitable for this coursework project because it can learn nonlinear relationships between features such as:

- GRE score
- TOEFL score
- CGPA
- University rating
- SOP strength
- LOR strength
- Research experience

The model uses a small architecture so it remains close to the original notebook.

### Neural-Network Architecture

The selected neural network contains:

- Input layer based on the processed features
- One hidden layer
- Three neurons in the hidden layer
- `tanh` activation function
- One output unit for binary classification
- Adam solver
- Batch size of 50
- Maximum of 200 iterations
- Random state of 123

#### Equivalent Keras Model

The main notebook logic uses `MLPClassifier`.

The learned Scikit-learn weights are also exported into an equivalent Keras model containing:

- Hidden layer: 3 neurons with `tanh`
- Output layer: 1 neuron with `sigmoid`
- Loss: binary cross-entropy
- Optimizer: Adam
- Metric: accuracy

The exported Keras model produces probabilities that closely match the Scikit-learn model.

### Training Settings

- Training data: 80%
- Testing data: 20%
- Split random state: 123
- Stratified split: yes
- Hidden layers: one
- Hidden-layer neurons: three
- Batch size: 50
- Maximum iterations: 200
- Selected activation: `tanh`
- Solver: Adam

A stratified split helps preserve the class distribution in both the training and testing datasets.

### Models Compared

Two activation settings from the notebook were compared:

- `relu`
- `tanh`

The remaining architecture and training settings were kept consistent.

### Evaluation Metrics

The classification models were evaluated using:

- Training accuracy
- Testing accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### Model Results

| Activation | Train accuracy | Test accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `relu` | 89.75% | 85.00% | 75.00% | 77.42% | 76.19% | 94.16% |
| `tanh` | 90.00% | 87.00% | 78.13% | 80.65% | 79.37% | 94.34% |

### Selected Model

The selected model is the neural network using:

```text
Activation: tanh
```

It achieved:

- Test accuracy: 87.00%
- Precision: 78.13%
- Recall: 80.65%
- F1-score: 79.37%
- ROC-AUC: 94.34%

The `tanh` model was selected because it performed better than the `relu` model on the reproduced testing dataset.

### Confusion Matrix

The selected model produced:

```text
[[62, 7],
 [ 6, 25]]
```

This means:

- 62 class-0 records were correctly predicted
- 25 class-1 records were correctly predicted
- 7 class-0 records were incorrectly predicted as class 1
- 6 class-1 records were incorrectly predicted as class 0

### Project Structure

```text
UCLA_Neural_Network_ML_Project/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── Admission.csv
├── notebooks/
│   └── UCLA_Neural_Network.ipynb
├── models/
│   ├── neural_network_model.keras
│   ├── notebook_mlp_classifier.joblib
│   ├── preprocessing_pipeline.joblib
│   └── model_metrics.json
├── logs/
│   └── ucla_neural_network.log
├── outputs/
│   ├── training_history.csv
│   └── test_predictions.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_builder.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction.py
│   ├── visualization.py
│   └── logger.py
└── tests/
    ├── __init__.py
    └── test_preprocessing.py
```

### Code Modularization

The original notebook code was divided into separate reusable modules.

#### `config.py`

Stores project paths, feature names, target settings, random seeds, and model configuration.

#### `data_loader.py`

Loads the admission dataset and validates the required columns.

#### `preprocessing.py`

Creates the binary target, removes the identifier, encodes categorical features, scales the data, and validates prediction inputs.

#### `model_builder.py`

Builds the equivalent Keras neural-network model and transfers the trained MLP weights.

#### `model_training.py`

Runs the complete training workflow, compares activation functions, selects the final model, and saves all model artifacts.

#### `model_evaluation.py`

Calculates classification metrics, confusion matrices, and evaluation reports.

#### `prediction.py`

Loads the saved model and preprocessing pipeline and generates predictions for new student information.

#### `visualization.py`

Creates reusable charts for model results, training history, and prediction evaluation.

#### `logger.py`

Creates reusable project logging for information, warnings, and errors.

### Saved Model Files

Running the training workflow creates:

```text
models/neural_network_model.keras
models/notebook_mlp_classifier.joblib
models/preprocessing_pipeline.joblib
models/model_metrics.json
```

#### Model File Purposes

- `neural_network_model.keras` contains the equivalent Keras neural network.
- `notebook_mlp_classifier.joblib` contains the trained Scikit-learn MLPClassifier.
- `preprocessing_pipeline.joblib` contains the fitted encoding and scaling steps.
- `model_metrics.json` contains the actual evaluation results.

### Generated Outputs

The project also creates:

```text
outputs/training_history.csv
outputs/test_predictions.csv
```

These files preserve the model-training history and testing predictions for review.

### Logging and Error Handling

Project logs are written to:

```text
logs/ucla_neural_network.log
```

#### Logged Activities

The log may record:

- Dataset loading
- Dataset validation
- Preprocessing
- Train/test splitting
- Model training
- Model evaluation
- Model saving
- Model loading
- Prediction activity
- Warnings
- Errors

#### Error Handling

The project includes handling for:

- Missing dataset files
- Missing required columns
- Invalid feature values
- Missing model files
- Preprocessing errors
- Model-loading errors
- Prediction errors

The Streamlit application displays understandable messages instead of raw Python tracebacks.

### Setup Instructions

This project requires Python 3.12 because TensorFlow was not available for the Python 3.14 environment used on the development computer.

#### Windows

Create a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv312
```

Allow environment activation in the current PowerShell terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell
.\.venv312\Scripts\Activate.ps1
```

Upgrade Pip:

```powershell
python -m pip install --upgrade pip
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Install Jupyter kernel support when running the notebook:

```powershell
python -m pip install ipykernel
```

#### macOS or Linux

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

### Run the Notebook

Open:

```text
notebooks/UCLA_Neural_Network.ipynb
```

Select the Python interpreter from:

```text
.venv312
```

Then run all notebook cells from beginning to end.

### Train the Model

Run:

```powershell
python -m src.model_training
```

The expected final training summary includes:

```text
Training completed.
Selected activation: tanh
Test accuracy: 0.8700
Test F1-score: 0.7937
```

TensorFlow may display CPU and GPU information messages before training. These messages do not mean that the training failed.

### Run the Automated Tests

Run:

```powershell
python -m pytest
```

If Windows blocks the normal Pytest temporary folder, run:

```powershell
New-Item -ItemType Directory -Force .pytest_temp
python -m pytest --basetemp=.pytest_temp
```

The completed test run produced:

```text
6 passed
```

The tests check important project functions such as:

- Dataset loading
- Required-column validation
- Feature and target separation
- Preprocessing
- Prediction-input validation
- Model-input formatting

### Run the Streamlit Application

Run:

```powershell
python -m streamlit run app.py
```

The application loads the saved model and preprocessing files. It does not retrain the model each time a user enters new information.

### Streamlit Application Features

The application includes:

- Project overview
- Dataset information
- Dataset preview
- Missing-value summary
- Target-class distribution
- Preprocessing explanation
- Neural-network architecture
- Training settings
- Model comparison
- Selected-model metrics
- Confusion matrix
- Training and evaluation charts
- New student input form
- Admission-class prediction
- Predicted probability
- Project limitations

### Prediction Inputs

The application accepts:

- GRE score
- TOEFL score
- University rating
- SOP rating
- LOR rating
- CGPA
- Research experience

The saved preprocessing pipeline converts these inputs into the same format used during model training.

The neural network then predicts whether the student belongs to:

- Class `0`: admission chance below 0.80
- Class `1`: admission chance of at least 0.80

### Improvements Made to the Notebook

The modular project includes the following improvements:

- Divided notebook code into reusable Python modules
- Added an organized VS Code project structure
- Added required-column validation
- Added train-only preprocessing to reduce data leakage
- Added reusable prediction functions
- Added model comparison
- Added classification metrics beyond accuracy
- Saved the preprocessing pipeline
- Saved the Scikit-learn neural network
- Exported an equivalent Keras model
- Saved model metrics and test predictions
- Added automated tests
- Added logging
- Added error handling
- Added a Streamlit application
- Added user-friendly prediction inputs
- Added reproducible random states

### Project Limitations

- The dataset contains only 500 records.
- The positive class is smaller than the negative class.
- The original continuous admission chance is converted into a binary class.
- The 0.80 threshold is fixed by the notebook logic.
- The model reached the maximum 200 iterations before complete convergence.
- The network contains only one small hidden layer.
- The dataset may not represent all universities or admission systems.
- Admissions decisions involve factors that are not included in this dataset.

This project is intended for education and demonstration. It should not be used to make real university admission decisions.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- Pytest
- Jupyter Notebook
- VS Code
- Git
- GitHub

### Author

Zainularab Zarabi  
Business Intelligence Systems Infrastructure  


