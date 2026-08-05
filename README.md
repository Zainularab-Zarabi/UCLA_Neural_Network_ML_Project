## UCLA Neural Network ML Project

### Project overview

This project modularizes the UCLA neural-network notebook for the CST2216 Individual Term Project. The model predicts whether a student belongs to the higher admission-chance class based on application profile features.

The original notebook converts `Admit_Chance` into a binary classification target:

- `1`: `Admit_Chance >= 0.80`
- `0`: `Admit_Chance < 0.80`

### Why a neural network is appropriate

The project follows the course notebook, which uses a feed-forward neural network through `sklearn.neural_network.MLPClassifier`. A neural network is reasonable for this learning project because the model can learn nonlinear relationships between admission inputs such as GRE score, TOEFL score, CGPA, university rating, SOP/LOR strength, and research experience.

### Dataset description

- Dataset file: `Admission.csv`
- Original shape: `500 rows x 9 columns`
- Missing values: none
- Duplicate rows: none
- Excluded identifier: `Serial_No`

### Target variable

- Target column: `Admit_Chance`
- Task: binary classification
- Threshold used by notebook: `0.80`
- Class distribution after threshold:
  - Class `0`: 345 records
  - Class `1`: 155 records

### Input features

The model uses these raw input features:

- `GRE_Score`
- `TOEFL_Score`
- `University_Rating`
- `SOP`
- `LOR`
- `CGPA`
- `Research`

### Preprocessing

The preprocessing follows the notebook:

- Drop `Serial_No`
- Convert `Admit_Chance` into a binary class using the 0.80 threshold
- Treat `University_Rating` and `Research` as categorical features
- One-hot encode the categorical features
- Scale the full feature matrix with `MinMaxScaler`
- Fit preprocessing only on training data to avoid data leakage

### Neural-network architecture

The notebook uses `MLPClassifier` with:

- One hidden layer
- 3 neurons in the hidden layer
- `adam` solver
- Batch size `50`
- Maximum iterations `200`
- Random state `123`

The final model in this project uses the notebook's second activation setting, `tanh`, because it performed better than the default `relu` run on the reproduced test set.

To meet the project requirement for a `.keras` model while preserving notebook logic, the project trains the notebook's `MLPClassifier` and exports the learned weights into an equivalent Keras model:

- Hidden layer: 3 units with `tanh`
- Output layer: 1 sigmoid unit
- Loss: binary crossentropy / binary log loss
- Optimizer: Adam
- Metric: accuracy

### Training settings

- Train/test split: 80% training, 20% testing
- Split random state: `123`
- Stratified split: yes
- Batch size: `50`
- Maximum iterations: `200`
- Selected activation: `tanh`

### Evaluation metrics

Classification metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### Actual results

Reproduced notebook-style results:

| Activation | Train accuracy | Test accuracy | Test precision | Test recall | Test F1 | ROC-AUC | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `relu` | 0.8975 | 0.8500 | 0.7500 | 0.7742 | 0.7619 | 0.9416 | `[[61, 8], [7, 24]]` |
| `tanh` | 0.9000 | 0.8700 | 0.7813 | 0.8065 | 0.7937 | 0.9434 | `[[62, 7], [6, 25]]` |

The selected model is the `tanh` MLPClassifier. The model reached the maximum 200 iterations before full convergence, which is documented as a limitation.

### Model limitations

- The dataset has only 500 records.
- The positive class is smaller than the negative class.
- The notebook changes a continuous admission chance into a binary class using a fixed threshold.
- The neural network reached `max_iter=200` before full convergence.
- This project is for education and should not be used to make real admissions decisions.

### Folder structure

```text
UCLA_Neural_Network_ML_Project/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── logs/
├── models/
├── notebooks/
├── outputs/
├── src/
└── tests/
```

### Modularization explanation

- `src/config.py`: paths, target name, feature names, and model settings
- `src/data_loader.py`: dataset loading and required-column validation
- `src/preprocessing.py`: feature/target separation, encoding, scaling, and input validation
- `src/model_builder.py`: equivalent Keras model creation and weight export
- `src/model_training.py`: complete training workflow
- `src/model_evaluation.py`: classification metrics
- `src/prediction.py`: saved-model loading and prediction
- `src/visualization.py`: plots used by the app and notebook
- `src/logger.py`: reusable logging setup

### Logging and error handling

Logs are written to `logs/ucla_neural_network.log`. The code logs dataset loading, preprocessing, training, model saving, model loading, warnings, and prediction errors. Streamlit shows user-friendly messages instead of raw tracebacks.

### Setup instructions

From the project folder:

```powershell
pip install -r requirements.txt
```

### Model-training command

```powershell
python -m src.model_training
```

This creates:

- `models/neural_network_model.keras`
- `models/notebook_mlp_classifier.joblib`
- `models/preprocessing_pipeline.joblib`
- `models/model_metrics.json`
- `outputs/training_history.csv`
- `outputs/test_predictions.csv`

### Test command

```powershell
python -m pytest
```

If Windows temp-folder permissions cause pytest issues, use:

```powershell
New-Item -ItemType Directory -Force .pytest_temp
python -m pytest --basetemp=.pytest_temp
```

### Streamlit command

```powershell
streamlit run app.py
```

### Technologies used

- Python
- pandas
- numpy
- scikit-learn
- TensorFlow/Keras
- joblib
- matplotlib
- seaborn
- Streamlit
- pytest

### GitHub link placeholder

Add your GitHub repository link here after publishing.

### Streamlit link placeholder

Add your Streamlit Community Cloud app link here after deployment.

### Author

Prepared for the CST2216 Individual Term Project.

