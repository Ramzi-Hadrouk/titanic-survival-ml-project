# Titanic Survival Prediction

A machine-learning pipeline to predict passenger survival on the Titanic. This project covers data ingestion, cleaning, feature engineering, modeling, evaluation, and a minimal API for inference.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Getting Started](#getting-started)
4. [Data](#data)
5. [Handling Missing Data](#handling-missing-data)
6. [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
7. [Data Analysis](#data-analysis)
8. [Modeling](#modeling)
9. [Evaluation](#evaluation)
10. [Inference API](#inference-api)
11. [API Requests & Responses](#api-requests--responses)
12. [Utilities](#utilities)
13. [License](#license)

---

## Project Overview

We build a classifier to predict whether a passenger survived the Titanic disaster. The pipeline includes:

* Data cleaning & missing-value handling
* Feature engineering (e.g., family size, is alone)
* Encoding categorical variables
* Training a scikit-learn model (persisted as `titanic_model.pkl`)
* Automated evaluation on hold-out test set
* A minimal FastAPI service to serve predictions

---
![Survival vs Family Size](screenshots/Titanic-Survival-Prediction.png)
---
## Directory Structure

```
.
├── .venv/ # Python virtual environment
├── api/ # FastAPI inference server
├── data/
│ ├── raw/ # Original Kaggle CSVs
│ │ ├── gender_submission.csv
│ │ ├── test.csv
│ │ └── train.csv
│ └── processed/
│ └── train_processed.csv
├── data-analysis/ # Jupyter notebooks for EDA & analysis
│ └── train-data-analysis.ipynb
├── data-manipulation/ # Notebooks on cleaning & feature eng.
│ ├── 1-handling-missing-values.ipynb
│ └── 2-feature-engineering.ipynb
├── models/
│ └── titanic_model.pkl # Serialized model
├── results/
│ └── test_results.md # Evaluation metrics, confusion matrix
├── screenshots/ # EDA plots
│ ├── Survival-vs-Sex.png
│ ├── Survival-vs-Pclass.png
│ ├── Survival-vs-IsAlone.png
│ └── Survival-vs-Family-Size.png
├── src/
│ ├── utils/ # Utility scripts
│ │ ├── add_family_features.py
│ │ ├── encode_categorical_variables.py
│ │ ├── load_model.py
│ │ ├── save_models.py
│ │ └── save_test_results.py
│ ├── config.py # Paths & hyperparameters
│ ├── main.py # Orchestration entry point
│ ├── train.py # Training script
│ └── test.py # Evaluation script
├── .gitignore
├── README.md # ← you are here
└── requirements.txt
```

---

## Getting Started

### Requirements

* Python 3.8+
* pip

### Installation

1. **Clone the repo**:

```bash
git clone https://github.com/your-username/titanic-prediction.git
cd titanic-prediction
```

2. **Create & activate a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Data

### Raw Data

* `data/raw/train.csv` – Training labels & features
* `data/raw/test.csv` – Hold-out set for final submission
* `data/raw/gender_submission.csv` – Kaggle's example submission

### Processed Data

* After cleaning & feature engineering, the processed training set lives in:

```
data/processed/train_processed.csv
```

---

## Handling Missing Data

Missing data is handled using predictive techniques for patterned features like age, and removing records that do not contribute meaningfully to the model.

* See `data-manipulation/1-handling-missing-values.ipynb` for detailed implementation

---

## Feature Engineering & Preprocessing

* New features: family size, is alone, title extraction, etc.
* Implementation details in `data-manipulation/2-feature-engineering.ipynb`
* Encoding & scaling performed via utility scripts in `src/utils/encode_categorical_variables.py`

---

## Data Analysis

This section covers exploratory analysis performed on the training data. Use the Jupyter notebook `data-analysis/train-data-analysis.ipynb` to generate and review the following:

<table>
<tr>
<td><img src="screenshots/Survival-vs-Sex.png" alt="Survival vs Sex" width="300"></td>
<td><img src="screenshots/Survival-vs-Pclass.png" alt="Survival vs Pclass" width="300"></td>
</tr>
<tr>
<td><img src="screenshots/Survival-vs-IsAlone.png" alt="Survival vs IsAlone" width="300"></td>
<td><img src="screenshots/Survival-vs-Famly-Size.png" alt="Survival vs Family Size" width="300"></td>
</tr>
</table>

Feel free to add additional graphs or insights here as needed.

---

## Modeling

### Training

```bash
python src/train.py
```

This will:

1. Load `data/processed/train_processed.csv`
2. Encode data
3. Fit a classifier (e.g., LogisticRegression)
4. Serialize model to `models/titanic_model.pkl`

---

## Evaluation

```bash
python src/test.py
```

The evaluation process applies the same missing data handling, feature engineering, and encoding steps to the test data before prediction.

Results (accuracy, precision, recall, confusion matrix) are saved to `results/test_results.md`.

---

## Inference API

A simple FastAPI service lives in `api/`. To launch:

```bash
cd api
uvicorn api.api:app --reload
```

* **Endpoint 01**: `POST /predict`
* **Input**: JSON payload with passenger features
* **Output**: `{ "survived": 0|1, "probability": float }`
.

* **Endpoint 02**: `Get /model_info`
* **Input**: Empty Body
* **Output**: `{
"model_type":"LogisticRegression",
"features_used":["Pclass","Sex","Age","SibSp","Parch","Fare","FamilySize","IsAlone","Embarked"],
"training_data":"/home/ramzi/Desktop/titanic-prediction/data/processed/train_processed.csv"
}`

![Survival vs Family Size](screenshots/Swagger-Doc.png)
---

## API Requests & Responses

###  Request (CURL) to Make Prediction 

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '[{"Pclass":1,"Sex":"female","Age":30,"SibSp":1,"Parch":0,"Fare":50,"Embarked":"S"}]'
```

###  Response

```json
{
"passenger_id":0,
"survival_probability":0.9336290099510725,
"survived":true
}
```
### Request (CURL) to Get Model Information

```bash
curl -s "http://localhost:8000/model_info" 
```
### Response

```json
{
"model_type":"LogisticRegression",
"features_used":["Pclass","Sex","Age","SibSp","Parch","Fare","FamilySize","IsAlone","Embarked"],
"training_data":"/home/ramzi/Desktop/titanic-prediction/data/processed/train_processed.csv"
}
```

---

## Utilities

* **`add_family_features.py`** – Compute `FamilySize`, `IsAlone`
* **`encode_categorical_variables.py`** – One-hot / Label encode
* **`load_model.py`** – Load serialized model
* **`save_models.py`** – Helpers to persist models
* **`save_test_results.py`** – Write evaluation summary

---

## License

This project is released under the MIT License.