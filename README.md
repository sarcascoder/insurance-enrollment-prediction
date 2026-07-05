# Insurance Enrollment Prediction

An end-to-end machine-learning pipeline that predicts whether an employee will
opt in to a new voluntary insurance product, from census-style demographic and
employment data.

> **Note on this repository.** This project answers the ML take-home brief in
> [`ASSIGNMENT.md`](ASSIGNMENT.md) (predicting insurance enrollment). The dataset
> lives at [`data/employee_data.csv`](data/employee_data.csv), so the project runs
> straight after `git clone`.

---

## What's inside

| Path | Purpose |
|------|---------|
| `src/config.py` | Central config: paths, feature groups, seeds, constants |
| `src/data.py` | Load, validate, and stratified train/test split |
| `src/preprocess.py` | `ColumnTransformer` — scale numerics, one-hot encode categoricals |
| `src/eda.py` | Exploratory analysis → figures + `artifacts/eda_summary.json` |
| `src/train.py` | Train 3 models, tune with grid search, track in MLflow, select & save best |
| `src/evaluate.py` | Metrics (accuracy/precision/recall/F1/ROC-AUC) + confusion-matrix & ROC plots |
| `src/api.py` | FastAPI service serving predictions from the saved model |
| `tests/` | Pytest suite for data, preprocessing, and the API |
| `report.md` | Findings: data observations, model choices, results, next steps |

The preprocessing is bundled **inside** each model's `Pipeline`, so the fitted
scaler/encoder are learned only on training folds (no leakage) and travel with
the saved model — the API takes raw JSON and applies the identical transform.

---

## Setup

Requires **Python 3.10+** (developed on 3.12).

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

---

## How to run

All commands are run from the project root. The `src` package is invoked with
`python -m` so imports resolve correctly.

### 1. Exploratory data analysis (optional)
```bash
python -m src.eda
```
Prints summary stats and writes figures + `eda_summary.json` to `artifacts/`.

### 2. Train, tune, and select the model
```bash
python -m src.train
```
This:
- loads and stratified-splits the data (80/20),
- trains Logistic Regression, Random Forest, and HistGradientBoosting, each
  tuned with 5-fold cross-validated grid search,
- logs every run to MLflow,
- selects the best model by test ROC-AUC and saves it to `models/model.joblib`,
- writes `artifacts/metrics.json`, a confusion matrix, and a ROC curve.

### 3. Inspect experiments in MLflow (optional, bonus)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open <http://127.0.0.1:5000>.

### 4. Serve predictions via the REST API (bonus)
```bash
uvicorn src.api:app --reload
```
Open the interactive docs at <http://127.0.0.1:8000/docs>.

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 42, "gender": "Female", "marital_status": "Married",
        "salary": 85000, "employment_type": "Full-time",
        "region": "West", "has_dependents": "Yes", "tenure_years": 6.5
      }'
# -> {"enrolled": 1, "probability": 1.0}
```

Endpoints:
- `GET  /health` — liveness check + whether the model is loaded
- `POST /predict` — single-employee prediction
- `POST /predict/batch` — list of employees in one call

> Run `python -m src.train` **before** starting the API so `models/model.joblib`
> exists; otherwise `/predict` returns `503`.

### 5. Run the tests
```bash
python -m pytest -q
```

---

## Results at a glance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| Logistic Regression | 0.894 | 0.936 | 0.889 | 0.912 | 0.971 |
| **Random Forest (selected)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| HistGradientBoosting | 0.9995 | 1.000 | 0.999 | 1.000 | 1.000 |

The tree-based models separate this **synthetic** dataset almost perfectly
because its target is a near-deterministic rule over `salary`,
`employment_type`, `has_dependents`, and `age`. See `report.md` for the full
discussion (including why perfect scores are expected here and would be a red
flag on real data).

---

## Reproducibility

Every stochastic step (split, model init, CV) uses a fixed seed
(`config.RANDOM_STATE = 42`), so runs are deterministic.
