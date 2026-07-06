# Insurance Enrollment Prediction

[![CI](https://github.com/sarcascoder/insurance-enrollment-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/sarcascoder/insurance-enrollment-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

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
| `src/preprocess.py` | `ColumnTransformer` — impute + scale numerics, impute + one-hot encode categoricals |
| `src/eda.py` | Exploratory analysis → figures + `artifacts/eda_summary.json` |
| `src/train.py` | Dummy baseline + 3 models, tune with grid search, track in MLflow, select & save best |
| `src/evaluate.py` | Metrics (acc/precision/recall/F1/ROC-AUC/PR-AUC/Brier) + confusion, ROC, PR, calibration, importance plots |
| `src/api.py` | FastAPI service serving predictions (model loaded at startup) |
| `tests/` | Pytest suite for data, preprocessing, evaluation, and the API |
| `report.md` | Findings: data observations, model choices, results, next steps |
| `Dockerfile` | Multi-stage, non-root containerised API |
| `pyproject.toml` | Packaging + ruff/pytest config |
| `.github/workflows/ci.yml` | CI: ruff lint + tests + training smoke-test on 3.10 & 3.12 |

The preprocessing (imputer → scaler/encoder) is bundled **inside** each model's
`Pipeline`, so it is learned only on training folds (no leakage) and travels with
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
- selects the best model by cross-validated ROC-AUC and saves it to
  `models/model.joblib` (the test set is used only for the final report),
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

### 5. Run the tests and linter
```bash
pip install -r requirements-dev.txt   # pytest, httpx, ruff
python -m pytest -q
ruff check src/ tests/
```

### 6. Run the API in Docker (optional, bonus)
Multi-stage build: the model is trained in the builder stage and only the trained
artifact + runtime are copied into a slim, non-root final image (no dataset or
training DB baked in).
```bash
docker build -t insurance-api .
docker run -p 8000:8000 insurance-api
# -> API at http://127.0.0.1:8000/docs
```

---

## Results at a glance

| Model | CV ROC-AUC | Test Acc | Test F1 | Test ROC-AUC | PR-AUC | Brier ↓ |
|-------|:----------:|:--------:|:-------:|:------------:|:------:|:-------:|
| Dummy (majority-class floor) | 0.5000 | 0.6175 | 0.764 | 0.5000 | 0.618 | 0.383 |
| Logistic Regression | 0.9663 | 0.894 | 0.912 | 0.971 | 0.982 | 0.072 |
| **Random Forest (selected)** | **1.0000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.002 |
| HistGradientBoosting | 0.99999769 | 0.9995 | 1.000 | 1.000 | 1.000 | 0.0005 |

Every model clears the **0.6175 majority-class floor**. The model is selected on
**cross-validated** ROC-AUC (training folds only); the test set is reported
afterwards as an unbiased estimate, never used for selection. Brier score
(probability calibration) and a PR curve are reported too — see `report.md`.

The tree-based models separate this **synthetic** dataset almost perfectly
because its target is a near-deterministic rule over `salary`,
`employment_type`, `has_dependents`, and `age`. See `report.md` for the full
discussion (including why perfect scores are expected here and would be a red
flag on real data).

---

## Reproducibility

Runs are deterministic: every stochastic step (split, model init, CV) uses a
fixed seed (`config.RANDOM_STATE = 42`), **and** all dependencies are pinned to
exact versions in `requirements.txt`. Seed + locked environment together mean the
committed `metrics.json` and figures reproduce bit-for-bit.
