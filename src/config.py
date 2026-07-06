"""Central configuration for the insurance-enrollment project.

Keeping paths, column names, and constants in one place means the rest of the
codebase never hard-codes a string like ``"salary"`` or a magic number like the
random seed. If the dataset schema changes, this is the only file to touch.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = the directory that contains the ``src`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Raw dataset. A self-contained copy lives in the repo's ``data/`` folder so the
# project runs after a plain ``git clone`` without the original assignment repo.
DATA_PATH: Path = PROJECT_ROOT / "data" / "employee_data.csv"

# Output locations. These are created on demand by the scripts that write to
# them, so they do not need to exist ahead of time.
MODELS_DIR: Path = PROJECT_ROOT / "models"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"  # figures, metrics, reports

# MLflow tracking. Recent MLflow deprecated the plain-file store, so we use a
# local SQLite database as the tracking backend (no server to run) and a plain
# directory for logged model artifacts.
MLFLOW_DB: Path = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACTS_DIR: Path = PROJECT_ROOT / "mlartifacts"

# Trained-model artifact consumed by the API and the tests.
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET: str = "enrolled"

# ``employee_id`` is a unique identifier with no predictive signal; it is
# dropped before modelling to avoid the model memorising individual rows.
ID_COLUMN: str = "employee_id"

# Feature groups drive the preprocessing pipeline: numeric columns are scaled,
# categorical columns are one-hot encoded.
NUMERIC_FEATURES: list[str] = ["age", "salary", "tenure_years"]
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "marital_status",
    "employment_type",
    "region",
    "has_dependents",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42  # fixed seed everywhere for reproducibility
TEST_SIZE: float = 0.2  # 20% held out for the final test evaluation
CV_FOLDS: int = 5  # folds for cross-validated hyperparameter search

# MLflow experiment name.
EXPERIMENT_NAME: str = "insurance-enrollment"
