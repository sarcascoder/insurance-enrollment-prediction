"""Data loading, validation, and train/test splitting.

This module is the single entry point for turning the raw CSV on disk into the
in-memory feature matrix and target vector the rest of the pipeline consumes.
Validation lives here (not in the training script) so that both training and the
API surface exactly the same guarantees about the data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Load the raw employee CSV into a DataFrame.

    Parameters
    ----------
    path:
        Location of the CSV. Defaults to the dataset shipped with the
        assignment (see :data:`config.DATA_PATH`).

    Raises
    ------
    FileNotFoundError
        If the file does not exist, with a helpful message pointing at the
        expected location.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Expected the CSV at "
            f"'{config.DATA_PATH}'. Pass a different path if it lives elsewhere."
        )
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the raw frame against the expected schema and return it unchanged.

    We fail loudly on structural problems (missing columns, an unusable target)
    rather than letting a subtle error surface deep inside model training.
    """
    expected = set(config.FEATURE_COLUMNS + [config.TARGET, config.ID_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if df[config.TARGET].isna().any():
        raise ValueError("Target column 'enrolled' contains missing values.")

    # The target must be binary 0/1 for the classifiers and metrics used here.
    target_values = set(df[config.TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target 'enrolled' must be binary 0/1, found values: {sorted(target_values)}"
        )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated frame into the feature matrix ``X`` and target ``y``.

    ``employee_id`` is dropped: it is a unique identifier with no predictive
    value, and leaving it in would let a model overfit to individual rows.
    """
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET].copy()
    return X, y


def get_train_test_split(
    path: Path | str = config.DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, validate, and split the data into stratified train/test sets.

    Stratifying on the target preserves the ~62/38 enrolled ratio in both
    splits, which keeps the held-out evaluation representative.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = validate(load_raw(path))
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
