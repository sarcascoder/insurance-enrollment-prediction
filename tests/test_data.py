"""Tests for data loading, validation, and splitting."""

from __future__ import annotations

import pytest

from src import config, data


def test_validate_accepts_good_frame(sample_df):
    # Should return the frame unchanged when the schema is correct.
    assert data.validate(sample_df) is sample_df


def test_validate_rejects_missing_column(sample_df):
    broken = sample_df.drop(columns=["salary"])
    with pytest.raises(ValueError, match="missing required columns"):
        data.validate(broken)


def test_validate_rejects_non_binary_target(sample_df):
    sample_df.loc[0, config.TARGET] = 2  # not 0/1
    with pytest.raises(ValueError, match="binary"):
        data.validate(sample_df)


def test_validate_rejects_missing_target(sample_df):
    sample_df.loc[0, config.TARGET] = None
    with pytest.raises(ValueError, match="missing values"):
        data.validate(sample_df)


def test_split_features_target_drops_id_and_target(sample_df):
    X, y = data.split_features_target(sample_df)
    # employee_id and enrolled must not leak into the features.
    assert config.ID_COLUMN not in X.columns
    assert config.TARGET not in X.columns
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert y.tolist() == [0, 1, 1, 0]


def test_load_raw_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        data.load_raw("does/not/exist.csv")


def test_get_train_test_split_shapes_and_stratification():
    # Uses the real dataset; skip cleanly if it isn't present.
    if not config.DATA_PATH.exists():
        pytest.skip("dataset CSV not available")

    X_train, X_test, y_train, y_test = data.get_train_test_split()

    total = len(X_train) + len(X_test)
    assert total == 10000
    # ~20% held out.
    assert abs(len(X_test) / total - config.TEST_SIZE) < 0.01
    # Stratification keeps the positive rate close between splits.
    assert abs(y_train.mean() - y_test.mean()) < 0.02
