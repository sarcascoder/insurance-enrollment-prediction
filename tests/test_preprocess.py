"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from src import config
from src.preprocess import build_preprocessor


def test_preprocessor_transforms_to_numeric_matrix(sample_df):
    X = sample_df[config.FEATURE_COLUMNS]
    pre = build_preprocessor()
    out = pre.fit_transform(X)

    # Output is a dense numeric array with one row per input row.
    assert out.shape[0] == len(X)
    assert np.issubdtype(out.dtype, np.floating)

    # 3 scaled numeric columns + one-hot columns for every observed category.
    n_categorical = sum(sample_df[c].nunique() for c in config.CATEGORICAL_FEATURES)
    assert out.shape[1] == len(config.NUMERIC_FEATURES) + n_categorical


def test_numeric_columns_are_standardised(sample_df):
    X = sample_df[config.FEATURE_COLUMNS]
    pre = build_preprocessor()
    out = pre.fit_transform(X)

    # The first 3 columns are the scaled numeric features -> ~zero mean.
    numeric_block = out[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-9)


def test_handles_unknown_category_at_transform_time(sample_df):
    X = sample_df[config.FEATURE_COLUMNS]
    pre = build_preprocessor()
    pre.fit(X)

    # A region unseen during fit must not raise (handle_unknown="ignore").
    novel = X.iloc[[0]].copy()
    novel.loc[:, "region"] = "Atlantis"
    out = pre.transform(novel)
    assert out.shape[0] == 1
