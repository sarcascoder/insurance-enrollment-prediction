"""Tests for the evaluation metrics and interpretability helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config, evaluate
from src.preprocess import build_preprocessor


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = y_true.copy()
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])
    m = evaluate.compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    # Every metric is a plain float in [0, 1] (JSON-serialisable).
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in m.values())


def test_roc_points_shapes():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr = evaluate.roc_points(y_true, y_proba)
    assert len(fpr) == len(tpr)
    assert fpr[0] == 0.0 and tpr[-1] == 1.0


def test_permutation_importance_ranks_features(sample_df):
    # Fit a real pipeline and check the importance frame is well-formed.
    X = sample_df[config.FEATURE_COLUMNS]
    y = sample_df[config.TARGET]
    model = Pipeline(
        steps=[("preprocess", build_preprocessor()),
               ("clf", LogisticRegression(max_iter=1000))]
    )
    model.fit(X, y)

    imp = evaluate.compute_permutation_importance(model, X, y, n_repeats=3)
    # One row per original feature, sorted descending by mean importance.
    assert set(imp["feature"]) == set(config.FEATURE_COLUMNS)
    assert imp["importance_mean"].is_monotonic_decreasing
