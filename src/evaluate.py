"""Evaluation metrics and diagnostic plots.

Separated from ``train.py`` so the same metric definitions are used everywhere
and can be unit-tested in isolation. For an enrollment-prediction use case we
care about more than raw accuracy: the classes are imbalanced (~62/38), so we
report precision, recall, F1, and ROC-AUC. We also save a confusion matrix, a
multi-model ROC comparison, and permutation feature importances for the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe on headless machines/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Compute the headline classification metrics.

    Parameters
    ----------
    y_true:   ground-truth labels (0/1)
    y_pred:   predicted labels (0/1) at the default 0.5 threshold
    y_proba:  predicted probability of the positive class (used for ROC-AUC)
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Confusion matrix"
) -> None:
    """Render and save a confusion matrix figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not enrolled", "Enrolled"], ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, out_path: Path, title: str = "ROC curve"
) -> None:
    """Render and save a ROC curve figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)  # chance line
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_roc_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    aucs: dict[str, float],
    out_path: Path,
    title: str = "ROC curve comparison",
) -> None:
    """Overlay several models' ROC curves on one axis.

    Far more informative than a single (here, perfect) curve: it shows at a
    glance how the linear baseline compares to the tree ensembles.

    Parameters
    ----------
    curves:  {model_name: (fpr, tpr)}
    aucs:    {model_name: roc_auc} used to annotate the legend
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def roc_points(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for a model, for use in a comparison plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr


def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Model-agnostic feature importance via permutation on the test set.

    Permutation importance measures how much the ROC-AUC drops when a single
    feature's values are shuffled. Unlike a tree's built-in impurity importance,
    it works for *any* model, is computed on held-out data, and is expressed on
    the original (pre-one-hot) columns — so it answers the business question
    "which employee attributes matter?" directly.

    Returns a DataFrame sorted by importance (descending).
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance(
    importance_df: pd.DataFrame, out_path: Path, title: str = "Permutation importance"
) -> None:
    """Horizontal bar chart of permutation importances (with std error bars)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = importance_df.sort_values("importance_mean")  # smallest at bottom
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color="steelblue",
    )
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a human-readable one-line summary of a metrics dict."""
    return "  ".join(f"{name}={value:.4f}" for name, value in metrics.items())
