"""Evaluation metrics and diagnostic plots.

Separated from ``train.py`` so the same metric definitions are used everywhere
and can be unit-tested in isolation. For an enrollment-prediction use case we
care about more than raw accuracy: the classes are imbalanced (~62/38), so we
report precision, recall, F1, and ROC-AUC, and we save a confusion matrix and
ROC curve for the written report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe on headless machines/CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a human-readable one-line summary of a metrics dict."""
    return "  ".join(f"{name}={value:.4f}" for name, value in metrics.items())
