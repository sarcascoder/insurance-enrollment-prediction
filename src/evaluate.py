"""Evaluation metrics and diagnostic plots.

Separated from ``train.py`` so the same metric definitions are used everywhere
and can be unit-tested in isolation. For an enrollment-prediction use case we
care about more than raw accuracy: the classes are imbalanced (~62/38), so we
report precision, recall, F1, ROC-AUC, average precision (PR-AUC), and the Brier
score (a proper scoring rule for probability quality). We also save a confusion
matrix, ROC / precision-recall / calibration comparisons, and permutation
feature importances for the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe on headless machines/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
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
    y_proba:  predicted probability of the positive class

    Notes
    -----
    - ``roc_auc`` and ``pr_auc`` (average precision) use probabilities; PR-AUC is
      the imbalance-aware complement to ROC-AUC.
    - ``brier`` is a proper scoring rule (mean squared error of the predicted
      probability). Lower is better; it rewards *calibrated* probabilities, not
      just correct rankings, which matters because the product predicts a
      "likelihood" of enrollment.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
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


def pr_points(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (recall, precision) arrays for a model's precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return recall, precision


def save_pr_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    ap_scores: dict[str, float],
    positive_rate: float,
    out_path: Path,
    title: str = "Precision-Recall comparison",
) -> None:
    """Overlay models' precision-recall curves — the imbalance-aware view.

    On imbalanced data ROC-AUC can look optimistic; the PR curve makes the
    precision cost of chasing recall explicit. The dashed line marks the
    no-skill baseline (a classifier that predicts the positive class rate).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap_scores[name]:.3f})")
    ax.axhline(
        positive_rate,
        linestyle="--",
        color="grey",
        linewidth=1,
        label=f"no-skill ({positive_rate:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def calibration_points(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_predicted, fraction_positive) for a reliability curve."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos


def save_calibration_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    briers: dict[str, float],
    out_path: Path,
    title: str = "Calibration (reliability) comparison",
) -> None:
    """Overlay reliability curves: predicted probability vs observed frequency.

    A perfectly calibrated model lies on the diagonal. This is the plot that
    matters when the product consumes a *probability* rather than a hard label.
    Legend annotates each model's Brier score (lower = better calibrated).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (mean_pred, frac_pos) in curves.items():
        ax.plot(
            mean_pred, frac_pos, marker="o", linewidth=2, label=f"{name} (Brier={briers[name]:.3f})"
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
