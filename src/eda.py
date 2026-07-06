"""Exploratory data analysis.

Run with::

    python -m src.eda

Generates the summary statistics and figures referenced in ``report.md`` and
writes them to ``artifacts/``. Kept as a script (not a notebook) so the analysis
is version-controlled, diff-able, and runs the same way in CI as on a laptop.
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config, data

logger = logging.getLogger(__name__)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the numeric features."""
    return df[config.NUMERIC_FEATURES].describe().T


def enrollment_rate_by_category(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Enrollment rate broken down by each categorical feature.

    This is the most useful signal for the report: it shows which groups are
    more likely to enroll, hinting at what the model will learn.
    """
    rates = {}
    for col in config.CATEGORICAL_FEATURES:
        rates[col] = df.groupby(col)[config.TARGET].mean().sort_values(ascending=False)
    return rates


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Histogram of each numeric feature split by enrollment outcome."""
    fig, axes = plt.subplots(1, len(config.NUMERIC_FEATURES), figsize=(14, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES, strict=True):
        for label, group in df.groupby(config.TARGET):
            ax.hist(group[col], bins=30, alpha=0.55, label=f"enrolled={label}")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend()
    fig.suptitle("Numeric feature distributions by enrollment")
    fig.tight_layout()
    out = config.ARTIFACTS_DIR / "numeric_distributions.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_enrollment_by_category(rates: dict[str, pd.Series]) -> None:
    """Bar chart of enrollment rate per category for every categorical feature."""
    n = len(rates)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (col, series) in zip(axes, rates.items(), strict=True):
        series.plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.set_ylabel("enrollment rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Enrollment rate by category")
    fig.tight_layout()
    out = config.ARTIFACTS_DIR / "enrollment_by_category.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = data.validate(data.load_raw())
    logger.info("Dataset shape: %s", df.shape)

    # Target balance.
    balance = df[config.TARGET].value_counts(normalize=True).to_dict()
    logger.info("Target balance (enrolled): %s", balance)

    # Numeric summary.
    logger.info("Numeric summary:\n%s", numeric_summary(df))

    # Category-level enrollment rates.
    rates = enrollment_rate_by_category(df)
    for col, series in rates.items():
        logger.info("Enrollment rate by %s:\n%s", col, series.round(3))

    # Persist a compact JSON summary for the report.
    summary = {
        "shape": list(df.shape),
        "target_balance": balance,
        "numeric_summary": json.loads(numeric_summary(df).to_json()),
        "enrollment_rate_by_category": {
            col: series.round(4).to_dict() for col, series in rates.items()
        },
    }
    out = config.ARTIFACTS_DIR / "eda_summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved %s", out)

    # Figures.
    plot_numeric_distributions(df)
    plot_enrollment_by_category(rates)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
