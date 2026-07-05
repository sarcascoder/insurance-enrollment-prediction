"""Shared pytest fixtures.

A small synthetic DataFrame stands in for the real CSV so the data and
preprocessing tests run fast and don't depend on the dataset file being present.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A tiny, schema-correct frame covering both target classes."""
    return pd.DataFrame(
        {
            config.ID_COLUMN: [1, 2, 3, 4],
            "age": [25, 40, 55, 33],
            "gender": ["Male", "Female", "Other", "Female"],
            "marital_status": ["Single", "Married", "Divorced", "Widowed"],
            "salary": [40000.0, 85000.0, 120000.0, 60000.0],
            "employment_type": ["Part-time", "Full-time", "Full-time", "Contract"],
            "region": ["West", "South", "Midwest", "Northeast"],
            "has_dependents": ["No", "Yes", "Yes", "No"],
            "tenure_years": [1.0, 8.5, 20.0, 3.2],
            config.TARGET: [0, 1, 1, 0],
        }
    )
