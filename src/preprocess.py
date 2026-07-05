"""Feature preprocessing pipeline.

The preprocessing is expressed as a scikit-learn ``ColumnTransformer`` so it can
be bundled *inside* the model ``Pipeline``. That is deliberate: fitting the
scaler and encoder only on the training folds (never the test set) prevents data
leakage, and shipping the transformer with the estimator means the API receives
raw JSON and the exact same transformation is applied at inference time.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer used for every model in this project.

    - Numeric features are standardised (zero mean, unit variance). Tree-based
      models do not need this, but it is essential for Logistic Regression and
      harmless for the trees, so a single shared transformer keeps things simple.
    - Categorical features are one-hot encoded. ``handle_unknown="ignore"`` means
      a category unseen during training (e.g. a new region sent to the API)
      encodes to all-zeros instead of raising an error.
    """
    numeric = Pipeline(steps=[("scaler", StandardScaler())])

    categorical = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # any unexpected column is ignored, not passed through
    )
