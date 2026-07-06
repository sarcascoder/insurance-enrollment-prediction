"""Feature preprocessing pipeline.

The preprocessing is expressed as a scikit-learn ``ColumnTransformer`` so it can
be bundled *inside* the model ``Pipeline``. That is deliberate: fitting the
imputer, scaler, and encoder only on the training folds (never the test set)
prevents data leakage, and shipping the transformer with the estimator means the
API receives raw JSON and the exact same transformation is applied at inference
time.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer used for every model in this project.

    Numeric branch: median imputation -> standardisation.
    Categorical branch: most-frequent imputation -> one-hot encoding.

    Design notes
    ------------
    - **Imputation is included even though the supplied CSV has no missing
      values.** The brief states the data simulates "what's typically collected
      during group benefits enrollment", where missing salary/tenure/marital
      status are the norm. Without imputers a single NaN would silently
      propagate to a NaN prediction, so they are part of the contract, not an
      afterthought. Median (numeric) and most-frequent (categorical) are robust,
      leakage-free defaults fit per training fold.
    - **Standardisation** is essential for Logistic Regression and harmless for
      the tree models, so one shared transformer keeps the code simple. (For a
      trees-only deployment it could be dropped.)
    - **``handle_unknown="ignore"``** encodes an unseen category (e.g. a new
      region posted to the API) as all-zeros instead of raising. This is why we
      do *not* pass ``drop="first"``: scikit-learn forbids combining ``drop``
      with ``handle_unknown="ignore"`` (a dropped level and an unknown level
      would both map to all-zeros and become indistinguishable). We accept the
      extra dummy column and rely on L2 regularisation to handle the resulting
      collinearity in the linear model — robustness at inference is worth more
      here than shaving one column.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # any unexpected column is ignored, not passed through
    )
