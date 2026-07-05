"""Model training, hyperparameter tuning, experiment tracking, and selection.

Run with::

    python -m src.train

The script:
  1. loads and splits the data (stratified train/test),
  2. defines several candidate models, each wrapped in a full
     preprocess -> classifier ``Pipeline``,
  3. tunes each with cross-validated grid search, logging every run to MLflow,
  4. evaluates the tuned models on the held-out test set,
  5. picks the best model by ROC-AUC and saves it, plus metrics and figures,
     for the API and the report to consume.

Wrapping preprocessing inside each Pipeline is what makes cross-validation and
the saved artifact leak-free and self-contained: the scaler/encoder are refit on
each training fold and travel with the model to inference time.
"""

from __future__ import annotations

import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from . import config, data, evaluate
from .preprocess import build_preprocessor


def build_candidates() -> dict[str, dict]:
    """Define the candidate models and their hyperparameter search grids.

    Each candidate is a full pipeline: the shared preprocessor followed by a
    classifier. The grids are intentionally small — enough to show tuning adds
    value without turning a take-home into an hours-long search.

    Model rationale:
      - Logistic Regression: fast, interpretable linear baseline. If a linear
        model already does well, that tells us the signal is mostly additive.
      - Random Forest: captures non-linearities and feature interactions with
        little tuning; robust to the mixed feature types here.
      - HistGradientBoosting: sklearn's boosted-trees implementation, usually
        the strongest tabular performer, and built in (no extra dependency).
    """
    preprocessor = build_preprocessor()

    def pipe(clf) -> Pipeline:
        return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    return {
        "logistic_regression": {
            "pipeline": pipe(
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # counter the 62/38 class imbalance
                    random_state=config.RANDOM_STATE,
                )
            ),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "random_forest": {
            "pipeline": pipe(
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            "param_grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 5],
            },
        },
        "hist_gradient_boosting": {
            "pipeline": pipe(
                HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
            ),
            "param_grid": {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [None, 6],
                "clf__max_iter": [200, 400],
            },
        },
    }


def tune_and_evaluate(
    name: str,
    spec: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float], dict]:
    """Grid-search one candidate, log it to MLflow, and score it on the test set.

    Returns the fitted best estimator, its test metrics, and the best params.
    """
    grid = GridSearchCV(
        estimator=spec["pipeline"],
        param_grid=spec["param_grid"],
        scoring="roc_auc",          # optimise ranking quality, robust to imbalance
        cv=config.CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name=name):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Test-set predictions.
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = evaluate.compute_metrics(y_test.to_numpy(), y_pred, y_proba)

        # Log everything for reproducibility / comparison in the MLflow UI.
        mlflow.log_param("model", name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_best_roc_auc", float(grid.best_score_))
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        mlflow.sklearn.log_model(best_model, name="model")

    print(f"[{name}] cv_roc_auc={grid.best_score_:.4f} | test {evaluate.format_metrics(metrics)}")
    return best_model, metrics, grid.best_params_


def main() -> None:
    """Full training entry point: train all candidates, select and persist best."""
    # GridSearch with class_weight can emit convergence noise; keep output clean.
    warnings.filterwarnings("ignore", category=UserWarning)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config.MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # Create the experiment with an explicit artifact location the first time;
    # on subsequent runs it already exists and we just select it.
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            config.EXPERIMENT_NAME,
            artifact_location=config.MLFLOW_ARTIFACTS_DIR.as_uri(),
        )
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    print("Loading data...")
    X_train, X_test, y_train, y_test = data.get_train_test_split()
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, spec in build_candidates().items():
        print(f"\n=== Tuning {name} ===")
        model, metrics, params = tune_and_evaluate(
            name, spec, X_train, y_train, X_test, y_test
        )
        results[name] = {"metrics": metrics, "best_params": params}
        fitted[name] = model

    # Select the best model by test ROC-AUC.
    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} "
          f"(ROC-AUC={results[best_name]['metrics']['roc_auc']:.4f})")

    # Persist the winning pipeline for the API.
    joblib.dump(best_model, config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")

    y_true = y_test.to_numpy()

    # --- Diagnostic figures for the report ------------------------------------
    # 1. Confusion matrix for the selected model.
    y_pred = best_model.predict(X_test)
    evaluate.save_confusion_matrix(
        y_true, y_pred,
        config.ARTIFACTS_DIR / "confusion_matrix.png",
        title=f"Confusion matrix — {best_name}",
    )

    # 2. ROC comparison across ALL models on one axis (more informative than a
    #    single, here near-perfect, curve).
    curves, aucs = {}, {}
    for name, model in fitted.items():
        proba = model.predict_proba(X_test)[:, 1]
        curves[name] = evaluate.roc_points(y_true, proba)
        aucs[name] = results[name]["metrics"]["roc_auc"]
    evaluate.save_roc_comparison(
        curves, aucs, config.ARTIFACTS_DIR / "roc_comparison.png"
    )

    # 3. Permutation importance for the selected model (model-agnostic, on the
    #    original feature columns) — answers "which attributes matter?".
    print("\nComputing permutation importance (this can take a moment)...")
    importance_df = evaluate.compute_permutation_importance(
        best_model, X_test, y_test, random_state=config.RANDOM_STATE
    )
    evaluate.save_feature_importance(
        importance_df,
        config.ARTIFACTS_DIR / "feature_importance.png",
        title=f"Permutation importance — {best_name}",
    )
    print(importance_df.to_string(index=False))

    # --- Machine-readable summary for the report/README -----------------------
    summary = {
        "best_model": best_name,
        "feature_columns": config.FEATURE_COLUMNS,
        "results": results,
        "permutation_importance": importance_df.to_dict(orient="records"),
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
