
"""
Hyperparameter tuning for the original IBM Telco churn project.

Models:
- Logistic Regression
- Gradient Boosting
- Random Forest

Uses:
- Existing project feature engineering
- Existing project preprocessing
- Stratified training/test split
- Stratified cross-validation for hyperparameter selection
- Average precision as the search metric

Outputs:
- results/ibm_tuning/tuning_results.csv
- results/ibm_tuning/tuning_results.json

This script does not overwrite the existing research results or model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.features import (
    load_raw_data,
    prepare_model_frame,
)
from src.modeling import build_pipeline_for_frame


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 4
DEFAULT_SEARCH_ITERATIONS = 8
DEFAULT_OUTPUT_DIR = "results/ibm_tuning"


def make_pipeline(preprocessor, estimator):
    """Combine a clone of the project's existing preprocessor with a model."""

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )


def get_model_searches(X_train, iterations):
    """Prepare model pipelines and their search spaces."""

    # Reuse the project's existing preprocessing and Random Forest setup.
    existing_rf_pipeline, _, _ = build_pipeline_for_frame(
        X_train
    )

    if "preprocessor" not in existing_rf_pipeline.named_steps:
        raise ValueError(
            "Expected the existing pipeline to have a "
            "'preprocessor' step."
        )

    preprocessor = existing_rf_pipeline.named_steps[
        "preprocessor"
    ]

    logistic_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
    )

    gradient_pipeline = make_pipeline(
        preprocessor,
        GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    )

    # Reuse the complete existing Random Forest pipeline.
    rf_pipeline = clone(existing_rf_pipeline)

    searches = {
        "logistic_regression": {
            "pipeline": logistic_pipeline,
            "parameters": {
                "model__C": [
                    0.01,
                    0.03,
                    0.1,
                    0.3,
                    1.0,
                    3.0,
                    10.0,
                    30.0,
                ],
                "model__class_weight": [
                    None,
                    "balanced",
                ],
            },
        },

        "gradient_boosting": {
            "pipeline": gradient_pipeline,
            "parameters": {
                "model__n_estimators": [
                    100,
                    150,
                    200,
                    250,
                    300,
                ],
                "model__learning_rate": [
                    0.03,
                    0.05,
                    0.07,
                    0.1,
                ],
                "model__max_depth": [
                    1,
                    2,
                    3,
                ],
                "model__min_samples_leaf": [
                    2,
                    5,
                    10,
                    15,
                ],
                "model__subsample": [
                    0.7,
                    0.85,
                    1.0,
                ],
            },
        },

        "random_forest": {
            "pipeline": rf_pipeline,
            "parameters": {
                "model__n_estimators": [
                    200,
                    300,
                    500,
                ],
                "model__max_depth": [
                    None,
                    8,
                    12,
                    16,
                    20,
                ],
                "model__min_samples_leaf": [
                    1,
                    2,
                    4,
                    8,
                ],
                "model__max_features": [
                    "sqrt",
                    0.7,
                    1.0,
                ],
                "model__class_weight": [
                    None,
                    "balanced",
                    "balanced_subsample",
                ],
            },
        },
    }

    return searches, min(iterations, None or iterations)


def evaluate_holdout(model, X_test, y_test):
    """Calculate holdout metrics using the default 0.50 threshold."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "classification_threshold": 0.50,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_test,
                probabilities,
            )
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune candidate models on the IBM Telco dataset."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the IBM Telco CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the experiment reports.",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_SEARCH_ITERATIONS,
        help="Random parameter combinations per model.",
    )

    args = parser.parse_args()

    if args.n_iter < 1:
        raise ValueError("--n-iter must be at least 1.")

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print("=" * 68)
    print("IBM TELCO CHURN HYPERPARAMETER SEARCH")
    print("=" * 68)

    print(f"\nDataset: {data_path}")

    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    if X.empty:
        raise ValueError("The feature dataset is empty.")

    if y.isna().any():
        raise ValueError(
            "The target contains missing values."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Expected a binary churn target."
        )

    # Reserve the holdout before hyperparameter tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Total rows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Holdout rows: {len(X_test):,}")
    print(f"Training churn rate: {y_train.mean():.2%}")
    print(f"Holdout churn rate: {y_test.mean():.2%}")
    print(
        "\nSearch metric: average precision "
        "(area under the precision-recall curve)"
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    searches, iterations = get_model_searches(
        X_train,
        args.n_iter,
    )

    all_results = []

    for model_name, configuration in searches.items():
        print("\n" + "-" * 68)
        print(f"Searching: {model_name}")
        print("-" * 68)

        search = RandomizedSearchCV(
            estimator=configuration["pipeline"],
            param_distributions=configuration["parameters"],
            n_iter=iterations,
            scoring="average_precision",
            n_jobs=1,
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            verbose=1,
            return_train_score=False,
            error_score="raise",
        )

        # Only the training subset is used during tuning.
        search.fit(
            X_train,
            y_train,
        )

        # Best estimator is refitted on the full training subset.
        best_model = search.best_estimator_

        # Evaluate once on the held-out subset.
        holdout_metrics = evaluate_holdout(
            best_model,
            X_test,
            y_test,
        )

        result = {
            "model": model_name,
            "best_cv_average_precision": float(
                search.best_score_
            ),
            "best_parameters": search.best_params_,
            "holdout_metrics": holdout_metrics,
        }

        all_results.append(result)

        print(
            "\nBest CV average precision: "
            f"{search.best_score_:.4f}"
        )

        print("Best parameters:")
        print(
            json.dumps(
                search.best_params_,
                indent=2,
                default=str,
            )
        )

        print("\nHoldout results:")
        print(
            json.dumps(
                holdout_metrics,
                indent=2,
            )
        )

    # Rank candidates by the training-only cross-validation search score.
    # Do not pick a winner by looking at holdout metrics.
    all_results.sort(
        key=lambda item: item["best_cv_average_precision"],
        reverse=True,
    )

    selected = all_results[0]

    payload = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "train_test_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "hyperparameter_search": {
            "cv_folds": CV_FOLDS,
            "iterations_per_model": iterations,
            "selection_metric": "average_precision",
        },
        "selected_candidate_by_cv": selected["model"],
        "results": all_results,
        "notes": [
            (
                "The existing feature engineering and preprocessing "
                "were reused."
            ),
            (
                "The holdout was not used to tune hyperparameters."
            ),
            (
                "The default 0.50 classification threshold is used "
                "for the holdout report."
            ),
            (
                "The previous full-dataset cross-validation benchmark "
                "has already informed candidate selection, so this "
                "follow-up remains exploratory rather than a pristine "
                "final research evaluation."
            ),
            (
                "Accuracy must be considered alongside churn precision, "
                "recall, F1, average precision, ROC-AUC and Brier score."
            ),
        ],
    }

    json_path = output_dir / "tuning_results.json"
    csv_path = output_dir / "tuning_results.csv"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    table_rows = []

    for result in all_results:
        metrics = result["holdout_metrics"]

        table_rows.append({
            "model": result["model"],
            "best_cv_average_precision": (
                result["best_cv_average_precision"]
            ),
            "holdout_accuracy": metrics["accuracy"],
            "holdout_balanced_accuracy": (
                metrics["balanced_accuracy"]
            ),
            "holdout_precision": metrics["precision"],
            "holdout_recall": metrics["recall"],
            "holdout_f1": metrics["f1"],
            "holdout_roc_auc": metrics["roc_auc"],
            "holdout_average_precision": (
                metrics["average_precision"]
            ),
            "holdout_brier_score": metrics["brier_score"],
            "best_parameters": json.dumps(
                result["best_parameters"],
                default=str,
            ),
        })

    summary = pd.DataFrame(table_rows)
    summary.to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 100)
    print("TUNING SUMMARY")
    print("=" * 100)

    print(
        summary[
            [
                "model",
                "best_cv_average_precision",
                "holdout_accuracy",
                "holdout_balanced_accuracy",
                "holdout_precision",
                "holdout_recall",
                "holdout_f1",
                "holdout_roc_auc",
                "holdout_average_precision",
                "holdout_brier_score",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nLeading candidate by training cross-validation: "
        f"{selected['model']}"
    )

    print("\nSaved reports:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print(
        "\nNo existing production model or research results "
        "were overwritten."
    )


if __name__ == "__main__":
    main()
