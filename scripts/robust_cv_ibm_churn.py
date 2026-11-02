
"""
Repeated Cross-Validation for IBM Telco Churn
=============================================

Models:
- Tuned Gradient Boosting
- Sigmoid-calibrated Gradient Boosting
- Tuned Random Forest
- Tuned Logistic Regression

Uses the original project's feature engineering and preprocessing.
Reports average precision, ROC-AUC, accuracy, recall, F1,
Brier score, and log loss.

This is an exploratory robustness comparison. It does not overwrite
the production model or established research artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.features import load_raw_data, prepare_model_frame
from src.modeling import build_pipeline_for_frame


RANDOM_STATE = 314
N_SPLITS = 5
N_REPEATS = 3
OUTPUT_DIR = Path("results/ibm_robust_cv")


def attach_model(preprocessor, classifier):
    """Attach a fresh copy of the project's preprocessor to a classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", classifier),
        ]
    )


def build_models(preprocessor):
    """Create fixed candidate models for repeated cross-validation."""

    gradient_boosting = attach_model(
        preprocessor,
        GradientBoostingClassifier(
            subsample=0.85,
            n_estimators=250,
            min_samples_leaf=5,
            max_depth=3,
            learning_rate=0.03,
            random_state=42,
        ),
    )

    calibrated_gradient_boosting = CalibratedClassifierCV(
        estimator=clone(gradient_boosting),
        method="sigmoid",
        cv=3,
    )

    random_forest = attach_model(
        preprocessor,
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            max_features=0.7,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    )

    logistic_regression = attach_model(
        preprocessor,
        LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=3000,
            random_state=42,
        ),
    )

    return {
        "gradient_boosting": gradient_boosting,
        "gradient_boosting_sigmoid_calibrated": (
            calibrated_gradient_boosting
        ),
        "random_forest": random_forest,
        "logistic_regression": logistic_regression,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/raw/telco_customer_churn.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print("=" * 72)
    print("REPEATED STRATIFIED CROSS-VALIDATION")
    print("=" * 72)

    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    if y.isna().any() or y.nunique() != 2:
        raise ValueError(
            "The target must contain two classes and no missing labels."
        )

    print(f"\nRows: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Folds: {N_SPLITS}")
    print(f"Repeats: {N_REPEATS}")
    print(f"Total evaluation folds per model: {N_SPLITS * N_REPEATS}")

    # Get the established preprocessing from the original project.
    reference_pipeline, _, _ = build_pipeline_for_frame(X)

    if "preprocessor" not in reference_pipeline.named_steps:
        raise ValueError(
            "Expected a 'preprocessor' step in the existing pipeline."
        )

    preprocessor = reference_pipeline.named_steps["preprocessor"]
    models = build_models(preprocessor)

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "neg_brier_score": "neg_brier_score",
        "neg_log_loss": "neg_log_loss",
    }

    fold_rows = []
    summary_rows = []

    for model_name, model in models.items():
        print("\n" + "-" * 72)
        print(f"Evaluating: {model_name}")
        print("-" * 72)

        scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        fold_count = len(scores["test_accuracy"])

        metric_values = {}

        for score_name in scoring:
            values = scores[f"test_{score_name}"]

            if score_name.startswith("neg_"):
                # Convert sklearn's negative loss score to positive loss.
                actual_name = score_name.removeprefix("neg_")
                values = -values
            else:
                actual_name = score_name

            metric_values[actual_name] = values

        summary = {
            "model": model_name,
            "folds": fold_count,
        }

        for metric_name, values in metric_values.items():
            summary[f"{metric_name}_mean"] = float(
                np.mean(values)
            )
            summary[f"{metric_name}_std"] = float(
                np.std(values, ddof=1)
            )

        summary_rows.append(summary)

        for index in range(fold_count):
            repeat_number = index // N_SPLITS + 1
            fold_number = index % N_SPLITS + 1

            row = {
                "model": model_name,
                "repeat": repeat_number,
                "fold": fold_number,
            }

            for metric_name, values in metric_values.items():
                row[metric_name] = float(values[index])

            fold_rows.append(row)

        print(
            "Mean average precision: "
            f"{summary['average_precision_mean']:.4f} "
            f"+/- {summary['average_precision_std']:.4f}"
        )
        print(
            "Mean ROC-AUC: "
            f"{summary['roc_auc_mean']:.4f} "
            f"+/- {summary['roc_auc_std']:.4f}"
        )
        print(
            "Mean Brier score: "
            f"{summary['brier_score_mean']:.4f} "
            f"+/- {summary['brier_score_std']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)

    # Average precision is the primary ranking metric for this benchmark.
    summary_df = summary_df.sort_values(
        "average_precision_mean",
        ascending=False,
    ).reset_index(drop=True)

    summary_path = OUTPUT_DIR / "repeated_cv_summary.csv"
    folds_path = OUTPUT_DIR / "repeated_cv_fold_metrics.csv"
    json_path = OUTPUT_DIR / "repeated_cv_results.json"

    summary_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    report = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "cross_validation": {
            "method": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_STATE,
        },
        "classification_threshold": 0.50,
        "primary_metric": "average_precision",
        "results": summary_df.to_dict(orient="records"),
        "notes": [
            (
                "The established feature engineering and preprocessing "
                "are reused from the original project."
            ),
            (
                "All preprocessing is fitted within each training fold."
            ),
            (
                "The calibrated model uses internal cross-validation "
                "on each outer training fold."
            ),
            (
                "Fixed threshold 0.50 is used for classification metrics. "
                "No threshold is tuned against an evaluation fold."
            ),
            (
                "Configurations were selected after earlier experiments "
                "on this dataset, so these results remain exploratory."
            ),
            (
                "Fold standard deviations describe variation across "
                "the evaluated folds and are not formal confidence intervals."
            ),
        ],
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    columns = [
        "model",
        "average_precision_mean",
        "average_precision_std",
        "roc_auc_mean",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "brier_score_mean",
        "log_loss_mean",
    ]

    print("\n" + "=" * 110)
    print("REPEATED CROSS-VALIDATION SUMMARY")
    print("=" * 110)
    print(summary_df[columns].round(4).to_string(index=False))

    print("\nSaved files:")
    print(f"  Summary: {summary_path}")
    print(f"  Fold metrics: {folds_path}")
    print(f"  Full JSON: {json_path}")


if __name__ == "__main__":
    main()
