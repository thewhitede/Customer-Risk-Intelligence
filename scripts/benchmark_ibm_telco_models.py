"""
IBM Telco Churn Model Benchmark
================================

Uses the project's established feature engineering and compares:
1. Majority-class baseline
2. Logistic Regression
3. Existing Random Forest without calibration
4. Existing Random Forest with sigmoid calibration
5. Extra Trees
6. Gradient Boosting
7. Histogram-based Gradient Boosting

Evaluation:
- Stratified 5-fold cross-validation
- Average precision (PR-AUC) is the primary comparison metric
- Additional classification and probability-quality metrics

This is a separate experiment. It does not overwrite the original
research notebooks, production model, or established results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.features import (
    load_raw_data,
    prepare_model_frame,
)
from src.modeling import build_pipeline_for_frame


RANDOM_STATE = 42
NUMBER_OF_FOLDS = 5


def build_benchmark_models(X: pd.DataFrame) -> dict:
    """
    Create candidate models using the project's existing preprocessing.

    The current Random Forest configuration is taken from
    src.modeling.build_pipeline_for_frame().
    """

    # Reuse the exact established preprocessing and Random Forest setup.
    existing_rf_pipeline, _, _ = build_pipeline_for_frame(X)

    preprocessor_template = (
        existing_rf_pipeline.named_steps["preprocessor"]
    )

    def make_pipeline(model) -> Pipeline:
        """Attach the existing preprocessing to another classifier."""
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor_template),
                ),
                ("model", model),
            ]
        )

    models = {
        "majority_baseline": make_pipeline(
            DummyClassifier(strategy="prior")
        ),

        "logistic_regression": make_pipeline(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),

        # Same Random Forest configuration as the current research model.
        "random_forest_uncalibrated": clone(
            existing_rf_pipeline
        ),

        # Evaluate the existing architecture with sigmoid calibration.
        # Cross-validation below keeps the outer evaluation folds separate.
        "random_forest_calibrated": (
            CalibratedClassifierCV(
                estimator=clone(existing_rf_pipeline),
                method="sigmoid",
                cv=5,
            )
        ),

        "extra_trees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "gradient_boosting": make_pipeline(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            )
        ),

        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
    }

    return models


def evaluate_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Evaluate each candidate with the same stratified folds."""

    cv = StratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
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
    }

    results = []

    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")

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

        row = {"model": model_name}

        # Report mean and fold-to-fold standard deviation.
        for metric in scoring:
            values = scores[f"test_{metric}"]

            if metric == "neg_brier_score":
                # sklearn reports negative Brier score so that
                # larger values are better. Convert back to Brier loss.
                row["brier_score_mean"] = float(
                    -values.mean()
                )
                row["brier_score_std"] = float(
                    values.std(ddof=1)
                )
            else:
                row[f"{metric}_mean"] = float(
                    values.mean()
                )
                row[f"{metric}_std"] = float(
                    values.std(ddof=1)
                )

        results.append(row)

        print(
            f"  Average precision: "
            f"{row['average_precision_mean']:.4f} "
            f"+/- {row['average_precision_std']:.4f}"
        )

        print(
            f"  ROC-AUC: "
            f"{row['roc_auc_mean']:.4f} "
            f"+/- {row['roc_auc_std']:.4f}"
        )

        print(
            f"  Accuracy: "
            f"{row['accuracy_mean']:.4f} "
            f"+/- {row['accuracy_std']:.4f}"
        )

    summary = pd.DataFrame(results)

    # Model selection is based on average precision, not accuracy.
    summary = summary.sort_values(
        by="average_precision_mean",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark churn models on the original IBM Telco dataset."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the IBM Telco customer churn CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/ibm_model_benchmark",
        help="Directory for the separate experiment reports.",
    )

    args = parser.parse_args()

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("IBM TELCO CHURN MODEL BENCHMARK")
    print("=" * 65)

    print(f"\nDataset: {data_path}")

    # Use the project's existing data-loading and feature functions.
    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if y.isna().any():
        raise ValueError(
            "The target contains missing values."
        )

    if y.nunique() != 2:
        raise ValueError(
            "The churn target must contain exactly two classes."
        )

    if not X.columns.is_unique:
        raise ValueError(
            "Duplicate feature column names were found."
        )

    print(f"Rows: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Non-churn rate: {(1 - y.mean()):.2%}")

    print(
        "\nUsing the established AnnualRevenue, TenureGroup, "
        "and MonthlyChargeGroup features."
    )

    print(
        f"Cross-validation: {NUMBER_OF_FOLDS}-fold stratified"
    )

    models = build_benchmark_models(X,)

    summary = evaluate_models(
        models=models,
        X=X,
        y=y,
    )

    csv_path = output_dir / "cv_metrics.csv"
    json_path = output_dir / "cv_metrics.json"

    summary.to_csv(
        csv_path,
        index=False,
    )

    results_payload = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": NUMBER_OF_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "selection_metric": "average_precision_mean",
        "feature_columns": list(X.columns),
        "results": summary.to_dict(orient="records"),
        "notes": [
            (
                "This is a separate experimental benchmark; "
                "it does not overwrite established research results."
            ),
            (
                "Models use the same feature engineering and "
                "stratified folds."
            ),
            (
                "The model with the highest mean average precision "
                "is the leading candidate for further evaluation."
            ),
            (
                "These cross-validation results are exploratory. "
                "Model selection can make the winning estimate optimistic."
            ),
            (
                "Accuracy is not sufficient by itself for evaluating "
                "a churn prediction model."
            ),
        ],
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results_payload,
            file,
            indent=2,
        )

    display_columns = [
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
    ]

    print("\n" + "=" * 100)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 100)

    print(
        summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    winner = summary.iloc[0]

    print("\nLeading candidate by mean average precision:")
    print(f"  {winner['model']}")
    print(
        f"  Mean average precision: "
        f"{winner['average_precision_mean']:.4f}"
    )

    print("\nSaved reports:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print(
        "\nNo existing notebooks, production models, "
        "or established metrics were overwritten."
    )


if __name__ == "__main__":
    main()