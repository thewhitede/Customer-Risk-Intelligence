
"""
IBM Telco Churn Error Analysis
==============================

Trains the tuned Gradient Boosting candidate and examines:
- False positives: customers incorrectly predicted to churn.
- False negatives: churners the model misses.
- Overall test metrics at two thresholds.
- Performance across important customer segments.

The established project feature engineering and preprocessing are reused.
All outputs are saved separately from existing research artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import load_raw_data, prepare_model_frame
from src.modeling import build_pipeline_for_frame


RANDOM_STATE_TRAIN_TEST = 2026
RANDOM_STATE_TRAIN_VALIDATION = 2027
TEST_SIZE = 0.20
VALIDATION_FRACTION = 0.25

SEGMENT_COLUMNS = [
    "Contract",
    "TenureGroup",
    "MonthlyChargeGroup",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
]


def make_model(preprocessor):
    """Build the tuned Gradient Boosting model."""

    model = GradientBoostingClassifier(
        subsample=0.85,
        n_estimators=250,
        min_samples_leaf=5,
        max_depth=3,
        learning_rate=0.03,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )


def choose_f1_threshold(y_true, probabilities):
    """Choose a threshold using validation data only."""

    thresholds = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        results.append(
            (float(threshold), float(score))
        )

    best_f1 = max(score for _, score in results)

    tied = [
        (threshold, score)
        for threshold, score in results
        if np.isclose(score, best_f1)
    ]

    # Prefer the threshold closest to 0.50 if tied.
    return min(
        tied,
        key=lambda item: abs(item[0] - 0.50),
    )[0]


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate performance for a given classification threshold."""

    y_true = np.asarray(y_true, dtype=int)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def segment_report(
    predictions_df,
    segment_columns,
    threshold_columns,
):
    """Compare performance across customer segments."""

    results = []

    for segment in segment_columns:
        if segment not in predictions_df.columns:
            continue

        for segment_value, group in predictions_df.groupby(
            segment,
            dropna=False,
            observed=True,
        ):
            y_true = group["actual_churn"].to_numpy(
                dtype=int
            )

            probabilities = group[
                "churn_probability"
            ].to_numpy(dtype=float)

            for threshold_name, threshold in threshold_columns:
                predictions = (
                    probabilities >= threshold
                ).astype(int)

                results.append({
                    "segment": segment,
                    "segment_value": (
                        str(segment_value)
                        if pd.notna(segment_value)
                        else "Missing"
                    ),
                    "threshold_name": threshold_name,
                    "threshold": float(threshold),
                    "customers": int(len(group)),
                    "actual_churn_rate": float(
                        np.mean(y_true)
                    ),
                    "predicted_churn_rate": float(
                        np.mean(predictions)
                    ),
                    "accuracy": float(
                        accuracy_score(y_true, predictions)
                    ),
                    "precision": float(
                        precision_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    "f1": float(
                        f1_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        )
                    ),
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze IBM Telco churn model errors."
    )

    parser.add_argument(
        "--input",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the original IBM Telco CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/ibm_error_analysis",
        help="Directory for diagnostic reports.",
    )

    args = parser.parse_args()

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print("=" * 68)
    print("IBM TELCO CHURN ERROR ANALYSIS")
    print("=" * 68)

    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    # Preserve original row indices to link predictions to customer
    # attributes in the raw dataset.
    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE_TRAIN_TEST,
            stratify=y,
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_STATE_TRAIN_VALIDATION,
            stratify=y_train_validation,
        )
    )

    print(f"\nDataset rows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows: {len(X_test):,}")

    reference_pipeline, _, _ = build_pipeline_for_frame(
        X_train
    )

    if "preprocessor" not in reference_pipeline.named_steps:
        raise ValueError(
            "Expected a 'preprocessor' step in the existing pipeline."
        )

    pipeline = make_model(
        reference_pipeline.named_steps["preprocessor"]
    )

    print("\nTraining Gradient Boosting...")
    pipeline.fit(X_train, y_train)

    validation_probabilities = (
        pipeline.predict_proba(X_validation)[:, 1]
    )

    selected_threshold = choose_f1_threshold(
        y_validation,
        validation_probabilities,
    )

    print(
        "Threshold selected using validation data: "
        f"{selected_threshold:.2f}"
    )

    # The test set is not used to choose the threshold.
    test_probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    default_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        0.50,
    )

    selected_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    # Construct an aligned diagnostic table.
    predictions_df = pd.DataFrame(
        index=X_test.index
    )

    if "customerID" in raw_data.columns:
        predictions_df["customerID"] = raw_data.loc[
            X_test.index,
            "customerID",
        ]

    predictions_df["actual_churn"] = y_test.astype(int)
    predictions_df["churn_probability"] = test_probabilities

    predictions_df["prediction_at_0_50"] = (
        test_probabilities >= 0.50
    ).astype(int)

    predictions_df["prediction_at_selected_threshold"] = (
        test_probabilities >= selected_threshold
    ).astype(int)

    for column in SEGMENT_COLUMNS:
        if column in X_test.columns:
            predictions_df[column] = X_test[column]

    predictions_df["outcome_at_selected_threshold"] = np.select(
        [
            (
                predictions_df["actual_churn"].eq(1)
                & predictions_df[
                    "prediction_at_selected_threshold"
                ].eq(1)
            ),
            (
                predictions_df["actual_churn"].eq(0)
                & predictions_df[
                    "prediction_at_selected_threshold"
                ].eq(1)
            ),
            (
                predictions_df["actual_churn"].eq(1)
                & predictions_df[
                    "prediction_at_selected_threshold"
                ].eq(0)
            ),
        ],
        [
            "True Positive",
            "False Positive",
            "False Negative",
        ],
        default="True Negative",
    )

    segment_results = segment_report(
        predictions_df,
        SEGMENT_COLUMNS,
        [
            ("threshold_0.50", 0.50),
            (
                "validation_selected_threshold",
                selected_threshold,
            ),
        ],
    )

    predictions_path = output_dir / "test_predictions.csv"
    segment_path = output_dir / "segment_performance.csv"
    false_positive_path = output_dir / "false_positives.csv"
    false_negative_path = output_dir / "false_negatives.csv"
    metrics_path = output_dir / "error_analysis_metrics.json"

    predictions_df.to_csv(predictions_path, index=False)
    segment_results.to_csv(segment_path, index=False)

    predictions_df.loc[
        predictions_df[
            "outcome_at_selected_threshold"
        ].eq("False Positive")
    ].to_csv(false_positive_path, index=False)

    predictions_df.loc[
        predictions_df[
            "outcome_at_selected_threshold"
        ].eq("False Negative")
    ].to_csv(false_negative_path, index=False)

    report = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "model": "GradientBoostingClassifier",
        "selected_threshold": float(selected_threshold),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "test_metrics_at_0_50": default_metrics,
        "test_metrics_at_selected_threshold": selected_metrics,
        "false_positive_count_at_selected_threshold": int(
            predictions_df[
                "outcome_at_selected_threshold"
            ].eq("False Positive").sum()
        ),
        "false_negative_count_at_selected_threshold": int(
            predictions_df[
                "outcome_at_selected_threshold"
            ].eq("False Negative").sum()
        ),
        "notes": [
            "Threshold was selected using validation data.",
            "The test set was not used for threshold selection.",
            (
                "This analysis is exploratory because the dataset "
                "has already informed earlier model comparisons."
            ),
            (
                "Segment-level results describe error patterns. They "
                "do not establish the causes of churn."
            ),
        ],
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 68)
    print("OVERALL TEST METRICS")
    print("=" * 68)

    print("\nThreshold 0.50:")
    print(json.dumps(default_metrics, indent=2))

    print("\nValidation-selected threshold:")
    print(json.dumps(selected_metrics, indent=2))

    print("\nSegment performance (selected threshold):")
    if not segment_results.empty:
        print(
            segment_results.loc[
                segment_results["threshold_name"].eq(
                    "validation_selected_threshold"
                )
            ]
            .round(3)
            .to_string(index=False)
        )

    print("\nSaved reports:")
    print(f"  Predictions:       {predictions_path}")
    print(f"  Segment metrics:   {segment_path}")
    print(f"  False positives:   {false_positive_path}")
    print(f"  False negatives:   {false_negative_path}")
    print(f"  Overall metrics:   {metrics_path}")


if __name__ == "__main__":
    main()
