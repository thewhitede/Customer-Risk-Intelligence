
"""
Gradient Boosting Probability Calibration Experiment
===================================================

Compares the tuned Gradient Boosting model:
1. Without probability calibration
2. With sigmoid probability calibration

Workflow:
- Reuses the existing project feature engineering.
- Uses separate train, validation, and test splits.
- Fits and calibrates models using training data only.
- Selects F1 thresholds on validation data.
- Reports final exploratory test metrics at threshold 0.50
  and at each validation-selected threshold.
- Saves results separately from existing research artifacts.

This script does not replace the production model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.features import load_raw_data, prepare_model_frame
from src.modeling import build_pipeline_for_frame


RANDOM_STATE = 42
OUTPUT_DIR_DEFAULT = "results/ibm_calibration_experiment"


def choose_f1_threshold(y_true, probabilities):
    """Select a threshold using validation data only."""

    candidates = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    scores = []

    for threshold in candidates:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        scores.append(
            (float(threshold), float(score))
        )

    best_f1 = max(score for _, score in scores)

    tied = [
        (threshold, score)
        for threshold, score in scores
        if np.isclose(score, best_f1)
    ]

    # Prefer a threshold closer to 0.50 when scores are tied.
    return min(
        tied,
        key=lambda item: abs(item[0] - 0.50),
    )[0]


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification and probability metrics."""

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
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
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
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def make_calibration_bins(y_true, probabilities, model_name):
    """
    Summarize predicted probabilities against observed churn rates.

    These bins are diagnostic, not a substitute for a dedicated
    calibration evaluation on a genuinely untouched dataset.
    """

    frame = pd.DataFrame({
        "actual_churn": np.asarray(y_true, dtype=int),
        "predicted_probability": np.asarray(
            probabilities,
            dtype=float,
        ),
    })

    frame["probability_bin"] = pd.qcut(
        frame["predicted_probability"],
        q=10,
        duplicates="drop",
    )

    summary = (
        frame.groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            customers=("actual_churn", "size"),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_churn_rate=(
                "actual_churn",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["model"] = model_name
    summary["probability_bin"] = (
        summary["probability_bin"].astype(str)
    )

    return summary[
        [
            "model",
            "probability_bin",
            "customers",
            "mean_predicted_probability",
            "observed_churn_rate",
        ]
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare calibrated and uncalibrated "
            "Gradient Boosting probabilities."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the original IBM Telco CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory for the separate experiment outputs.",
    )

    args = parser.parse_args()

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
    print("GRADIENT BOOSTING CALIBRATION EXPERIMENT")
    print("=" * 68)

    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    if y.isna().any() or y.nunique() != 2:
        raise ValueError(
            "Expected a binary target with no missing labels."
        )

    # Reserve the test split before fitting or threshold selection.
    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=2026,
        )
    )

    # Split the remaining rows into train and validation subsets.
    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=0.25,
            stratify=y_train_validation,
            random_state=2027,
        )
    )

    print(f"\nTotal rows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Training churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    # Reuse the project's existing preprocessing pipeline.
    reference_pipeline, _, _ = build_pipeline_for_frame(
        X_train
    )

    if "preprocessor" not in reference_pipeline.named_steps:
        raise ValueError(
            "Expected 'preprocessor' in the existing pipeline."
        )

    preprocessor = reference_pipeline.named_steps[
        "preprocessor"
    ]

    gradient_boosting = GradientBoostingClassifier(
        subsample=0.85,
        n_estimators=250,
        min_samples_leaf=5,
        max_depth=3,
        learning_rate=0.03,
        random_state=RANDOM_STATE,
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", gradient_boosting),
        ]
    )

    # Model A: uncalibrated.
    print("\nFitting uncalibrated Gradient Boosting...")
    uncalibrated_model = clone(base_pipeline)
    uncalibrated_model.fit(X_train, y_train)

    # Model B: sigmoid calibration, fitted using training data only.
    print("Fitting sigmoid-calibrated Gradient Boosting...")

    calibration_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=clone(base_pipeline),
        method="sigmoid",
        cv=calibration_cv,
    )

    calibrated_model.fit(X_train, y_train)

    models = {
        "gradient_boosting_uncalibrated": (
            uncalibrated_model
        ),
        "gradient_boosting_sigmoid_calibrated": (
            calibrated_model
        ),
    }

    all_results = []
    all_predictions = []
    all_calibration_bins = []

    for model_name, model in models.items():
        print("\n" + "-" * 68)
        print(f"Evaluating: {model_name}")
        print("-" * 68)

        # The validation set selects the operating threshold.
        validation_probabilities = (
            model.predict_proba(X_validation)[:, 1]
        )

        threshold = choose_f1_threshold(
            y_validation,
            validation_probabilities,
        )

        # The test set is used for reporting, not threshold selection.
        test_probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        metrics_050 = calculate_metrics(
            y_test,
            test_probabilities,
            0.50,
        )

        metrics_selected = calculate_metrics(
            y_test,
            test_probabilities,
            threshold,
        )

        result = {
            "model": model_name,
            "validation_selected_threshold": float(
                threshold
            ),
            "test_metrics_at_threshold_0_50": metrics_050,
            "test_metrics_at_validation_selected_threshold": (
                metrics_selected
            ),
        }

        all_results.append(result)

        print(
            "Selected threshold from validation: "
            f"{threshold:.2f}"
        )

        print("\nTest metrics at threshold 0.50:")
        print(json.dumps(metrics_050, indent=2))

        print("\nTest metrics at selected threshold:")
        print(json.dumps(metrics_selected, indent=2))

        # Store test probabilities for inspection.
        model_predictions = pd.DataFrame({
            "actual_churn": y_test.to_numpy(dtype=int),
            f"{model_name}_probability": test_probabilities,
            f"{model_name}_prediction_at_0_50": (
                test_probabilities >= 0.50
            ).astype(int),
            f"{model_name}_prediction_at_selected_threshold": (
                test_probabilities >= threshold
            ).astype(int),
        })

        all_predictions.append(model_predictions)

        all_calibration_bins.append(
            make_calibration_bins(
                y_test,
                test_probabilities,
                model_name,
            )
        )

    prediction_table = all_predictions[0]

    for next_table in all_predictions[1:]:
        prediction_table = prediction_table.join(
            next_table.drop(columns=["actual_churn"])
        )

    calibration_table = pd.concat(
        all_calibration_bins,
        ignore_index=True,
    )

    # Summarize key results in a compact table.
    summary_rows = []

    for result in all_results:
        for threshold_name, metrics in [
            (
                "default_0_50",
                result["test_metrics_at_threshold_0_50"],
            ),
            (
                "validation_selected",
                result[
                    "test_metrics_at_validation_selected_threshold"
                ],
            ),
        ]:
            summary_rows.append({
                "model": result["model"],
                "threshold_choice": threshold_name,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
            })

    summary = pd.DataFrame(summary_rows)

    report = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "calibration_method": "sigmoid",
        "threshold_selection_metric": (
            "F1 on validation data"
        ),
        "models": all_results,
        "notes": [
            (
                "Preprocessing and feature engineering are reused "
                "from the original project."
            ),
            (
                "Calibration is fitted using training data only."
            ),
            (
                "The classification threshold is selected using "
                "validation data, not test data."
            ),
            (
                "The dataset has informed previous experiments, "
                "so these results remain exploratory."
            ),
            (
                "A lower Brier score and log loss indicate better "
                "probability quality; ROC-AUC and average precision "
                "measure ranking ability."
            ),
        ],
    }

    metrics_path = output_dir / "calibration_metrics.json"
    summary_path = output_dir / "calibration_summary.csv"
    predictions_path = output_dir / "test_probabilities.csv"
    bins_path = output_dir / "calibration_bins.csv"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    summary.to_csv(summary_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    calibration_table.to_csv(bins_path, index=False)

    print("\n" + "=" * 110)
    print("CALIBRATION COMPARISON SUMMARY")
    print("=" * 110)
    print(summary.round(4).to_string(index=False))

    print("\nSaved reports:")
    print(f"  Metrics:      {metrics_path}")
    print(f"  Summary:      {summary_path}")
    print(f"  Test scores:  {predictions_path}")
    print(f"  Calibration:  {bins_path}")

    print(
        "\nNo existing production model, research notebook, "
        "or frozen result was overwritten."
    )


if __name__ == "__main__":
    main()
