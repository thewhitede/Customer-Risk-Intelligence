
"""
IBM Telco Churn Threshold Analysis
==================================

Compares tuned Logistic Regression, Gradient Boosting, and Random Forest.

Workflow:
1. Load the original IBM Telco dataset.
2. Reuse the existing project feature engineering and preprocessing.
3. Split data into training, validation, and test subsets.
4. Fit candidate models on training data only.
5. Select an F1-maximizing threshold on validation data.
6. Compare threshold 0.50 and the selected threshold on test data.
7. Save results separately from existing research artifacts.

This experiment does not overwrite production models or notebooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import (
    load_raw_data,
    prepare_model_frame,
)
from src.modeling import build_pipeline_for_frame


RANDOM_STATE = 42
OUTPUT_DIR_DEFAULT = "results/ibm_threshold_analysis"


def make_pipeline(preprocessor, estimator):
    """Combine the existing project preprocessor with a classifier."""

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )


def make_models(preprocessor):
    """Create models using the best parameters from the last search."""

    return {
        "logistic_regression": make_pipeline(
            preprocessor,
            LogisticRegression(
                C=1.0,
                class_weight=None,
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),

        "gradient_boosting": make_pipeline(
            preprocessor,
            GradientBoostingClassifier(
                subsample=0.85,
                n_estimators=250,
                min_samples_leaf=5,
                max_depth=3,
                learning_rate=0.03,
                random_state=RANDOM_STATE,
            ),
        ),

        "random_forest": make_pipeline(
            preprocessor,
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=8,
                max_features=0.7,
                max_depth=8,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def choose_threshold(y_valid, valid_probabilities):
    """
    Choose the threshold with the highest F1 on validation data only.

    If multiple thresholds produce the same maximum F1, prefer the
    threshold closest to 0.50.
    """

    candidates = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    scores = []

    for threshold in candidates:
        predictions = (
            valid_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_valid,
            predictions,
            zero_division=0,
        )

        scores.append(
            {
                "threshold": float(threshold),
                "f1": float(score),
            }
        )

    maximum_f1 = max(item["f1"] for item in scores)

    tied_candidates = [
        item for item in scores
        if np.isclose(item["f1"], maximum_f1)
    ]

    selected = min(
        tied_candidates,
        key=lambda item: abs(item["threshold"] - 0.50),
    )

    return selected["threshold"], scores


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification and probability-ranking metrics."""

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
            "Compare churn decision thresholds on the IBM Telco dataset."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the original IBM Telco dataset.",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for threshold analysis reports.",
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

    print("=" * 68)
    print("IBM TELCO CHURN THRESHOLD ANALYSIS")
    print("=" * 68)

    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    if y.isna().any() or y.nunique() != 2:
        raise ValueError(
            "The target must contain two valid classes and no missing labels."
        )

    # First reserve 20% for test evaluation.
    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=2026,
        )
    )

    # Of the remaining 80%, reserve 25% for validation.
    # Overall proportions are approximately 60% train, 20% validation,
    # and 20% test.
    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=0.25,
            stratify=y_train_validation,
            random_state=2027,
        )
    )

    print(f"\nDataset rows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Overall churn rate: {y.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    # Obtain the established preprocessing pipeline from the project.
    reference_pipeline, _, _ = build_pipeline_for_frame(
        X_train
    )

    if "preprocessor" not in reference_pipeline.named_steps:
        raise ValueError(
            "The existing pipeline has no 'preprocessor' step."
        )

    preprocessor = reference_pipeline.named_steps[
        "preprocessor"
    ]

    models = make_models(preprocessor)
    all_results = []

    for model_name, model in models.items():
        print("\n" + "-" * 68)
        print(f"Training: {model_name}")
        print("-" * 68)

        # Fit only on training rows.
        model.fit(
            X_train,
            y_train,
        )

        # Choose threshold using validation rows only.
        validation_probabilities = (
            model.predict_proba(X_validation)[:, 1]
        )

        selected_threshold, threshold_scores = choose_threshold(
            y_validation,
            validation_probabilities,
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            selected_threshold,
        )

        # Evaluate the model on test rows. The test set is not used
        # to select the threshold.
        test_probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        default_metrics = calculate_metrics(
            y_test,
            test_probabilities,
            threshold=0.50,
        )

        tuned_metrics = calculate_metrics(
            y_test,
            test_probabilities,
            threshold=selected_threshold,
        )

        result = {
            "model": model_name,
            "selected_threshold": selected_threshold,
            "validation_metrics_at_selected_threshold": (
                validation_metrics
            ),
            "test_metrics_at_threshold_0_50": (
                default_metrics
            ),
            "test_metrics_at_validation_selected_threshold": (
                tuned_metrics
            ),
            "validation_threshold_search": threshold_scores,
        }

        all_results.append(result)

        print(
            f"Selected validation threshold: "
            f"{selected_threshold:.2f}"
        )

        print("\nTest performance at threshold 0.50:")
        print(
            json.dumps(
                default_metrics,
                indent=2,
            )
        )

        print("\nTest performance at selected threshold:")
        print(
            json.dumps(
                tuned_metrics,
                indent=2,
            )
        )

    # Accuracy of predicting the majority class on test data.
    all_negative_accuracy = float(
        (y_test == 0).mean()
    )

    summary_rows = []

    for result in all_results:
        default = result["test_metrics_at_threshold_0_50"]
        tuned = result[
            "test_metrics_at_validation_selected_threshold"
        ]

        summary_rows.append(
            {
                "model": result["model"],
                "selected_threshold": result["selected_threshold"],
                "test_accuracy_at_0_50": default["accuracy"],
                "test_f1_at_0_50": default["f1"],
                "test_recall_at_0_50": default["recall"],
                "test_precision_at_0_50": default["precision"],
                "test_accuracy_at_selected_threshold": tuned["accuracy"],
                "test_f1_at_selected_threshold": tuned["f1"],
                "test_recall_at_selected_threshold": tuned["recall"],
                "test_precision_at_selected_threshold": tuned["precision"],
                "test_average_precision": tuned["average_precision"],
                "test_roc_auc": tuned["roc_auc"],
                "test_brier_score": tuned["brier_score"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    csv_path = output_dir / "threshold_results.csv"
    json_path = output_dir / "threshold_results.json"

    summary.to_csv(
        csv_path,
        index=False,
    )

    report = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "split": {
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "training_fraction_approx": 0.60,
            "validation_fraction_approx": 0.20,
            "test_fraction_approx": 0.20,
        },
        "all_negative_test_accuracy_baseline": (
            all_negative_accuracy
        ),
        "threshold_selection_metric": (
            "F1 on validation data"
        ),
        "models": all_results,
        "notes": [
            (
                "Thresholds are selected using validation labels, "
                "not test labels."
            ),
            (
                "Test metrics at threshold 0.50 and at the selected "
                "threshold are reported separately."
            ),
            (
                "This dataset has been used in earlier experiments. "
                "Treat this comparison as exploratory, not as a "
                "pristine final research evaluation."
            ),
            (
                "A higher threshold can increase accuracy while reducing "
                "the number of churners detected. Accuracy alone is "
                "not sufficient for model selection."
            ),
        ],
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("THRESHOLD COMPARISON SUMMARY")
    print("=" * 110)

    print(
        summary.round(4).to_string(index=False)
    )

    print(
        "\nAll-negative test accuracy baseline: "
        f"{all_negative_accuracy:.2%}"
    )

    print("\nSaved reports:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print(
        "\nExisting research artifacts and production models "
        "were not overwritten."
    )


if __name__ == "__main__":
    main()
