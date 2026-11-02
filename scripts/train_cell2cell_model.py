
"""
Cell2Cell Churn Prediction Benchmark
====================================

Compares:
1. Logistic Regression
2. Random Forest
3. Extra Trees
4. Histogram-based Gradient Boosting

Training:
    CALIBRAT == 1

Target:
    CHURN

External evaluation:
    CALIBRAT == 0

Important:
- CHURN and CHURNDEP are excluded from model features.
- Model selection uses an internal split of the calibration sample.
- The external validation sample is not used to choose the model.
- Accuracy is reported alongside churn-focused metrics.
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.50


def build_pipeline(model, X):
    """Build a preprocessing and classification pipeline."""

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            )
        )

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns were found."
        )

    # Dense output supports HistGradientBoostingClassifier.
    # This dataset has a moderate number of feature columns.
    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def calculate_metrics(y_true, probabilities, threshold=0.50):
    """Calculate classification and ranking metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
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
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
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
            "Train and benchmark Cell2Cell churn models."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/cell2cell_churn.csv",
        help="Path to the Cell2Cell CSV dataset.",
    )

    parser.add_argument(
        "--output-dir",
        default="models/cell2cell",
        help="Directory for the saved model and metrics.",
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

    print(f"Loading dataset: {data_path}")

    df = pd.read_csv(data_path)

    required_columns = {
        "CALIBRAT",
        "CHURN",
        "CHURNDEP",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"Dataset shape: {df.shape}")

    # Validate the sample indicator.
    df["CALIBRAT"] = pd.to_numeric(
        df["CALIBRAT"],
        errors="raise",
    )

    if df["CALIBRAT"].isna().any():
        raise ValueError(
            "CALIBRAT contains missing values."
        )

    if not set(df["CALIBRAT"].unique()).issubset(
        {0, 1}
    ):
        raise ValueError(
            "CALIBRAT must contain only 0 and 1."
        )

    # This benchmark uses CHURN consistently as the target.
    # The previous audit found CHURN and CHURNDEP identical
    # on the 40,000 calibration records.
    calibration = df.loc[
        df["CALIBRAT"] == 1
    ].copy()

    external = df.loc[
        df["CALIBRAT"] == 0
    ].copy()

    calibration = calibration.dropna(
        subset=["CHURN"]
    )

    external = external.dropna(
        subset=["CHURN"]
    )

    if calibration.empty:
        raise ValueError(
            "No calibration records have a CHURN label."
        )

    if external.empty:
        raise ValueError(
            "No external validation records have a CHURN label."
        )

    # Exclude identifiers, sample indicators, and label columns.
    # The column names are compared case-insensitively.
    excluded_columns = {
        "customer",
        "customer_id",
        "calibrat",
        "churn",
        "churndep",
    }

    feature_columns = [
        col for col in df.columns
        if col.lower() not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns remain after exclusions."
        )

    X_calibration = calibration[
        feature_columns
    ].copy()

    y_calibration = pd.to_numeric(
        calibration["CHURN"],
        errors="raise",
    ).astype(int)

    X_external = external[
        feature_columns
    ].copy()

    y_external = pd.to_numeric(
        external["CHURN"],
        errors="raise",
    ).astype(int)

    valid_labels = {0, 1}

    if not set(y_calibration.unique()).issubset(
        valid_labels
    ):
        raise ValueError(
            "Calibration CHURN labels must be 0 or 1."
        )

    if not set(y_external.unique()).issubset(
        valid_labels
    ):
        raise ValueError(
            "External CHURN labels must be 0 or 1."
        )

    # Drop columns entirely missing in the training sample.
    usable_columns = [
        col for col in X_calibration.columns
        if X_calibration[col].notna().any()
    ]

    dropped_columns = sorted(
        set(X_calibration.columns) -
        set(usable_columns)
    )

    X_calibration = X_calibration[
        usable_columns
    ].copy()

    X_external = X_external[
        usable_columns
    ].copy()

    if not usable_columns:
        raise ValueError(
            "All feature columns are missing in calibration data."
        )

    print(
        f"Calibration records: {len(X_calibration):,}"
    )
    print(
        f"External validation records: {len(X_external):,}"
    )
    print(
        "Calibration churn rate: "
        f"{y_calibration.mean():.2%}"
    )
    print(
        "External validation churn rate: "
        f"{y_external.mean():.2%}"
    )

    if dropped_columns:
        print(
            "Dropped all-missing training columns: "
            f"{dropped_columns}"
        )

    # Hold out an internal development split for model selection.
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_calibration,
        y_calibration,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_calibration,
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
    }

    development_results = {}
    best_name = None
    best_average_precision = -1.0

    print(
        "\nComparing models on the internal "
        "calibration-data development split..."
    )

    for name, estimator in candidates.items():
        print(f"\nTraining: {name}")

        pipeline = build_pipeline(
            estimator,
            X_train,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        development_probabilities = (
            pipeline.predict_proba(X_dev)[:, 1]
        )

        development_average_precision = (
            average_precision_score(
                y_dev,
                development_probabilities,
            )
        )

        development_results[name] = {
            "development_average_precision": float(
                development_average_precision
            )
        }

        print(
            f"Internal development average precision: "
            f"{development_average_precision:.4f}"
        )

        if (
            development_average_precision
            > best_average_precision
        ):
            best_average_precision = float(
                development_average_precision
            )

            best_name = name

    if best_name is None:
        raise RuntimeError(
            "No model was successfully selected."
        )

    print(
        "\nSelected model by internal average precision: "
        f"{best_name}"
    )

    # Refit the chosen algorithm on all labeled calibration records.
    final_pipeline = build_pipeline(
        candidates[best_name],
        X_calibration,
    )

    print(
        "\nRefitting selected model on all calibration records..."
    )

    final_pipeline.fit(
        X_calibration,
        y_calibration,
    )

    # Evaluate once using the provided external sample.
    print(
        "\nEvaluating selected model on external validation..."
    )

    external_probabilities = (
        final_pipeline.predict_proba(
            X_external
        )[:, 1]
    )

    external_metrics = calculate_metrics(
        y_external,
        external_probabilities,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    majority_class_accuracy = max(
        float((y_external == 0).mean()),
        float((y_external == 1).mean()),
    )

    results = {
        "dataset": str(data_path),
        "selected_model": best_name,
        "model_selection_metric": (
            "internal development average precision"
        ),
        "training_rows": int(
            len(X_calibration)
        ),
        "external_validation_rows": int(
            len(X_external)
        ),
        "target": "CHURN",
        "training_churn_rate": float(
            y_calibration.mean()
        ),
        "external_validation_churn_rate": float(
            y_external.mean()
        ),
        "majority_class_accuracy_baseline": (
            majority_class_accuracy
        ),
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "external_validation_metrics": (
            external_metrics
        ),
        "internal_development_results": (
            development_results
        ),
        "features_used": list(
            X_calibration.columns
        ),
        "excluded_columns": sorted(
            excluded_columns
        ),
        "dropped_all_missing_training_columns": (
            dropped_columns
        ),
        "notes": [
            (
                "CHURN was used as the target in both "
                "calibration and external validation samples."
            ),
            (
                "CHURN and CHURNDEP were excluded from "
                "input features."
            ),
            (
                "The external validation sample was not "
                "used to choose the model."
            ),
            (
                "The calibration and external validation "
                "churn prevalences differ substantially. "
                "Interpret metrics with this limitation in mind."
            ),
        ],
    }

    model_path = (
        output_dir / "best_model.joblib"
    )

    metrics_path = (
        output_dir / "metrics.json"
    )

    joblib.dump(
        final_pipeline,
        model_path,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("EXTERNAL VALIDATION RESULTS")
    print("=" * 60)

    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    print("\nSaved files:")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
