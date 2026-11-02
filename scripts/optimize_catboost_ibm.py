
"""
CatBoost + Optuna benchmark for IBM Telco churn.

- Uses the original IBM Telco dataset and existing feature engineering.
- Handles categorical variables natively with CatBoost.
- Tunes CatBoost with Optuna using 3-fold stratified CV.
- Selects an F1 threshold using validation data only.
- Compares CatBoost with the current Gradient Boosting candidate.
- Saves all outputs separately from established research artifacts.

This is an exploratory experiment. It does not replace the
existing production model or research results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.base import clone
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
OUTPUT_DIR_DEFAULT = "results/ibm_catboost_optuna"


def identify_and_prepare_features(X):
    """Identify categorical columns and prepare their values for CatBoost."""

    X = X.copy()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_columns = [
        column for column in X.columns
        if column not in categorical_columns
    ]

    # CatBoost categorical values must be consistently represented.
    # Use an explicit category for missing categorical values.
    for column in categorical_columns:
        X[column] = (
            X[column]
            .astype("object")
            .where(X[column].notna(), "__MISSING__")
            .astype(str)
        )

    # Keep numeric missing values as NaN; CatBoost supports them.
    for column in numeric_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )
        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return X, categorical_columns


def select_threshold(y_true, probabilities):
    """Choose the F1-maximizing threshold using validation data only."""

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
        threshold
        for threshold, score in scores
        if np.isclose(score, best_f1)
    ]

    return min(
        tied,
        key=lambda value: abs(value - 0.50),
    )


def calculate_metrics(y_true, probabilities, threshold):
    """Return classification and probability-quality metrics."""

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


def make_catboost(params, categorical_columns):
    """Build a CatBoost classifier with the selected hyperparameters."""

    return CatBoostClassifier(
        **params,
        cat_features=categorical_columns,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
        thread_count=4,
    )


def tune_catboost(
    X_train,
    y_train,
    categorical_columns,
    n_trials,
):
    """Optimize average precision with stratified CV on training data."""

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=314,
    )

    def objective(trial):
        params = {
            "iterations": trial.suggest_int(
                "iterations",
                400,
                1000,
                step=100,
            ),
            "depth": trial.suggest_int(
                "depth",
                4,
                8,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.015,
                0.15,
                log=True,
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                0.5,
                20.0,
                log=True,
            ),
            "random_strength": trial.suggest_float(
                "random_strength",
                0.0,
                2.0,
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0.0,
                5.0,
            ),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights",
                [None, "Balanced"],
            ),
        }

        fold_scores = []

        for train_indices, validation_indices in cv.split(
            X_train,
            y_train,
        ):
            X_fold_train = X_train.iloc[train_indices]
            X_fold_valid = X_train.iloc[validation_indices]

            y_fold_train = y_train.iloc[train_indices]
            y_fold_valid = y_train.iloc[validation_indices]

            model = make_catboost(
                params,
                categorical_columns,
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            probabilities = model.predict_proba(
                X_fold_valid
            )[:, 1]

            fold_scores.append(
                average_precision_score(
                    y_fold_valid,
                    probabilities,
                )
            )

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        study_name="ibm_telco_catboost",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    return study


def main():
    parser = argparse.ArgumentParser(
        description="Tune CatBoost on the original IBM Telco churn dataset."
    )

    parser.add_argument(
        "--input",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the original IBM Telco CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Separate directory for CatBoost experiment outputs.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna hyperparameter trials.",
    )

    args = parser.parse_args()

    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1.")

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print("=" * 72)
    print("CATBOOST + OPTUNA: IBM TELCO CHURN")
    print("=" * 72)

    raw_data = load_raw_data(data_path)
    X, y = prepare_model_frame(raw_data)

    if y.isna().any() or y.nunique() != 2:
        raise ValueError(
            "The churn target must contain two valid classes."
        )

    X, categorical_columns = identify_and_prepare_features(X)

    # Overall proportions are approximately 60% train, 20% validation,
    # and 20% test. Keep the test set away from model tuning and threshold
    # selection.
    X_train_valid, X_test, y_train_valid, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=2026,
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_valid,
            y_train_valid,
            test_size=0.25,
            stratify=y_train_valid,
            random_state=2027,
        )
    )

    print(f"\nRows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Categorical features: {categorical_columns}")

    print("\nStarting Optuna search...")
    print(
        "Primary objective: mean 3-fold cross-validation "
        "average precision."
    )

    study = tune_catboost(
        X_train,
        y_train,
        categorical_columns,
        args.n_trials,
    )

    print("\nBest CV average precision:")
    print(f"{study.best_value:.4f}")

    print("\nBest parameters:")
    print(
        json.dumps(
            study.best_params,
            indent=2,
        )
    )

    # Fit CatBoost on the training split.
    catboost_model = make_catboost(
        study.best_params,
        categorical_columns,
    )

    print("\nFitting best CatBoost model...")
    catboost_model.fit(
        X_train,
        y_train,
    )

    # Tune the decision threshold using validation rows only.
    catboost_validation_probabilities = (
        catboost_model.predict_proba(X_validation)[:, 1]
    )

    catboost_threshold = select_threshold(
        y_validation,
        catboost_validation_probabilities,
    )

    # Fit and evaluate the Gradient Boosting reference on the same split.
    reference_pipeline, _, _ = build_pipeline_for_frame(
        X_train
    )

    if "preprocessor" not in reference_pipeline.named_steps:
        raise ValueError(
            "The existing pipeline must contain a 'preprocessor' step."
        )

    gradient_boosting = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(
                    reference_pipeline.named_steps["preprocessor"]
                ),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    subsample=0.85,
                    n_estimators=250,
                    min_samples_leaf=5,
                    max_depth=3,
                    learning_rate=0.03,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nFitting the Gradient Boosting reference...")
    gradient_boosting.fit(
        X_train,
        y_train,
    )

    gb_validation_probabilities = (
        gradient_boosting.predict_proba(X_validation)[:, 1]
    )

    gb_threshold = select_threshold(
        y_validation,
        gb_validation_probabilities,
    )

    # Final evaluation on the held-out test split.
    catboost_test_probabilities = (
        catboost_model.predict_proba(X_test)[:, 1]
    )

    gb_test_probabilities = (
        gradient_boosting.predict_proba(X_test)[:, 1]
    )

    results = {
        "catboost": {
            "selected_threshold": float(catboost_threshold),
            "test_at_threshold_0_50": calculate_metrics(
                y_test,
                catboost_test_probabilities,
                0.50,
            ),
            "test_at_validation_selected_threshold": calculate_metrics(
                y_test,
                catboost_test_probabilities,
                catboost_threshold,
            ),
        },
        "gradient_boosting_reference": {
            "selected_threshold": float(gb_threshold),
            "test_at_threshold_0_50": calculate_metrics(
                y_test,
                gb_test_probabilities,
                0.50,
            ),
            "test_at_validation_selected_threshold": calculate_metrics(
                y_test,
                gb_test_probabilities,
                gb_threshold,
            ),
        },
    }

    # Save the tuned CatBoost model separately.
    model_path = output_dir / "catboost_best.cbm"
    catboost_model.save_model(str(model_path))

    # Save the selected hyperparameters and metrics.
    report = {
        "dataset": str(data_path),
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "categorical_features": categorical_columns,
        "split": {
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "random_states": [2026, 2027],
        },
        "optuna": {
            "n_trials": int(args.n_trials),
            "cv_folds": 3,
            "selection_metric": "average_precision",
            "best_cv_average_precision": float(
                study.best_value
            ),
            "best_parameters": study.best_params,
        },
        "results": results,
        "notes": [
            (
                "CatBoost uses native categorical feature handling."
            ),
            (
                "Optuna model selection uses training data only."
            ),
            (
                "Thresholds are selected on validation data, "
                "not test labels."
            ),
            (
                "The Gradient Boosting model is evaluated on the same "
                "split for comparison."
            ),
            (
                "This is exploratory because the dataset has already "
                "been used in earlier experiments."
            ),
        ],
    }

    metrics_path = output_dir / "metrics.json"
    trials_path = output_dir / "optuna_trials.csv"
    predictions_path = output_dir / "test_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    study.trials_dataframe().to_csv(
        trials_path,
        index=False,
    )

    prediction_table = pd.DataFrame({
        "actual_churn": y_test.to_numpy(dtype=int),
        "catboost_probability": catboost_test_probabilities,
        "gradient_boosting_probability": gb_test_probabilities,
        "catboost_prediction_at_0_50": (
            catboost_test_probabilities >= 0.50
        ).astype(int),
        "catboost_prediction_at_selected_threshold": (
            catboost_test_probabilities >= catboost_threshold
        ).astype(int),
        "gradient_boosting_prediction_at_0_50": (
            gb_test_probabilities >= 0.50
        ).astype(int),
        "gradient_boosting_prediction_at_selected_threshold": (
            gb_test_probabilities >= gb_threshold
        ).astype(int),
    })

    prediction_table.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL TEST COMPARISON")
    print("=" * 100)

    for model_name, model_result in results.items():
        print(f"\n{model_name}")
        print(
            "Validation-selected threshold: "
            f"{model_result['selected_threshold']:.2f}"
        )

        print("\nThreshold 0.50:")
        print(
            json.dumps(
                model_result["test_at_threshold_0_50"],
                indent=2,
            )
        )

        print("\nValidation-selected threshold:")
        print(
            json.dumps(
                model_result[
                    "test_at_validation_selected_threshold"
                ],
                indent=2,
            )
        )

    print("\nSaved files:")
    print(f"  Model:      {model_path}")
    print(f"  Metrics:    {metrics_path}")
    print(f"  Optuna:     {trials_path}")
    print(f"  Test data:  {predictions_path}")

    print(
        "\nExisting production models and established research "
        "artifacts were not overwritten."
    )


if __name__ == "__main__":
    main()
