"""Model training and serialization for the research-consistent pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import load_raw_data, prepare_model_frame, feature_groups


def build_research_pipeline() -> Pipeline:
    numeric_features = None
    categorical_features = None
    raise RuntimeError("Use build_pipeline_for_frame(X) after feature engineering so column groups are explicit.")


def build_pipeline_for_frame(X: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    numeric_features, categorical_features = feature_groups(X)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipe, numeric_features, categorical_features


def fit_calibrated_production_model(X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """Fit a production refit using the same research architecture.

    This is a deployment artifact, not the research evaluation run. The research
    metrics remain those reported in Notebook 03.
    """
    base_pipeline, _, _ = build_pipeline_for_frame(X)
    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X, y)
    return calibrated


def train_from_csv(data_path: str | Path, model_path: str | Path, metadata_path: str | Path) -> dict[str, Any]:
    df = load_raw_data(data_path)
    X, y = prepare_model_frame(df)
    model = fit_calibrated_production_model(X, y)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        "artifact_type": "production_refit",
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "rf_n_estimators": 300,
        "rf_random_state": 42,
        "rf_class_weight": "balanced",
        "calibration_method": "sigmoid",
        "calibration_cv": 5,
        "research_reference_metrics": {
            "accuracy": 0.7686,
            "precision": 0.5548,
            "recall": 0.6497,
            "f1": 0.5985,
            "roc_auc": 0.8208,
            "pr_auc": 0.6069,
        },
        "research_reference_calibration": {
            "brier_original": 0.1562,
            "brier_calibrated": 0.1431,
            "roc_auc_original": 0.8208,
            "roc_auc_calibrated": 0.8276,
            "log_loss_original": 0.4768,
            "log_loss_calibrated": 0.4397,
        },
    }
    Path(metadata_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
