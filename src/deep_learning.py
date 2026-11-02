"""Deep-learning add-on for Customer Risk Intelligence Platform.

This module trains a PyTorch MLP against the existing research feature set and,
optionally, a hybrid MLP with aggregated customer-text features.

The training process includes:
- Automatic weight and bias learning through backpropagation.
- AdamW optimization and automatic learning-rate scheduling.
- Early stopping based on validation loss.
- Validation-based probability calibration and threshold selection.
- Evaluation on a separate, held-out test set.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    ID_COL,
    TARGET,
    add_research_features,
    feature_groups,
    load_raw_data,
)
from .modeling import build_pipeline_for_frame

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    DataLoader = None
    TensorDataset = None


TEXT_FEATURE_COLUMNS = [
    "ticket_count",
    "negative_ticket_count",
    "negative_ticket_ratio",
    "mean_sentiment_score",
    "latest_sentiment_score",
]


def _prepare_addon_model_frame(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the existing research feature set and churn target."""

    out = add_research_features(raw_df.copy())

    if TARGET not in out.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    target = out[TARGET]

    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
    ):
        normalized = target.astype(str).str.strip().str.lower()
        mapping = {"no": 0, "yes": 1, "0": 0, "1": 1}
        y = normalized.map(mapping)
    else:
        y = pd.to_numeric(target, errors="coerce")

    if (
        y.isna().any()
        or not set(pd.Series(y).unique()).issubset({0, 1})
    ):
        raise ValueError(
            "Churn target must contain only No/Yes or 0/1 values "
            "without missing labels."
        )

    X = out.drop(columns=[TARGET, ID_COL], errors="ignore").copy()

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"],
            errors="coerce",
        )

    return X, pd.Series(
        y.to_numpy(dtype=int),
        index=X.index,
        name=TARGET,
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible neural-network training configuration."""

    random_state: int = 42
    test_size: float = 0.20
    validation_size_of_development_set: float = 0.20

    # Current experimental architecture.
    hidden_dims: tuple[int, ...] = (256, 128, 64, 32)
    dropout: float = 0.35

    learning_rate: float = 5e-4
    weight_decay: float = 5e-4

    batch_size: int = 64
    max_epochs: int = 120
    patience: int = 15
    min_delta: float = 1e-4
    threshold: float = 0.50

    # Automatic learning-rate scheduler.
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-5


def _require_torch():
    """Check that PyTorch is installed."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required. Install dependencies using "
            "'python -m pip install -r requirements.txt'."
        )

    return torch


def _make_mlp_class():
    """Build a feed-forward neural network."""

    torch_module = _require_torch()
    nn_module = torch_module.nn

    class CustomerChurnMLP(nn_module.Module):

        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            dropout: float,
        ):
            super().__init__()

            layers: list[Any] = []
            previous = input_dim

            for width in hidden_dims:
                layers.extend(
                    [
                        nn_module.Linear(previous, int(width)),
                        nn_module.ReLU(),
                        nn_module.Dropout(float(dropout)),
                    ]
                )
                previous = int(width)

            # One output logit for binary classification.
            layers.append(nn_module.Linear(previous, 1))
            self.network = nn_module.Sequential(*layers)

        def forward(self, features):
            return self.network(features).squeeze(1)

    return CustomerChurnMLP


def _seed_everything(seed: int) -> None:
    """Make training as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)

    torch_module = _require_torch()
    torch_module.manual_seed(seed)

    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _build_preprocessor(
    X: pd.DataFrame,
    extra_numeric_features: Sequence[str] = (),
):
    """Build preprocessing fitted on training rows only."""

    extras = list(extra_numeric_features)

    base_columns = [
        column for column in X.columns
        if column not in extras
    ]

    unknown_extras = set(extras) - set(X.columns)

    if unknown_extras:
        raise ValueError(
            f"Extra numeric features are missing: "
            f"{sorted(unknown_extras)}"
        )

    base_numeric, categorical = feature_groups(X[base_columns])

    for column in extras:
        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(
                f"NLP feature must be numeric: {column}"
            )

    numeric = base_numeric + extras

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )


def _metrics_and_curves(
    y_true: Any,
    probabilities: Any,
    threshold: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Calculate test metrics and data for evaluation plots."""

    y_true_array = np.asarray(y_true, dtype=int)

    probs = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-7,
        1 - 1e-7,
    )

    predictions = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true_array, predictions)
        ),
        "precision": float(
            precision_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true_array, probs)
        ),
        "pr_auc": float(
            average_precision_score(y_true_array, probs)
        ),
        "brier_score": float(
            brier_score_loss(y_true_array, probs)
        ),
        "log_loss": float(
            log_loss(y_true_array, probs, labels=[0, 1])
        ),
        "decision_threshold": float(threshold),
    }

    fpr, tpr, _ = roc_curve(y_true_array, probs)

    precision, recall, _ = precision_recall_curve(
        y_true_array,
        probs,
    )

    observed_rate, mean_probability = calibration_curve(
        y_true_array,
        probs,
        n_bins=min(10, max(2, len(y_true_array) // 5)),
        strategy="quantile",
    )

    curves = {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "precision_recall": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
        "calibration": {
            "mean_predicted_probability": (
                mean_probability.tolist()
            ),
            "observed_positive_rate": observed_rate.tolist(),
        },
    }

    return metrics, curves


def _select_accuracy_threshold(
    y_true: Any,
    probabilities: Any,
    default_threshold: float = 0.5,
) -> tuple[float, float]:
    """Select a decision threshold using validation accuracy.

    Do not pass the held-out test labels into this function.
    """

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    thresholds = np.linspace(0.05, 0.95, 181)

    results = [
        (
            float(threshold),
            float(
                accuracy_score(
                    y_true,
                    (probabilities >= threshold).astype(int),
                )
            ),
        )
        for threshold in thresholds
    ]

    best_accuracy = max(
        accuracy for _, accuracy in results
    )

    # Prefer a threshold closer to 0.5 when accuracy ties.
    best_threshold = min(
        (
            threshold
            for threshold, accuracy in results
            if np.isclose(accuracy, best_accuracy)
        ),
        key=lambda threshold: abs(
            threshold - default_threshold
        ),
    )

    return best_threshold, best_accuracy


def _fit_mlp_variant(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    variant_name: str,
    extra_numeric_features: Sequence[str],
    config: TrainingConfig,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    """Train, calibrate, evaluate and save one MLP."""

    torch_module = _require_torch()
    _seed_everything(config.random_state)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessor = _build_preprocessor(
        X_train,
        extra_numeric_features,
    )

    train_array = np.asarray(
        preprocessor.fit_transform(X_train),
        dtype=np.float32,
    )

    val_array = np.asarray(
        preprocessor.transform(X_val),
        dtype=np.float32,
    )

    test_array = np.asarray(
        preprocessor.transform(X_test),
        dtype=np.float32,
    )

    model_class = _make_mlp_class()

    device = torch_module.device(
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )

    model = model_class(
        train_array.shape[1],
        config.hidden_dims,
        config.dropout,
    ).to(device)

    # Weight the positive class because churn is the minority class.
    positive_count = max(
        int((y_train == 1).sum()),
        1,
    )

    negative_count = max(
        int((y_train == 0).sum()),
        1,
    )

    pos_weight = torch_module.tensor(
        [negative_count / positive_count],
        dtype=torch_module.float32,
        device=device,
    )

    criterion = torch_module.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # Optimizer learns weights and biases via backpropagation.
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Automatically lower the learning rate when validation loss stalls.
    scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.min_delta,
        min_lr=config.scheduler_min_lr,
    )

    train_dataset = torch_module.utils.data.TensorDataset(
        torch_module.tensor(
            train_array,
            dtype=torch_module.float32,
        ),
        torch_module.tensor(
            y_train.to_numpy(dtype=np.float32),
            dtype=torch_module.float32,
        ),
    )

    generator = torch_module.Generator()
    generator.manual_seed(config.random_state)

    train_loader = torch_module.utils.data.DataLoader(
        train_dataset,
        batch_size=max(
            1,
            min(config.batch_size, len(train_dataset)),
        ),
        shuffle=True,
        generator=generator,
    )

    X_val_tensor = torch_module.tensor(
        val_array,
        dtype=torch_module.float32,
        device=device,
    )

    y_val_tensor = torch_module.tensor(
        y_val.to_numpy(dtype=np.float32),
        dtype=torch_module.float32,
        device=device,
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()

        running_loss = 0.0
        rows_seen = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward propagation.
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_features)
            loss = criterion(logits, batch_labels)

            # Backpropagation computes gradients.
            loss.backward()

            # Prevent excessively large gradients.
            torch_module.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            # AdamW updates network weights and biases.
            optimizer.step()

            running_loss += (
                float(loss.item()) * len(batch_labels)
            )
            rows_seen += len(batch_labels)

        # Evaluate validation loss after each training epoch.
        model.eval()

        with torch_module.no_grad():
            val_logits_tensor = model(X_val_tensor)

            val_loss = float(
                criterion(
                    val_logits_tensor,
                    y_val_tensor,
                ).item()
            )

        # ADDED: the scheduler adjusts the learning rate automatically.
        scheduler.step(val_loss)

        train_loss = running_loss / max(rows_seen, 1)

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": current_learning_rate,
            }
        )

        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= config.patience:
                break

    # Restore the weights from the best validation-loss epoch.
    model.load_state_dict(best_state)
    model.eval()

    with torch_module.no_grad():
        val_logits = (
            model(X_val_tensor)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

        test_logits = (
            model(
                torch_module.tensor(
                    test_array,
                    dtype=torch_module.float32,
                    device=device,
                )
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

    if y_val.nunique() < 2:
        raise ValueError(
            "The validation split must contain both churn "
            "classes for probability calibration."
        )

    # Calibrate probabilities using validation logits only.
    probability_calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=config.random_state,
    )

    probability_calibrator.fit(
        val_logits,
        y_val.to_numpy(dtype=int),
    )

    val_probabilities = (
        probability_calibrator
        .predict_proba(val_logits)[:, 1]
    )

    # Select threshold using validation accuracy.
    selected_threshold, validation_accuracy = (
        _select_accuracy_threshold(
            y_val.to_numpy(dtype=int),
            val_probabilities,
            config.threshold,
        )
    )

    # Evaluate on the untouched test set.
    test_probabilities = (
        probability_calibrator
        .predict_proba(test_logits)[:, 1]
    )

    metrics, curves = _metrics_and_curves(
        y_test.to_numpy(),
        test_probabilities,
        selected_threshold,
    )

    metrics[
        "validation_accuracy_at_selected_threshold"
    ] = float(validation_accuracy)

    # Save fitted artifacts.
    joblib.dump(
        preprocessor,
        output_dir / "preprocessor.joblib",
    )

    joblib.dump(
        probability_calibrator,
        output_dir / "probability_calibrator.joblib",
    )

    torch_module.save(
        model.state_dict(),
        output_dir / "model_state.pt",
    )

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    metadata = {
        "artifact_type": "customer_risk_mlp",
        "variant": variant_name,
        "framework": "PyTorch",
        "feature_columns": list(X_train.columns),
        "extra_numeric_features": list(
            extra_numeric_features
        ),
        "input_dim": int(train_array.shape[1]),
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "random_state": config.random_state,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "training_device": str(device),
        "class_weighting": (
            "positive_class_weight = negative_count / positive_count"
        ),
        "probability_calibration": (
            "Platt scaling fitted on validation-set logits; "
            "test set held out"
        ),
        "selected_threshold": float(selected_threshold),
        "validation_accuracy_at_selected_threshold": float(
            validation_accuracy
        ),
        "threshold_selection": (
            "Selected by validation accuracy; "
            "test set not used for selection"
        ),
        "learning_rate_scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": config.scheduler_factor,
            "patience": config.scheduler_patience,
            "min_lr": config.scheduler_min_lr,
        },
        "config": asdict(config),
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return metrics, curves, metadata


def _add_text_features(
    X_base: pd.DataFrame,
    raw_df: pd.DataFrame,
    text_features_path: str | Path | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Join customer-level NLP features to structured customer data."""

    if text_features_path is None:
        return None, []

    if ID_COL not in raw_df.columns:
        raise ValueError(
            f"Base dataset must contain {ID_COL} to join NLP features."
        )

    text_path = Path(text_features_path)

    if not text_path.exists():
        raise FileNotFoundError(
            f"NLP features file not found: {text_path}"
        )

    text_df = pd.read_csv(text_path)

    needed = {ID_COL, *TEXT_FEATURE_COLUMNS}
    missing = sorted(needed - set(text_df.columns))

    if missing:
        raise ValueError(
            f"NLP features file is missing columns: {missing}"
        )

    if (
        text_df[ID_COL].isna().any()
        or text_df[ID_COL].duplicated().any()
    ):
        raise ValueError(
            f"NLP features must contain one non-null row per {ID_COL}."
        )

    # Preserve the customer cohort and join one row per customer.
    joined = raw_df[[ID_COL]].merge(
        text_df[[ID_COL, *TEXT_FEATURE_COLUMNS]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    extras = (
        joined[TEXT_FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    X_hybrid = X_base.reset_index(drop=True).copy()

    for column in TEXT_FEATURE_COLUMNS:
        X_hybrid[column] = extras[column].to_numpy(dtype=float)

    return X_hybrid, list(TEXT_FEATURE_COLUMNS)


def train_from_csv(
    data_path: str | Path,
    output_dir: str | Path,
    text_features_path: str | Path | None = None,
    config: TrainingConfig | None = None,
    include_rf_baseline: bool = True,
) -> dict[str, Any]:
    """Train structured MLP, optional hybrid MLP, and RF comparator.

    The same stratified split is reused for all models.
    Preprocessing is fitted on training rows only.
    Threshold selection uses validation data.
    Final metrics are measured on the held-out test set.
    """

    _require_torch()
    config = config or TrainingConfig()

    raw_df = load_raw_data(data_path)

    X_base, y = _prepare_addon_model_frame(raw_df)

    X_hybrid, extra_features = _add_text_features(
        X_base,
        raw_df,
        text_features_path,
    )

    if y.nunique() != 2:
        raise ValueError(
            "Churn target must contain both classes: No and Yes."
        )

    if len(raw_df) < 50:
        raise ValueError(
            "At least 50 labeled customer rows are recommended "
            "for a stable train/validation/test split."
        )

    all_indices = np.arange(len(raw_df))

    development_indices, test_indices = train_test_split(
        all_indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    train_indices, val_indices = train_test_split(
        development_indices,
        test_size=config.validation_size_of_development_set,
        random_state=config.random_state,
        stratify=y.iloc[development_indices],
    )

    def split_frame(frame: pd.DataFrame):
        return (
            frame.iloc[train_indices].reset_index(drop=True),
            frame.iloc[val_indices].reset_index(drop=True),
            frame.iloc[test_indices].reset_index(drop=True),
        )

    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_val = y.iloc[val_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison: dict[str, Any] = {
        "evaluation_protocol": {
            "dataset_rows": int(len(raw_df)),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(val_indices)),
            "test_rows": int(len(test_indices)),
            "random_state": config.random_state,
            "stratified": True,
            "preprocessing_fit_on_training_only": True,
            "decision_threshold": config.threshold,
            "threshold_selection": (
                "Accuracy threshold selected on validation data "
                "independently per model; final metrics use the "
                "held-out test set."
            ),
            "note": (
                "These are add-on hold-out results; they do not replace "
                "the project's established research metrics."
            ),
        },
        "metrics": {},
        "roc_curves": {},
        "precision_recall_curves": {},
        "calibration_curves": {},
        "artifacts": {},
    }

    variants: list[
        tuple[str, pd.DataFrame, list[str], str]
    ] = [
        (
            "DeepLearningMLP_Structured",
            X_base,
            [],
            "structured",
        ),
    ]

    if X_hybrid is not None:
        variants.append(
            (
                "DeepLearningMLP_HybridNLP",
                X_hybrid,
                extra_features,
                "hybrid",
            )
        )

    if include_rf_baseline:
        X_train_rf, X_val_rf, X_test_rf = split_frame(X_base)

        min_class_count = int(y_train.value_counts().min())

        if min_class_count < 2:
            raise ValueError(
                "The training split needs at least two examples "
                "per class for RF calibration."
            )

        cv_folds = min(5, min_class_count)

        rf_pipeline, _, _ = build_pipeline_for_frame(X_base)

        rf_model = CalibratedClassifierCV(
            estimator=rf_pipeline,
            method="sigmoid",
            cv=cv_folds,
        )

        rf_model.fit(X_train_rf, y_train)

        # Select the Random Forest threshold using validation data.
        rf_val_probabilities = rf_model.predict_proba(
            X_val_rf
        )[:, 1]

        rf_threshold, rf_validation_accuracy = (
            _select_accuracy_threshold(
                y_val.to_numpy(dtype=int),
                rf_val_probabilities,
                config.threshold,
            )
        )

        rf_probabilities = rf_model.predict_proba(
            X_test_rf
        )[:, 1]

        rf_metrics, rf_curves = _metrics_and_curves(
            y_test,
            rf_probabilities,
            rf_threshold,
        )

        rf_metrics[
            "validation_accuracy_at_selected_threshold"
        ] = float(rf_validation_accuracy)

        comparison["metrics"][
            "RandomForestCalibrated_Structured"
        ] = rf_metrics

        comparison["roc_curves"][
            "RandomForestCalibrated_Structured"
        ] = rf_curves["roc"]

        comparison["precision_recall_curves"][
            "RandomForestCalibrated_Structured"
        ] = rf_curves["precision_recall"]

        comparison["calibration_curves"][
            "RandomForestCalibrated_Structured"
        ] = rf_curves["calibration"]

        comparison["artifacts"][
            "RandomForestCalibrated_Structured"
        ] = (
            "Existing model pipeline evaluated on a new shared split; "
            "no RF artifact is saved by this add-on."
        )

    for model_key, frame, extras, folder in variants:
        X_train, X_val, X_test = split_frame(frame)

        metrics, curves, metadata = _fit_mlp_variant(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            output_dir / folder,
            model_key,
            extras,
            config,
        )

        comparison["metrics"][model_key] = metrics

        comparison["roc_curves"][model_key] = curves["roc"]

        comparison["precision_recall_curves"][
            model_key
        ] = curves["precision_recall"]

        comparison["calibration_curves"][
            model_key
        ] = curves["calibration"]

        comparison["artifacts"][model_key] = str(
            (output_dir / folder).as_posix()
        )

    comparison["dataset"] = {
        "path": str(Path(data_path)),
        "text_features_path": (
            str(Path(text_features_path))
            if text_features_path
            else None
        ),
        "text_features_used": extra_features,
    }

    comparison["training_config"] = asdict(config)

    comparison_path = output_dir / "comparison_metrics.json"

    comparison_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    return comparison


def _inference_frame(
    raw_customers: pd.DataFrame,
    metadata: dict[str, Any],
    text_features: pd.DataFrame | None,
) -> pd.DataFrame:
    """Prepare inference features in the same order as training."""

    base = raw_customers.copy()

    if "TotalCharges" in base.columns:
        base["TotalCharges"] = pd.to_numeric(
            base["TotalCharges"],
            errors="coerce",
        )

    engineered = add_research_features(base)

    X = engineered.drop(
        columns=[TARGET, ID_COL],
        errors="ignore",
    )

    extras = metadata.get("extra_numeric_features", [])

    if extras:
        if text_features is None:
            raise ValueError(
                "This hybrid model requires customer-level NLP "
                "feature data at prediction time."
            )

        if (
            ID_COL not in raw_customers.columns
            or ID_COL not in text_features.columns
        ):
            raise ValueError(
                f"Both input customers and NLP features "
                f"must include {ID_COL}."
            )

        if text_features[ID_COL].duplicated().any():
            raise ValueError(
                f"NLP features must contain one row per {ID_COL}."
            )

        missing = sorted(
            set(extras) - set(text_features.columns)
        )

        if missing:
            raise ValueError(
                f"NLP feature data is missing columns: {missing}"
            )

        joined = raw_customers[[ID_COL]].merge(
            text_features[[ID_COL, *extras]],
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )

        for column in extras:
            X[column] = (
                pd.to_numeric(
                    joined[column],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy()
            )

    feature_columns = metadata["feature_columns"]

    missing_base = sorted(
        set(feature_columns) - set(X.columns)
    )

    if missing_base:
        raise ValueError(
            f"Customer input is missing required features: {missing_base}"
        )

    return X[feature_columns]


def predict_deep_learning(
    raw_customers: pd.DataFrame,
    model_dir: str | Path,
    text_features: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return churn probabilities from a saved structured or hybrid MLP."""

    torch_module = _require_torch()

    model_dir = Path(model_dir)

    metadata = json.loads(
        (model_dir / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    preprocessor = joblib.load(
        model_dir / "preprocessor.joblib"
    )

    X = _inference_frame(
        raw_customers,
        metadata,
        text_features,
    )

    transformed = np.asarray(
        preprocessor.transform(X),
        dtype=np.float32,
    )

    model_class = _make_mlp_class()

    model = model_class(
        int(metadata["input_dim"]),
        metadata["hidden_dims"],
        float(metadata["dropout"]),
    )

    try:
        state = torch_module.load(
            model_dir / "model_state.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older supported PyTorch releases.
        state = torch_module.load(
            model_dir / "model_state.pt",
            map_location="cpu",
        )

    model.load_state_dict(state)
    model.eval()

    with torch_module.no_grad():
        logits = (
            model(
                torch_module.tensor(
                    transformed,
                    dtype=torch_module.float32,
                )
            )
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

    probability_calibrator = joblib.load(
        model_dir / "probability_calibrator.joblib"
    )

    probabilities = probability_calibrator.predict_proba(
        logits
    )[:, 1]

    return probabilities.astype(float)