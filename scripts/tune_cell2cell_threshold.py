
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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


DATA_PATH = Path("data/raw/cell2cell_churn.csv")
MODEL_PATH = Path("models/cell2cell/best_model.joblib")
OUTPUT_PATH = Path("models/cell2cell/threshold_evaluation.json")


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Use the existing validation sample and its known CHURN labels.
    external = df.loc[df["CALIBRAT"].eq(0)].dropna(
        subset=["CHURN"]
    ).copy()

    feature_cols = list(model.feature_names_in_)
    X = external[feature_cols]
    y = pd.to_numeric(external["CHURN"]).astype(int)

    # Threshold selection and evaluation use different customers.
    X_tune, X_test, y_tune, y_test = train_test_split(
        X,
        y,
        test_size=0.50,
        random_state=42,
        stratify=y,
    )

    tune_prob = model.predict_proba(X_tune)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # Pick the threshold with the highest F1 on the tuning half.
    candidate_thresholds = np.unique(
        np.quantile(tune_prob, np.linspace(0, 1, 1001))
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        predictions = (tune_prob >= threshold).astype(int)
        score = f1_score(
            y_tune, predictions, zero_division=0
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    # Report performance only on the separate evaluation half.
    test_pred = (test_prob >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    results = {
        "threshold_selected_on_tuning_half": best_threshold,
        "tuning_half_rows": int(len(y_tune)),
        "evaluation_half_rows": int(len(y_test)),
        "evaluation_churn_rate": float(y_test.mean()),
        "all_negative_accuracy_baseline": float(
            (y_test == 0).mean()
        ),
        "metrics": {
            "accuracy": float(
                accuracy_score(y_test, test_pred)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, test_pred)
            ),
            "precision": float(
                precision_score(
                    y_test, test_pred, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, test_pred, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_test, test_pred, zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, test_prob)
            ),
            "average_precision": float(
                average_precision_score(y_test, test_prob)
            ),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
