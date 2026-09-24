"""SHAP explanations for the underlying Random Forest used by the app.

The displayed churn probability is calibrated. SHAP values here explain the
underlying Random Forest probability output because the calibration layer is a
post-model probability transformation.
"""
from __future__ import annotations

import pandas as pd


def explain_customer(calibrated_model, customer_frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    import shap

    if len(customer_frame) != 1:
        raise ValueError("explain_customer expects exactly one customer row")

    # CalibratedClassifierCV stores one fitted base estimator per CV fold.
    # Use the first fitted estimator for an interpretable local explanation.
    calibrated_estimators = getattr(calibrated_model, "calibrated_classifiers_", None)
    if not calibrated_estimators:
        raise ValueError("Model does not expose fitted calibrated estimators.")
    base_pipeline = calibrated_estimators[0].estimator
    preprocessor = base_pipeline.named_steps["preprocessor"]
    rf = base_pipeline.named_steps["model"]

    transformed = preprocessor.transform(customer_frame)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    explainer = shap.TreeExplainer(rf)
    values = explainer.shap_values(transformed)
    # SHAP API differs by version: class probabilities may be a list or 3-D array.
    if isinstance(values, list):
        contribution = values[1][0]
    else:
        arr = values
        if getattr(arr, "ndim", 2) == 3:
            contribution = arr[0, :, 1]
        else:
            contribution = arr[0]

    result = pd.DataFrame({
        "Feature": feature_names,
        "SHAPValue": contribution,
        "AbsoluteSHAP": abs(contribution),
    }).sort_values("AbsoluteSHAP", ascending=False).head(top_n)
    return result.reset_index(drop=True)
