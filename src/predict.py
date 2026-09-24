"""Customer scoring service used by Streamlit and other application layers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import add_research_features
from .risk import assign_calibrated_risk, revenue_at_risk, retention_priority_score, assign_retention_priority, retention_recommendation


class CustomerRiskScorer:
    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def score(self, customers: pd.DataFrame) -> pd.DataFrame:
        data = add_research_features(customers)
        X = data.drop(columns=["Churn", "customerID"], errors="ignore")
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        calibrated_prob = self.model.predict_proba(X)[:, 1]
        data = data.copy()
        data["CalibratedChurnProbability"] = calibrated_prob
        data["CalibratedRiskLevel"] = assign_calibrated_risk(data["CalibratedChurnProbability"])
        data["RevenueAtRisk"] = revenue_at_risk(
            data["CalibratedChurnProbability"], data["MonthlyCharges"]
        )
        data["RevenueRiskScore"] = data["RevenueAtRisk"] / data["RevenueAtRisk"].max()
        data["RetentionPriorityScore"] = retention_priority_score(
            data["CalibratedChurnProbability"], data["RevenueAtRisk"]
        )
        data["RetentionPriority"] = assign_retention_priority(data["RetentionPriorityScore"])
        data["RetentionRecommendation"] = data.apply(retention_recommendation, axis=1)
        return data
