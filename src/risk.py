"""Risk, revenue exposure, and retention prioritization logic."""
from __future__ import annotations

import pandas as pd


def assign_calibrated_risk(probability: pd.Series) -> pd.Categorical:
    return pd.cut(
        probability,
        bins=[-float("inf"), 0.30, 0.60, float("inf")],
        labels=["Low", "Medium", "High"],
        right=False,
    )


def revenue_at_risk(probability: pd.Series, monthly_charges: pd.Series) -> pd.Series:
    return probability * monthly_charges


def retention_priority_score(probability: pd.Series, revenue_risk: pd.Series) -> pd.Series:
    max_risk = revenue_risk.max()
    if max_risk <= 0:
        return pd.Series(0.0, index=revenue_risk.index)
    revenue_risk_score = revenue_risk / max_risk
    return 0.60 * probability + 0.40 * revenue_risk_score


def assign_retention_priority(score: pd.Series) -> pd.Categorical:
    return pd.cut(
        score,
        bins=[-0.01, 0.25, 0.50, 0.75, 1.00],
        labels=["Low", "Moderate", "High", "Critical"],
    )


def retention_recommendation(row: pd.Series) -> str:
    priority = str(row.get("RetentionPriority", ""))
    contract = str(row.get("Contract", ""))
    tenure = row.get("tenure")
    payment = str(row.get("PaymentMethod", ""))
    internet = str(row.get("InternetService", ""))
    signals: list[str] = []

    if contract == "Month-to-month":
        signals.append("offer a longer-term contract incentive")
    if pd.notna(tenure) and float(tenure) <= 12:
        signals.append("use an early-tenure onboarding/check-in intervention")
    if payment == "Electronic check":
        signals.append("offer a lower-friction payment option")
    if internet == "Fiber optic":
        signals.append("review service/value concerns and premium-plan fit")
    if str(row.get("TechSupport", "")) == "No":
        signals.append("offer technical-support outreach")

    if priority == "Critical":
        action = "Immediate high-touch retention outreach"
    elif priority == "High":
        action = "Priority retention outreach"
    elif priority == "Moderate":
        action = "Targeted retention campaign"
    else:
        action = "Low-cost monitoring and proactive engagement"

    if signals:
        return action + ": " + "; ".join(signals) + "."
    return action + "."
