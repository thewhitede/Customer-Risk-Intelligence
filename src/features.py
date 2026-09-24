"""Feature engineering used by the Customer Risk Intelligence research model."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49-72 months"]
CHARGE_BINS = [-np.inf, 40, 70, 90, np.inf]
CHARGE_LABELS = ["Low", "Medium-Low", "Medium-High", "High"]

BASE_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup", "MonthlyChargeGroup"
]
BASE_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "AnnualRevenue"]
TARGET = "Churn"
ID_COL = "customerID"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load and perform the same basic cleaning used in the research notebook."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three established research features without altering source columns."""
    out = df.copy()
    out["AnnualRevenue"] = out["MonthlyCharges"] * 12
    out["TenureGroup"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    )
    out["MonthlyChargeGroup"] = pd.cut(
        out["MonthlyCharges"], bins=CHARGE_BINS, labels=CHARGE_LABELS
    )
    return out


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X/y exactly aligned to the research feature set."""
    out = add_research_features(df)
    if TARGET not in out.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    y = out[TARGET].map({"No": 0, "Yes": 1}) if out[TARGET].dtype == object else out[TARGET]
    if y.isna().any():
        raise ValueError("Target contains unmapped/null values after target conversion.")
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = out.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, y.map({"No": 0, "Yes": 1}).astype(int)


def feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in BASE_NUMERIC if c in X.columns]
    categorical = [c for c in BASE_CATEGORICAL if c in X.columns]
    if set(numeric + categorical) != set(X.columns):
        missing = sorted(set(X.columns) - set(numeric + categorical))
        raise ValueError(f"Unexpected model columns: {missing}")
    return numeric, categorical
