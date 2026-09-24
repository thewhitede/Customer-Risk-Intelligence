import pandas as pd
from src.features import add_research_features
from src.risk import assign_calibrated_risk, revenue_at_risk, retention_priority_score


def test_research_features():
    df = pd.DataFrame({"tenure": [1, 8, 41, 72], "MonthlyCharges": [29.85, 56.95, 78.35, 114.05]})
    out = add_research_features(df)
    assert round(out.loc[0, "AnnualRevenue"], 1) == 358.2
    assert str(out.loc[0, "TenureGroup"]) == "0-6 months"
    assert str(out.loc[3, "TenureGroup"]) == "49-72 months"
    assert str(out.loc[1, "MonthlyChargeGroup"]) == "Medium-Low"
    assert str(out.loc[2, "MonthlyChargeGroup"]) == "Medium-High"


def test_risk_bins():
    probs = pd.Series([0.2999, 0.30, 0.5999, 0.60, 0.9])
    risk = assign_calibrated_risk(probs)
    assert list(map(str, risk)) == ["Low", "Medium", "Medium", "High", "High"]


def test_revenue_and_priority():
    p = pd.Series([0.2, 0.5])
    m = pd.Series([50.0, 100.0])
    r = revenue_at_risk(p, m)
    s = retention_priority_score(p, r)
    assert r.tolist() == [10.0, 50.0]
    assert s.iloc[1] > s.iloc[0]
