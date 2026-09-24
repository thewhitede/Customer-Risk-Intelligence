"""Established research evidence used by documentation/app copy."""
RESEARCH_METRICS = {
    "baseline_logistic_roc_auc": 0.8419,
    "research_rf_accuracy": 0.7686,
    "research_rf_precision": 0.5548,
    "research_rf_recall": 0.6497,
    "research_rf_f1": 0.5985,
    "research_rf_roc_auc": 0.8208,
    "research_rf_pr_auc": 0.6069,
    "brier_original": 0.1562,
    "brier_calibrated": 0.1431,
    "roc_auc_calibrated": 0.8276,
    "log_loss_original": 0.4768,
    "log_loss_calibrated": 0.4397,
    "monthly_revenue_at_risk": 27257.11,
}

RISK_VALIDATION = [
    {"risk": "Low", "customers": 917, "observed_churn_pct": 11.78},
    {"risk": "Medium", "customers": 307, "observed_churn_pct": 44.63},
    {"risk": "High", "customers": 185, "observed_churn_pct": 69.73},
]

TARGETING = [
    {"target": "High", "customers": 185, "customer_pct": 13.13, "capture_pct": 34.49, "efficiency": 2.63},
    {"target": "High + Medium", "customers": 492, "customer_pct": 34.92, "capture_pct": 71.12, "efficiency": 2.04},
    {"target": "High + Medium + Low", "customers": 1409, "customer_pct": 100.00, "capture_pct": 100.00, "efficiency": 1.00},
]

RETENTION_PRIORITY = [
    {"priority": "Low", "customers": 870, "churners": 93, "churn_rate_pct": 10.69, "revenue_at_risk": 5691.25},
    {"priority": "Moderate", "customers": 280, "churners": 119, "churn_rate_pct": 42.50, "revenue_at_risk": 7573.33},
    {"priority": "High", "customers": 187, "churners": 107, "churn_rate_pct": 57.22, "revenue_at_risk": 9115.72},
    {"priority": "Critical", "customers": 72, "churners": 55, "churn_rate_pct": 76.39, "revenue_at_risk": 4876.81},
]
