"""Small MySQL persistence layer for scored customer records."""
from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


def get_connection():
    import mysql.connector
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "cri_user"),
        password=os.getenv("MYSQL_PASSWORD", "change_me"),
        database=os.getenv("MYSQL_DATABASE", "customer_risk_intelligence"),
    )


def upsert_scores(scored: pd.DataFrame, model_version: str = "production-refit-v1") -> int:
    conn = get_connection()
    sql = """
    INSERT INTO customer_risk_scores
    (customer_id, calibrated_churn_probability, calibrated_risk_level,
     revenue_at_risk, revenue_risk_score, retention_priority_score,
     retention_priority, retention_recommendation, model_version)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
      calibrated_churn_probability=VALUES(calibrated_churn_probability),
      calibrated_risk_level=VALUES(calibrated_risk_level),
      revenue_at_risk=VALUES(revenue_at_risk),
      revenue_risk_score=VALUES(revenue_risk_score),
      retention_priority_score=VALUES(retention_priority_score),
      retention_priority=VALUES(retention_priority),
      retention_recommendation=VALUES(retention_recommendation),
      scored_at=CURRENT_TIMESTAMP;
    """
    rows = []
    for _, r in scored.iterrows():
        rows.append((
            str(r["customerID"]),
            float(r["CalibratedChurnProbability"]),
            str(r["CalibratedRiskLevel"]),
            float(r["RevenueAtRisk"]),
            float(r["RevenueRiskScore"]),
            float(r["RetentionPriorityScore"]),
            str(r["RetentionPriority"]),
            str(r["RetentionRecommendation"]),
            model_version,
        ))
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    count = cur.rowcount
    cur.close()
    conn.close()
    return count

def load_customer_risk_scores() -> pd.DataFrame:
    """Load persisted customer risk scores from MySQL."""
    conn = get_connection()

    sql = """
    SELECT
        customer_id,
        contract,
        tenure,
        monthly_charges,
        internet_service,
        payment_method,
        calibrated_churn_probability,
        calibrated_risk_level,
        revenue_at_risk,
        retention_priority_score,
        retention_priority,
        retention_recommendation,
        model_version,
        scored_at
    FROM v_customer_risk
    ORDER BY retention_priority_score DESC;
    """

    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()