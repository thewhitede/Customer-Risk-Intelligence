USE customer_risk_intelligence;

CREATE OR REPLACE VIEW v_customer_risk AS
SELECT
    c.customer_id,
    c.contract,
    c.tenure,
    c.monthly_charges,
    c.internet_service,
    c.payment_method,
    r.calibrated_churn_probability,
    r.calibrated_risk_level,
    r.revenue_at_risk,
    r.retention_priority_score,
    r.retention_priority,
    r.retention_recommendation,
    r.model_version,
    r.scored_at
FROM customers c
JOIN customer_risk_scores r ON r.customer_id = c.customer_id;

CREATE OR REPLACE VIEW v_retention_priority_summary AS
SELECT
    retention_priority,
    COUNT(*) AS customers,
    SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) AS observed_churners,
    ROUND(AVG(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) * 100, 2) AS observed_churn_rate_pct,
    ROUND(SUM(r.revenue_at_risk), 2) AS total_revenue_at_risk,
    ROUND(AVG(r.revenue_at_risk), 2) AS average_revenue_at_risk
FROM customer_risk_scores r
JOIN customers c ON c.customer_id = r.customer_id
GROUP BY retention_priority;
