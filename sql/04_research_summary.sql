USE customer_risk_intelligence;

CREATE TABLE IF NOT EXISTS research_summary (
    metric_group VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value_decimal DECIMAL(14,6) NULL,
    value_text VARCHAR(255) NULL,
    PRIMARY KEY (metric_group, metric_name)
);

INSERT INTO research_summary(metric_group, metric_name, value_decimal, value_text) VALUES
('baseline', 'logistic_roc_auc', 0.8419, NULL),
('research_model', 'accuracy', 0.7686, NULL),
('research_model', 'precision', 0.5548, NULL),
('research_model', 'recall', 0.6497, NULL),
('research_model', 'f1', 0.5985, NULL),
('research_model', 'roc_auc', 0.8208, NULL),
('research_model', 'pr_auc', 0.6069, NULL),
('calibration', 'brier_original', 0.1562, NULL),
('calibration', 'brier_calibrated', 0.1431, NULL),
('calibration', 'roc_auc_calibrated', 0.8276, NULL),
('calibration', 'log_loss_original', 0.4768, NULL),
('calibration', 'log_loss_calibrated', 0.4397, NULL),
('revenue', 'monthly_revenue_at_risk', 27257.11, NULL),
('risk', 'high_risk_customers', 185, NULL),
('risk', 'high_risk_observed_churn_pct', 69.73, NULL)
ON DUPLICATE KEY UPDATE value_decimal = VALUES(value_decimal), value_text = VALUES(value_text);
