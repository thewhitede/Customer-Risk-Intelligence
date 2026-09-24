CREATE DATABASE IF NOT EXISTS customer_risk_intelligence;
USE customer_risk_intelligence;

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(20),
    senior_citizen TINYINT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INT,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(40),
    internet_service VARCHAR(40),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(30),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(40),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(12,2),
    churn VARCHAR(5),
    annual_revenue DECIMAL(12,2),
    tenure_group VARCHAR(30),
    monthly_charge_group VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_contract (contract),
    INDEX idx_internet_service (internet_service),
    INDEX idx_payment_method (payment_method)
);

CREATE TABLE IF NOT EXISTS customer_risk_scores (
    score_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    calibrated_churn_probability DECIMAL(8,6) NOT NULL,
    calibrated_risk_level VARCHAR(20) NOT NULL,
    revenue_at_risk DECIMAL(12,2) NOT NULL,
    revenue_risk_score DECIMAL(8,6) NOT NULL,
    retention_priority_score DECIMAL(8,6) NOT NULL,
    retention_priority VARCHAR(20) NOT NULL,
    retention_recommendation TEXT,
    model_version VARCHAR(100),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_customer_score_version (customer_id, model_version),
    CONSTRAINT fk_risk_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    INDEX idx_risk_level (calibrated_risk_level),
    INDEX idx_priority (retention_priority),
    INDEX idx_probability (calibrated_churn_probability)
);

CREATE TABLE IF NOT EXISTS retention_actions (
    action_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    action_status VARCHAR(30) DEFAULT 'Planned',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_action_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    INDEX idx_action_status (action_status)
);
