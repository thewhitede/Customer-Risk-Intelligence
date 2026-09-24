USE customer_risk_intelligence;

-- Update the file path for your local environment before executing.
-- Requires MySQL LOCAL INFILE to be enabled.
LOAD DATA LOCAL INFILE 'data/raw/telco_customer_churn.csv'
INTO TABLE customers
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(customer_id, gender, senior_citizen, partner, dependents, tenure,
 phone_service, multiple_lines, internet_service, online_security,
 online_backup, device_protection, tech_support, streaming_tv,
 streaming_movies, contract, paperless_billing, payment_method,
 monthly_charges, total_charges, churn)
SET total_charges = NULLIF(total_charges, ''),
    annual_revenue = monthly_charges * 12,
    tenure_group = CASE
        WHEN tenure BETWEEN 0 AND 6 THEN '0-6 months'
        WHEN tenure BETWEEN 7 AND 12 THEN '7-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        WHEN tenure BETWEEN 49 AND 72 THEN '49-72 months'
    END,
    monthly_charge_group = CASE
        WHEN monthly_charges < 40 THEN 'Low'
        WHEN monthly_charges < 70 THEN 'Medium-Low'
        WHEN monthly_charges <= 90 THEN 'Medium-High'
        ELSE 'High'
    END;
