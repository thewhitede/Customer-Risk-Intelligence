# Data Dictionary

| Field | Meaning |
|---|---|
| customerID | Customer identifier |
| tenure | Months with the telecom provider |
| MonthlyCharges | Current recurring monthly charge |
| TotalCharges | Historical total charges |
| AnnualRevenue | MonthlyCharges × 12 |
| TenureGroup | 0–6, 7–12, 13–24, 25–48, 49–72 months |
| MonthlyChargeGroup | Low, Medium-Low, Medium-High, High |
| CalibratedChurnProbability | Calibrated probability of churn |
| CalibratedRiskLevel | Low / Medium / High risk segment |
| RevenueAtRisk | CalibratedChurnProbability × MonthlyCharges |
| RevenueRiskScore | Revenue-at-risk normalized by the maximum evaluated exposure |
| RetentionPriorityScore | 60% churn probability + 40% revenue risk score |
| RetentionPriority | Low / Moderate / High / Critical |
