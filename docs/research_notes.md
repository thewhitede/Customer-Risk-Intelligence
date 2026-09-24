# Research Notes

## Research question

Can an explainable machine-learning framework identify high-risk telecom customers and prioritize retention interventions using churn probability and customer value?

## Evidence chain

The project deliberately moves from predictive performance toward decision usefulness:

**Prediction → Calibration → Explanation → Risk → Financial Exposure → Retention Priority**

## Interpretation

The research Random Forest achieved strong recall relative to the baseline Random Forest while calibrated probabilities improved probability quality on the held-out evaluation set. Risk segmentation then separated customers into visibly different observed churn-rate bands. Revenue-aware prioritization added customer-value information to the retention ranking.

The analysis is observational and historical. It does not estimate treatment effects and therefore cannot prove that a proposed retention action will prevent churn.
