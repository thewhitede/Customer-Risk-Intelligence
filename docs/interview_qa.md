# Interview Q&A

## Why calibrate churn probabilities?

Classification accuracy alone does not tell a business whether a predicted probability is reliable. Calibration makes predicted probabilities more useful for risk ranking, revenue exposure estimation, and threshold-based decision support.

## Why include revenue in retention prioritization?

Two customers can have similar churn probabilities but different financial exposure. The revenue-aware score adds customer value to the prioritization decision.

## Why use SHAP?

SHAP provides feature-level contribution explanations so a user can inspect why an individual prediction is high or low rather than receiving only a black-box score.

## Why not use GenAI as the churn model?

GenAI is used here as a language and decision-support layer. The tabular supervised model remains responsible for the numeric churn probability.

## What is a limitation?

The dataset is a static historical snapshot, and retention recommendations are not causal experiments. A future production system would require monitoring, retraining, and intervention-outcome measurement.
