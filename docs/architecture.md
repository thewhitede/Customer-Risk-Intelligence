# Architecture

## Analytical layer

1. Customer data
2. Cleaning and research feature engineering
3. Research Random Forest
4. Probability calibration
5. SHAP explanation
6. Calibrated risk segmentation
7. Revenue at Risk
8. Revenue Risk Score
9. Retention Priority Score
10. Retention targeting

## Application layer

MySQL provides persistent customer and scoring storage.

Streamlit provides a user-facing interface for:
- portfolio-level risk overview
- individual customer scoring
- retention prioritization
- established research evidence

## GenAI layer

Hugging Face is a language layer on top of already-computed model signals. It converts model outputs into plain-language business explanations. It is not the predictive engine.
