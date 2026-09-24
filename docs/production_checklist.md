# Production Checklist

## Before scoring real customers

- Place the approved customer data in `data/raw/`.
- Review schema and data-quality expectations.
- Run the production refit script.
- Record the generated model metadata/version.
- Score the customer population.
- Load customer scores into MySQL.
- Validate risk and retention distributions.
- Start Streamlit.

## Monitoring to add later

- feature distribution drift
- probability calibration drift
- missingness drift
- score distribution drift
- intervention response / uplift measurement
- model retraining schedule
