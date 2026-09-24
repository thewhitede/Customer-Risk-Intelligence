# Build Status — Everything Outside Power BI

## Built in this package

- MySQL schema and load scripts
- Research summary SQL seed
- Production-oriented calibrated Random Forest refit script
- Feature engineering module matching the research design
- Risk/revenue/retention logic
- Customer scoring service
- Streamlit application
- Optional Hugging Face explanation layer with deterministic fallback
- Dockerfile + docker-compose
- Unit tests
- README and architecture/deployment/data dictionary documentation
- Research notes and full research-paper draft
- Portfolio/resume copy
- Interview Q&A

## Deliberately not included

- Power BI implementation (user is maintaining it separately)
- A fabricated pre-trained model binary (the project currently does not contain the original binary artifact)
- The raw Telco CSV as a repository asset

## Important distinction

The production training script uses the same established model architecture but creates a deployment-oriented refit. The research metrics in the notebooks remain the authoritative held-out evidence.
