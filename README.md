# Customer Risk Intelligence Platform

An explainable, revenue-aware telecom customer churn intelligence platform that turns calibrated churn probabilities into risk segments, expected revenue exposure, and retention priorities.

## Project status

The core research analysis is already complete in the four analytical notebooks. This repository layer productionizes that work without replacing the established research evidence.

- `01_data_audit_eda.ipynb` — data audit + EDA
- `02_baseline_ml_shap.ipynb` — baseline ML + SHAP
- `03_research_model_calibration.ipynb` — research model + calibration + risk
- `04_risk_revenue_retention.ipynb` — revenue risk + retention
- `05_master_research_results.ipynb` — auxiliary consolidation
- Power BI is maintained separately and is intentionally not included here.

## Established research evidence

Research Random Forest:

- Accuracy: 0.7686
- Precision: 0.5548
- Recall: 0.6497
- F1: 0.5985
- ROC-AUC: 0.8208
- PR-AUC: 0.6069

Calibration:

- Brier: 0.1562 → 0.1431
- ROC-AUC: 0.8208 → 0.8276
- Log Loss: 0.4768 → 0.4397

Risk validation:

- Low: 917 customers, 11.78% observed churn
- Medium: 307 customers, 44.63%
- High: 185 customers, 69.73%

Revenue exposure:

- Total expected monthly revenue at risk: $27,257.11

These values are research evidence from the established notebooks; they should not be silently replaced by production-refit metrics.

## Architecture

```text
Telco Customer Churn CSV
        ↓
Python feature processing
        ↓
Random Forest
        ↓
Sigmoid probability calibration
        ↓
Risk segmentation
        ↓
Revenue-at-Risk
        ↓
Retention priority
        ↓
MySQL persistence
        ↓
Streamlit
        ↓
Optional Hugging Face / GenAI explanation layer
```

SHAP is maintained as an explainability layer over the underlying Random Forest rather than as a mandatory step in the prediction pipeline. It helps explain individual predictions without changing the underlying churn score.

## Productionization

The `scripts/train_production_model.py` script creates a deployment-oriented refit using the same research architecture (300-tree balanced Random Forest, random state 42, sigmoid calibration with 5-fold CV). The deployment artifact is intentionally separate from the held-out research evaluation.

Place the Telco Customer Churn dataset at:

`data/raw/telco_customer_churn.csv`

Then run from the project root:

```powershell
python -m scripts.train_production_model

python -m scripts.score_customers

python -m scripts.load_customers_to_mysql

python -m scripts.load_scores_to_mysql

streamlit run app/streamlit_app.py
```
The complete deployment workflow is documented in the [Deployment Guide](docs/deployment.md).
The production model is a deployment-oriented refit. Its predictions are kept separate from the frozen held-out research point estimates reported above.

## MySQL

The project uses MySQL 8.4 for persistence of customer records and production risk scores.

The validated database is:

```text
customer_risk_intelligence
```

### Database setup

Run the following SQL scripts using MySQL Workbench or the MySQL client:

```text
sql/01_schema.sql
sql/03_views.sql
sql/04_research_summary.sql
```

The project-native Python loader is used to populate customer records:

```powershell
python -m scripts.load_customers_to_mysql
```

After generating the production scoring file, load the persisted risk scores:

```powershell
python -m scripts.load_scores_to_mysql
```

### Core tables

```text
customers
customer_risk_scores
retention_actions
research_summary
```

The database also contains reporting views for customer risk and retention-priority summaries.

### Streamlit data sources

The Streamlit application supports two scoring data sources:

- **CSV:** Loads customer data and scores it using the production calibrated Random Forest artifact.
- **MySQL:** Loads persisted customer risk scores from MySQL without rescoring the customers.

The Streamlit application uses the dedicated MySQL application user:

```text
cri_user
```

The application should not require the MySQL `root` account during normal operation.

Database credentials must remain outside version control.
## GenAI / Hugging Face

The GenAI layer is optional. Set `HF_TOKEN` and `HF_MODEL` in the environment to enable model-generated business explanations. Without them, the app falls back to a deterministic, business-safe explanation template.

The GenAI layer does not change the churn prediction itself and is designed to sit on top of the deterministic risk and retention logic.

## Research caveats

- Revenue at Risk is an expected exposure measure, not guaranteed lost revenue.
- Historical churn capture does not prove that an intervention prevents churn.
- Retention recommendations are decision-support heuristics, not causal treatment effects.
- The Telco dataset is a static snapshot, so production deployment requires periodic retraining and monitoring.
