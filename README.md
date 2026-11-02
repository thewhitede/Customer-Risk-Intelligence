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

For the public IBM Telco benchmark, the add-on includes a downloader that verifies the source checksum and schema before saving the CSV:

```powershell
python -m scripts.download_telco_dataset
```

For a quick end-to-end run without internet access, use the fully synthetic dataset under `data/demo/` instead. See `data/README.md` and `data/demo/README.md` for source details and caveats.

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

## Optional Deep Learning + NLP add-on

This repository now includes an optional add-on that leaves the existing calibrated Random Forest production scorer and established research metrics unchanged.

- `src/deep_learning.py` trains a PyTorch MLP and can compare it against the calibrated Random Forest on one shared stratified hold-out split.
- `src/nlp.py` scores support text with a pretrained Transformer and discovers exploratory complaint topics with TF-IDF + NMF.
- `scripts/generate_demo_data.py` generates linked synthetic demo data. Three generated CSVs are included under `data/demo/`.
- `scripts/download_telco_dataset.py` downloads and validates the public IBM benchmark for the structured churn experiment.
- `data/README.md` documents source attribution, data access, and the limits of the synthetic linked text data.
- `notebooks/05_deep_learning_model_comparison.ipynb` and `notebooks/06_nlp_customer_voice.ipynb` document the experiments.
- `docs/ADDON_INTEGRATION.md` explains setup, commands, data schemas, and safeguards.

Install the optional dependencies in the active virtual environment:

```powershell
python -m pip install -r requirements-addon.txt
```

Train and compare the structured models on the included demo data:

```powershell
python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv --output-dir models/deep_learning
```

Run NLP sentiment and topic analysis on the included demo messages (the first run downloads the Transformer checkpoint):

```powershell
python -m scripts.analyze_support_text --input data/demo/customer_support_text_demo.csv --output-dir data/processed/customer_voice
```

Then pass the NLP-produced customer feature file to the same training command with `--text-features data/processed/customer_voice/customer_text_features.csv` to compare the structured and hybrid MLPs. The dashboard's **Model Comparison** and **Customer Voice (NLP)** tabs read these outputs. If you have not run the Transformer, the customer-voice tab displays the included precomputed synthetic demo features and clearly labels them as such.

**Demo-data warning:** all data under `data/demo/` is synthetic. It is provided to exercise the workflow, not as actual customer data or valid evidence of predictive or sentiment performance. The synthetic `true_sentiment` label is not human annotated. Do not report demo metrics in a research paper or CV as production results. For research claims, use properly licensed/consented customer data, manually labelled support text, time-safe joins, and independently held-out evaluations.

## Research caveats

- Revenue at Risk is an expected exposure measure, not guaranteed lost revenue.
- Historical churn capture does not prove that an intervention prevents churn.
- Retention recommendations are decision-support heuristics, not causal treatment effects.
- The Telco dataset is a static snapshot, so production deployment requires periodic retraining and monitoring.
