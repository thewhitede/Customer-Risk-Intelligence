# Customer Risk Intelligence Platform

An explainable, revenue-aware telecom customer churn intelligence platform that turns calibrated churn probabilities into customer risk segments, expected revenue exposure, and retention priorities.

The platform combines traditional machine learning, probability calibration, explainability, customer segmentation, revenue analysis, database integration, and an interactive dashboard. Optional Deep Learning and NLP modules extend the research without replacing the established production scoring workflow.

1. Project Overview

Customer Risk Intelligence helps identify customers who are more likely to churn and translates model predictions into actionable business information.

The platform is designed to:

Estimate customer churn probabilities.

Classify customers into Low, Medium, and High risk segments.

Explain individual predictions using SHAP.

Estimate expected monthly revenue exposure.

Prioritize customers for retention activities.

Store customer and risk information in MySQL.

Present results through an interactive Streamlit dashboard.

Experiment with Deep Learning and NLP as optional extensions.

2. Project Status

The core research analysis is complete in the four main analytical notebooks. The repository productionizes that work while keeping the established research evaluation separate from deployment-oriented model refitting.

Core research notebooks

Notebook

Purpose

01_data_audit_eda.ipynb

Data audit and exploratory data analysis

02_baseline_ml_shap.ipynb

Baseline machine learning and SHAP explainability

03_research_model_calibration.ipynb

Research model, probability calibration, and risk analysis

04_risk_revenue_retention.ipynb

Revenue exposure and retention prioritization

The archived consolidation notebook is retained separately:

archive/05_master_research_results.ipynb

Optional extension notebooks

Notebook

Purpose

05_deep_learning_model_comparison.ipynb

Structured Deep Learning model comparison

06_nlp_customer_voice.ipynb

Customer-support sentiment and topic analysis

Power BI is maintained separately and is intentionally not included in this repository.

3. Established Research Results

The following values represent the established research evaluation from the original analytical workflow.

Random Forest performance

Metric

Result

Accuracy

0.7686

Precision

0.5548

Recall

0.6497

F1-score

0.5985

ROC-AUC

0.8208

PR-AUC

0.6069

Probability calibration

Calibration improved the probability quality of the research model.

Metric

Before calibration

After calibration

Brier score

0.1562

0.1431

ROC-AUC

0.8208

0.8276

Log Loss

0.4768

0.4397

Risk segment validation

Risk segment

Customers

Observed churn

Low

917

11.78%

Medium

307

44.63%

High

185

69.73%

The observed churn rate increases across the risk segments, supporting their usefulness for customer prioritization.

Revenue exposure

Total expected monthly revenue at risk: $27,257.11

This figure represents estimated revenue exposure associated with customers identified as being at risk. It is not a guarantee that this revenue will be lost.

These research values remain the established reference results. They should not be silently replaced with metrics from production refitting or later exploratory experiments.

4. System Architecture

Core production workflow

Telco Customer Churn CSV
          |
          v
Python Data Processing
          |
          v
Feature Engineering
          |
          v
Random Forest Model
          |
          v
Sigmoid Probability Calibration
          |
          v
Customer Churn Probabilities
          |
          v
Risk Segmentation
          |
          v
Revenue-at-Risk Analysis
          |
          v
Retention Prioritization
          |
          v
MySQL Database
          |
          v
Streamlit Dashboard
          |
          v
Optional GenAI Explanations

SHAP operates as an explainability layer over the Random Forest. It helps explain individual predictions without changing the underlying churn score.

Optional research extensions

The Deep Learning and NLP modules are separate experimental components.

Structured Customer Data
          |
          v
PyTorch MLP
          |
          v
Comparison with Random Forest


Customer Support Text
          |
          v
Transformer Sentiment Analysis
          |
          v
TF-IDF + NMF Topic Discovery
          |
          v
Aggregated Customer Text Features
          |
          v
Optional Hybrid MLP

The hybrid workflow requires valid customer-linked text features. Synthetic demonstration data is not evidence of real-world churn improvement.

5. Technologies Used

Area

Technologies

Programming

Python

Data manipulation

Pandas, NumPy

Notebook visualization

Matplotlib, Seaborn

Dashboard visualization

Altair, Streamlit

Machine Learning

Scikit-learn

Deep Learning

PyTorch

NLP

Hugging Face Transformers

Text feature extraction

TF-IDF, NMF

Model explainability

SHAP

Database

MySQL 8.4

Development

Jupyter Notebook, VS Code, Git, GitHub

6. Dataset and Data Sources

IBM Telco Customer Churn

The structured churn experiments use the public IBM Telco Customer Churn benchmark.

7,043 customer records.

21 original columns.

Target variable: Churn.

Non-churn customers: 5,174.

Churn customers: 1,869.

The original class distribution is approximately:

Class

Records

Percentage

No churn

5,174

73.46%

Churn

1,869

26.54%

Place the dataset at:

data/raw/telco_customer_churn.csv

The repository includes a downloader that validates the public IBM benchmark source and dataset schema before saving the file.

python -m scripts.download_telco_dataset

Synthetic demonstration data

The project also includes synthetic datasets under data/demo/:

telco_customer_churn_demo.csv

customer_support_text_demo.csv

customer_text_features_demo.csv

These files allow users to test the workflow without relying on actual customer-support records.

Important: Synthetic demonstration records are not real customer data. They must not be presented as actual customer evidence or used to support research-performance claims.

7. Production Model

The production training script is:

scripts/train_production_model.py

It creates a deployment-oriented refit using the established model architecture:

Random Forest with 300 trees.

Balanced class weighting.

Random state of 42.

Sigmoid probability calibration.

Five-fold calibration cross-validation.

The deployment model is separate from the original held-out research evaluation.

The production artifact is generated locally and is not supplied as a pre-trained model binary in the repository.

Run production training

python -m scripts.train_production_model

This generates the deployment model artifact and associated metadata.

Generate customer risk scores

python -m scripts.score_customers

The scoring workflow creates:

data/processed/customer_risk_scores.csv

Run production training before scoring on a fresh installation.

8. Deep Learning Extension

The optional Deep Learning module is implemented in:

src/deep_learning.py

It uses a PyTorch Multilayer Perceptron (MLP) for structured customer churn prediction.

The module includes:

Training and validation workflows.

Training-only preprocessing.

Early stopping.

Model evaluation.

Comparison with the calibrated Random Forest on a shared stratified hold-out split.

Structured model comparison

In one exploratory evaluation on the public IBM Telco benchmark, the models produced the following results:

Metric

Calibrated Random Forest

Deep Learning MLP

Accuracy

0.7786

0.7999

Precision

0.5923

0.6494

Recall

0.5321

0.5348

F1-score

0.5606

0.5865

ROC-AUC

0.8293

0.8435

PR-AUC

0.6190

0.6447

Brier score

0.1427

0.1375

The MLP performed better on these metrics in this particular evaluation. This is an exploratory result from one hold-out evaluation, not proof of general superiority.

The established production scorer remains the calibrated Random Forest. The MLP does not automatically replace it.

Train the MLP on the IBM benchmark

python -m scripts.train_deep_learning_model --data data/raw/telco_customer_churn.csv --output-dir models/deep_learning

Run the demonstration experiment

python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv --output-dir models/deep_learning

Demo-data metrics are for workflow testing only and must not be treated as research results.

9. NLP Extension

The NLP functionality is implemented in:

src/nlp.py

It extends the project with customer-support text analysis.

Transformer sentiment analysis

Uses a pretrained Hugging Face sentiment model:

distilbert/distilbert-base-uncased-finetuned-sst-2-english

It classifies text sentiment as positive or negative and provides a model confidence score.

This is a general English sentiment model, not a telecom-specific churn model. Its sentiment confidence is not a customer churn probability.

TF-IDF and NMF topic discovery

Uses:

TF-IDF to represent text as numerical features.

Non-negative Matrix Factorization (NMF) to identify exploratory themes in customer messages.

Customer-level text features

The NLP workflow can aggregate ticket-level information into customer-level features for an optional hybrid experiment.

Run NLP analysis on the synthetic demo tickets

python -m scripts.analyze_support_text --input data/demo/customer_support_text_demo.csv --output-dir data/processed/customer_voice_demo --as-of-date 2026-09-30

The script generates:

data/processed/customer_voice_demo/customer_tickets_scored.csv
data/processed/customer_voice_demo/customer_text_features.csv

The first run downloads the pretrained Transformer checkpoint if it is not already available locally.

Optional hybrid MLP demonstration

Use the matching synthetic customer records and customer-level text features generated from the same synthetic cohort:

python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv --text-features data/processed/customer_voice_demo/customer_text_features.csv --output-dir models/deep_learning_demo

The customer dataset and text features must come from the same cohort so their customer IDs can be joined correctly.

Offline hybrid demonstration

The repository also contains precomputed synthetic text features. These allow the hybrid workflow to be tested without running the Transformer first:

python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv --text-features data/demo/customer_text_features_demo.csv --output-dir models/deep_learning_demo

These precomputed features are synthetic demonstration data. They should not be described as new predictions from the Transformer.

NLP evaluation using manually labelled messages

The sentiment evaluation script compares Transformer predictions against manually labelled customer-support messages.

Prepare a CSV file containing:

Column

Description

ticket_text

Original customer-support message

true_sentiment

Manually assigned positive or negative label

The script requires at least 10 valid, manually labelled examples. More examples are recommended for a meaningful evaluation.

After preparing the file, run:

python -m scripts.evaluate_nlp_sentiment --input data/validated/manually_labelled_support_text.csv

The example path is a destination for a dataset you prepare. It is not an included repository file.

The evaluation report is saved by default to:

results/nlp_sentiment_evaluation.json

Important: The synthetic true_sentiment values in the demonstration dataset are not human annotated. Do not use them to claim genuine sentiment-model accuracy.

Advanced Customer Voice Risk Analysis

The project includes an exploratory customer voice risk analysis script:

python -m scripts.analyze_customer_voice_risk

This is an optional advanced analysis. It requires locally generated inputs, including:

data/raw/telco_customer_risk_and_voice_master.csv
data/processed/customer_voice_master/customer_text_features.csv
models/calibrated_research_rf.joblib

Its output is saved under:

data/processed/customer_voice_risk/

The analysis uses synthetic customer and support-ticket data. Its correlations and group comparisons are descriptive only and do not demonstrate that NLP improves churn prediction or retention.

It does not retrain or modify the production model.

10. MySQL Database

The project uses MySQL 8.4 to persist customer information and production risk scores.

Database name:

customer_risk_intelligence

Database setup

Run the SQL scripts using MySQL Workbench or the MySQL client, in this order:

sql/01_schema.sql
sql/03_views.sql
sql/04_research_summary.sql

Core tables

customers
customer_risk_scores
retention_actions
research_summary

The database also contains reporting views for customer risk and retention-priority summaries.

Load customer records

python -m scripts.load_customers_to_mysql

The customer loader defaults to the MySQL root account and securely prompts for its password. It also supports a dedicated application user when that user has already been created and granted the required permissions.

Configure MySQL environment variables

The score loader and Streamlit MySQL connection read their configuration from operating-system environment variables.

For PowerShell, configure the values in the active terminal:

$env:MYSQL_HOST = "localhost"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "cri_user"
$env:MYSQL_PASSWORD = "YOUR_PASSWORD"
$env:MYSQL_DATABASE = "customer_risk_intelligence"

Replace the password placeholder with your local database password. Do not commit passwords or other database credentials to GitHub.

These environment variables apply to the current PowerShell session. Configure them again in a new session or use your own secure local environment setup.

Load production risk scores

Run scoring before loading the risk scores:

python -m scripts.score_customers

python -m scripts.load_scores_to_mysql

Streamlit data sources

The application supports two scoring data sources:

CSV: Loads customer data and scores it using the production calibrated Random Forest artifact.

MySQL: Loads persisted customer risk scores from MySQL without rescoring the customers.

The application should not require the MySQL root account during normal operation. A dedicated application user such as cri_user is recommended.

11. Streamlit Dashboard

Run the application from the project root:

streamlit run app/streamlit_app.py

The dashboard provides six sections:

Dashboard section

Purpose

Risk Overview

Overall churn risk and customer distribution

Customer Scoring

Customer-level risk scores and explanations

Retention Command Center

Retention priorities and revenue exposure

Research Evidence

Established research metrics and validation

Model Comparison

Structured Random Forest and MLP comparison

Customer Voice (NLP)

Sentiment and customer-support text analysis

The application supports CSV and MySQL scoring data sources. CSV mode requires the locally generated production model artifact. MySQL mode requires persisted customer scores and a working database connection.

The customer voice and demonstration sections distinguish synthetic demo information from actual customer evidence.

12. GenAI and Hugging Face Explanation Layer

The GenAI explanation layer is optional.

Set the following environment variables to enable model-generated business explanations:

HF_TOKEN
HF_MODEL

Without these settings, the application uses a deterministic explanation template.

The explanation layer sits on top of the existing risk and retention logic. It does not change the underlying churn prediction.

13. Installation and Setup

Requirements

Python 3.12 recommended for the current development environment.

Git.

MySQL 8.4 for database functionality.

Internet access for downloading the public benchmark or pretrained Transformer checkpoint.

Clone the repository

Open PowerShell in a user-writable location, such as your home directory:

cd $HOME

git clone https://github.com/thewhitede/Customer-Risk-Intelligence.git

cd Customer-Risk-Intelligence

Do not clone the repository inside protected Windows system directories such as C:\Windows.

Create a virtual environment

python -m venv .venv

Activate the environment

.\.venv\Scripts\Activate.ps1

If PowerShell blocks script activation, use the appropriate execution policy for your local account or activate the environment through the VS Code terminal.

Install dependencies

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

Optional notebook dependencies

For working with Jupyter notebooks and the EDA visualizations, install:

python -m pip install jupyterlab matplotlib seaborn

Launch JupyterLab with:

jupyter lab

Download the benchmark dataset

python -m scripts.download_telco_dataset

Run the production workflow

python -m scripts.train_production_model

python -m scripts.score_customers

streamlit run app/streamlit_app.py

For MySQL functionality, configure the database and credentials before running the database loader scripts.

For the detailed deployment procedure, see docs/deployment.md.

For add-on setup, data schemas, and experiment safeguards, see docs/ADDON_INTEGRATION.md.

For the system design, see docs/architecture.md.

14. Testing

Run the project test suite from the repository root with the virtual environment activated:

python -m pytest -q

The latest reported run completed with:

12 passed, 8 warnings

The warnings were Joblib/NumPy deprecation warnings and did not cause test failures.

Passing software tests confirms that the tested code paths work as expected. It does not establish predictive performance or prove that the model will generalize to new customer populations.

15. Research Limitations and Responsible Use

Revenue at Risk is an expected exposure measure, not guaranteed lost revenue.

Historical churn capture does not prove that a retention intervention prevents churn.

Retention recommendations are decision-support heuristics, not causal treatment effects.

The public IBM Telco dataset is a static benchmark, not a live customer feed.

The structured MLP comparison is exploratory and based on a hold-out evaluation.

Synthetic support messages are demonstration data, not real customer evidence.

The sentiment model is a general English model and is not trained specifically to predict telecom churn.

The hybrid NLP model requires genuine, properly licensed or consented, customer-linked text data and time-safe joins before its performance can be meaningfully evaluated.

Synthetic demo results must not be presented as production results or evidence of real-world model improvement.

Production use requires independent evaluation, periodic retraining, monitoring, and appropriate data governance.

16. Project Structure

The following is a high-level view of the repository. Generated model artifacts and processed data may be created locally when you run the workflows.

Customer-Risk-Intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── archive/
│   └── 05_master_research_results.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── demo/
│
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   └── ADDON_INTEGRATION.md
│
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_baseline_ml_shap.ipynb
│   ├── 03_research_model_calibration.ipynb
│   ├── 04_risk_revenue_retention.ipynb
│   ├── 05_deep_learning_model_comparison.ipynb
│   └── 06_nlp_customer_voice.ipynb
│
├── scripts/
│
├── sql/
│
├── src/
│   ├── deep_learning.py
│   ├── nlp.py
│   └── ...
│
├── tests/
│
├── requirements.txt
└── README.md

17. Project Objective

The objective of Customer Risk Intelligence is to demonstrate how data science can move beyond predicting customer churn.

By combining calibrated machine learning predictions, explainability, revenue exposure analysis, retention prioritization, database integration, and an interactive dashboard, the project provides a structured approach to turning customer data into business decision support.

The optional Deep Learning and NLP extensions provide additional experimental capabilities while preserving the original research evidence and production scoring architecture.