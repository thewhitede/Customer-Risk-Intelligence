# Customer Risk Intelligence — Deep Learning + NLP Add-on

This is a drop-in extension, **not a rebuilt copy of the original project**.

The package includes linked, fully synthetic demo data under `data/demo/`, `scripts/generate_demo_data.py` to reproduce it, and `scripts/download_telco_dataset.py` to fetch and validate the public IBM Telco benchmark when you have internet access. Synthetic data is for smoke tests and learning only, not research claims.

It adds:

1. A PyTorch multilayer perceptron (MLP) for customer churn, with early stopping and training-only preprocessing.
2. Same-split comparison against the existing calibrated Random Forest architecture.
3. Transformer-based customer-support sentiment scoring.
4. Exploratory complaint-topic discovery using TF-IDF + non-negative matrix factorization (NMF).
5. Optional hybrid MLP that adds aggregated NLP features to structured customer features when valid linked text data is available.
6. Evaluation scripts, notebooks, tests, and an optional Streamlit dashboard patch.

Start with [`docs/ADDON_INTEGRATION.md`](docs/ADDON_INTEGRATION.md). It explains where to copy files, how to install the extra dependencies, how to use the included demo datasets, how to run the modules, and safeguards required for a professional evaluation.

**Important:** the uploaded project ZIP did not include the raw churn CSV or support-ticket text, so this add-on has not been benchmarked on the user's real customer data. The tests use generated fixture data to check that the pipeline works; those test outputs are not project performance results.
