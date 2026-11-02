# Deep Learning + NLP Add-on Integration Guide

This package is an **add-on only** for the existing Customer Risk Intelligence Platform. It does not replace `src/modeling.py`, the calibrated Random Forest production scorer, established research notebooks, MySQL logic, or the current research paper.

## 1. Copy the add-on into your existing repository

Extract the add-on ZIP at the root of your existing `Customer-Risk-Intelligence` repository, allowing it to merge the new files into `src/`, `scripts/`, `tests/`, `notebooks/`, and `docs/`. Do not delete or replace the existing project files.

Then install the extra dependencies inside the project's existing virtual environment:

```powershell
python -m pip install -r requirements-addon.txt
```

PyTorch installation may need a platform-specific wheel. If the command fails for `torch`, use the official selector at https://pytorch.org/get-started/locally/ and then install `transformers`.

## 2. New files

- `src/deep_learning.py` — PyTorch MLP training, prediction, shared-split evaluation, and optional hybrid text-feature support.
- `src/nlp.py` — batched Transformer sentiment scoring, customer-level aggregation, and exploratory TF-IDF + NMF topic discovery.
- `scripts/train_deep_learning_model.py` — CLI for model training and comparison.
- `scripts/analyze_support_text.py` — CLI for support-message analysis and aggregate customer features.
- `scripts/evaluate_nlp_sentiment.py` — evaluation against a manually labelled support-text sample.
- `notebooks/05_deep_learning_model_comparison.ipynb` and `notebooks/06_nlp_customer_voice.ipynb` — guided experiments.
- `data/demo/` — linked, fully synthetic customer and support-ticket data, plus synthetic precomputed aggregates for offline hybrid-pipeline smoke tests.
- `scripts/generate_demo_data.py` — deterministic generator for the demo CSVs.
- `scripts/download_telco_dataset.py` — downloads and checksum-validates the public IBM benchmark to `data/raw/telco_customer_churn.csv`.
- `data/README.md` — source attribution, raw-data download steps, and limitations.
- `data/demo/README.md` — schemas, limitations, and reproducibility notes.
- `tests/test_deep_learning_addon.py`, `tests/test_nlp_addon.py`, `tests/test_addon_source_imports.py`, and `tests/test_demo_data_addon.py`, and `tests/test_download_telco_dataset_addon.py` — regression, data-contract, and downloader-validation checks.
- `patches/streamlit_addon.patch` — small optional patch that adds Model Comparison and Customer Voice tabs to the existing Streamlit app, without replacing it.

## 3. Run the tests

From the repository root:

```powershell
python -m pytest tests/test_deep_learning_addon.py tests/test_nlp_addon.py tests/test_addon_source_imports.py -q
```

The MLP round-trip test uses generated test fixtures, not real customer data. It verifies that training, artifact serialization, and inference work; it is not a performance claim.

## 4. Run the Deep Learning model

### Included demo data (works without an external dataset)

This add-on includes reproducible, linked synthetic data in `data/demo/`:

- `telco_customer_churn_demo.csv` — 1,200 synthetic customer records using the expected Telco schema.
- `customer_support_text_demo.csv` — synthetic ticket messages linked by `customerID`; all ticket dates precede the simulated 2026-10-01 churn snapshot.
- `customer_text_features_demo.csv` — synthetic aggregate features for an offline smoke-test of the hybrid join, not Transformer output.

To train the MLP and compare it with the calibrated Random Forest on a fresh shared split:

```powershell
python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv
```

The included dataset is synthetic. Use the results to verify code and learn the workflow only. Do not report them as evidence of real-world performance.

### Your real labeled dataset

For a real benchmark instead of the synthetic demo, download the public IBM sample dataset:

```powershell
python -m scripts.download_telco_dataset
python -m scripts.train_deep_learning_model --data data/raw/telco_customer_churn.csv
```

The downloader fetches the public CSV from IBM's repository and verifies its expected checksum, 7,043 rows, and 21-column schema before saving it. The raw file is excluded by `.gitignore`. Read `data/README.md` for attribution and use notes. Alternatively, pass a path to your own appropriately licensed/consented labeled churn CSV through `--data PATH`.

The script trains a PyTorch MLP. By default it also evaluates the existing calibrated Random Forest architecture on the **same new stratified train/validation/test split**, so the new comparison is fair. It saves the MLP model/preprocessor under `models/deep_learning/structured/` and writes `models/deep_learning/comparison_metrics.json`.

The default split uses roughly 64% training, 16% validation, and 20% test rows. Preprocessing is fitted on training rows only. Validation loss drives early stopping, then a logistic/Platt probability calibrator is fitted on validation logits; the test split is held out for reporting. The 0.50 threshold is a transparent default, not necessarily the best business threshold. Tune decision thresholds on validation data before any deployment decision.

To run a quick development experiment:

```powershell
python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv --epochs 10 --patience 3
```

Development runs must not be presented as final benchmark results. The add-on writes new metrics and does not overwrite the established metrics in `src/research_summary.py` or the current research paper.

## 5. Run the NLP pipeline

### Try the included synthetic text dataset

```powershell
python -m scripts.analyze_support_text --input data/demo/customer_support_text_demo.csv --output-dir data/processed/customer_voice
```

The first run downloads the pretrained Transformer model from Hugging Face and may require network access and several hundred megabytes of storage. The synthetic `true_sentiment` column is for demonstration only; it is not a human-labelled evaluation set. The precomputed synthetic feature file can be used to test the hybrid-model join without downloading a Transformer:

```powershell
python -m scripts.train_deep_learning_model --data data/demo/telco_customer_churn_demo.csv --text-features data/demo/customer_text_features_demo.csv
```

This second command is only a smoke test of the feature-join and hybrid-model pipeline. Its text features are simulated and must not be presented as results from NLP.

### Use an actual customer-text dataset

Prepare a separate CSV at `data/raw/customer_support_text.csv` with at least:

| Column | Required? | Meaning |
|---|---|---|
| `customerID` | Yes | Stable customer key matching the churn table |
| `ticket_text` | Yes | Customer-written complaint, feedback, or support message |
| `ticket_date` | Strongly recommended | Time the message was created |

Example row (synthetic):

```csv
customerID,ticket_date,ticket_text
C0001,2026-01-12,"I was charged twice and support has not resolved it."
```

Do not use generated examples as evidence. Use licensed/public or appropriately consented data, and remove or protect personal and sensitive information before analysis. Do not commit raw ticket data or model artifacts to GitHub.

Then run the CLI with your actual file:

```powershell
python -m scripts.analyze_support_text --input data/raw/customer_support_text.csv
```

The first run downloads the pretrained model from Hugging Face and may require network access and several hundred megabytes of storage. Outputs are written to `data/processed/customer_voice/`. `customer_text_features.csv` contains one row per customer and the numeric columns required for the optional hybrid MLP.

The default model is `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, an English binary sentiment checkpoint trained on SST-2. It is a **starting point**, not a telecom-specific model. Its probabilities are model confidence scores, not churn probabilities. Validate it using manually labelled customer-support samples with `scripts/evaluate_nlp_sentiment.py` before operational use. Complaint topics from NMF are exploratory and must be reviewed by a person.

For a time-bounded feature export, set a cutoff:

```powershell
python -m scripts.analyze_support_text --as-of-date 2026-06-30
```

The script filters tickets after the inclusive cutoff date before aggregating features. Choose the cutoff to match the actual prediction date in your labelled churn dataset.

## 6. Train the optional hybrid MLP

Once you have a valid, linked, time-safe customer text feature file generated by the real NLP pipeline (not the synthetic demo aggregates):

```powershell
python -m scripts.train_deep_learning_model --data data/raw/telco_customer_churn.csv --text-features data/processed/customer_voice/customer_text_features.csv
```

The comparison then includes a structured-only MLP and a hybrid MLP with aggregated NLP features, using the same customer split. A structured Random Forest is still included as the baseline. This is the only appropriate time to make a claim about whether the added text features help. If the linked data, date cutoff, or sample size is not defensible, keep NLP as a separate customer-voice analysis instead of claiming an NLP-driven churn lift.

If you use the hybrid MLP at inference, pass the customer-level NLP feature DataFrame to `predict_deep_learning(..., model_dir="models/deep_learning/hybrid", text_features=features_df)`. The app's existing production scorer remains the original calibrated Random Forest unless you explicitly design, test, and approve a separate deployment path.

## 7. Optional Streamlit integration

From the repository root, inspect and apply the patch:

```powershell
git apply --check patches/streamlit_addon.patch
git apply patches/streamlit_addon.patch
```

If your app has changed since this add-on was made, apply the equivalent changes manually instead of forcing the patch. The patch adds two read-only views:

- **Model Comparison** reads the saved shared-split metrics and plots ROC / precision-recall curves.
- **Customer Voice (NLP)** reads customer aggregates and displays support-volume / negative-sentiment summaries.

The dashboard does not train models or download Transformer weights at app startup. Run the training/analysis scripts first.

## 8. Research and leakage safeguards

- Keep existing reported Random Forest and calibration metrics unchanged. The add-on creates a separate, fresh comparison experiment.
- Use the same customer split and held-out labels to compare the structured RF and structured MLP. The MLP probability calibrator is fitted on validation data, not the held-out test set. The hybrid MLP is assessed on that same row split when text features are available.
- Fit imputers, scalers, and encoders on training rows only.
- Ensure every text feature predates the relevant prediction cutoff. A support message created after the churn outcome is leakage if it is used to predict that earlier outcome.
- A higher ROC-AUC alone is not enough. Review PR-AUC, recall, precision, F1, Brier score, log loss, calibration, and intervention costs.
- Model-generated sentiment can be wrong or biased. Do not use it as the sole basis for decisions about individual customers.
- Never invent benchmark metrics in the README, CV, or paper. Report results generated from the actual dataset, test split, and reproducible configuration.
