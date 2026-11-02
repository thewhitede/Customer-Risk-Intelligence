# Data sources and usage

## Included synthetic demo files

The `data/demo/` directory includes a small, linked and fully synthetic dataset so the add-on's code paths can be exercised immediately:

- `telco_customer_churn_demo.csv` — synthetic structured customer data.
- `customer_support_text_demo.csv` — synthetic support messages, linked by `customerID`.
- `customer_text_features_demo.csv` — synthetic precomputed sentiment aggregates for offline hybrid-pipeline smoke tests.

These files are not real customers or real support records. The text labels are generated from message templates rather than manual annotation. They must not be used as real-world model validation or cited as empirical project results. See `data/demo/README.md` for details.

## Public IBM Telco churn benchmark

When internet access is available, run from the repository root:

```powershell
python -m scripts.download_telco_dataset
```

The script fetches the public IBM Telco Customer Churn sample data from the official IBM GitHub repository and verifies its expected SHA-256 before saving it to `data/raw/telco_customer_churn.csv`. The expected snapshot is 7,043 customer rows with 21 columns. Source and schema notes are documented in the [IBM repository](https://github.com/IBM/telco-customer-churn-on-icp4d) and the download URL is included in the downloader. The IBM repository lists an Apache-2.0 license. Review the source terms for the use you intend before distributing derivative work.

The raw CSV is ignored by `.gitignore` under `data/raw/*.csv` so it is not accidentally committed to Git. After downloading it, train with:

```powershell
python -m scripts.train_deep_learning_model --data data/raw/telco_customer_churn.csv
```

## NLP data limitation

The IBM churn benchmark contains no customer-written support messages. You cannot validly join its customers to the synthetic support text. Use the synthetic linked cohort to test the optional hybrid path, or provide a properly sourced, consented and temporally aligned text dataset before making any claim that NLP improves churn prediction.
