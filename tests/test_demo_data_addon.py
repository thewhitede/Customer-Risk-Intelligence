from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def test_demo_datasets_are_linked_and_time_safe():
    customers = pd.read_csv(DEMO / "telco_customer_churn_demo.csv")
    tickets = pd.read_csv(DEMO / "customer_support_text_demo.csv")
    features = pd.read_csv(DEMO / "customer_text_features_demo.csv")

    expected_customer_columns = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    }
    assert expected_customer_columns.issubset(customers.columns)
    assert customers["customerID"].is_unique
    assert set(customers["Churn"].dropna().unique()) == {"Yes", "No"}
    assert {"customerID", "ticket_date", "ticket_text", "true_sentiment"}.issubset(tickets.columns)
    assert tickets["ticket_text"].notna().all()
    assert set(tickets["true_sentiment"].unique()) == {"positive", "negative"}
    assert pd.to_datetime(tickets["ticket_date"]).max() < pd.Timestamp("2026-10-01")
    assert set(tickets["customerID"]).issubset(set(customers["customerID"]))
    assert set(features["customerID"]).issubset(set(customers["customerID"]))
    assert features["customerID"].is_unique
    assert {
        "ticket_count", "negative_ticket_count", "negative_ticket_ratio",
        "mean_sentiment_score", "latest_sentiment_score",
    }.issubset(features.columns)


def test_demo_data_is_explicitly_marked_as_synthetic():
    readme = (DEMO / "README.md").read_text(encoding="utf-8").lower()
    tickets = pd.read_csv(DEMO / "customer_support_text_demo.csv", nrows=1)
    assert "synthetic" in readme
    assert "not human" in readme or "not a human" in readme
    assert "demo_note" in tickets.columns
