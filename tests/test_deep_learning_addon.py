import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.deep_learning import TrainingConfig, predict_deep_learning, train_from_csv


def make_telco_sample(n=150, seed=42):
    rng = np.random.default_rng(seed)
    yes_no = np.array(["Yes", "No"])
    internet_options = np.array(["DSL", "Fiber optic", "No"])
    contract_options = np.array(["Month-to-month", "One year", "Two year"])
    payment_options = np.array([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    data = {
        "customerID": [f"TEST{i:04d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(yes_no, n),
        "Dependents": rng.choice(yes_no, n),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(yes_no, n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(internet_options, n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(contract_options, n),
        "PaperlessBilling": rng.choice(yes_no, n),
        "PaymentMethod": rng.choice(payment_options, n),
        "MonthlyCharges": rng.uniform(20, 120, n).round(2),
    }
    data["TotalCharges"] = (np.asarray(data["MonthlyCharges"]) * np.asarray(data["tenure"])).round(2)
    # A learnable but noisy target with both classes in every split.
    churn_probability = 1 / (1 + np.exp((np.asarray(data["tenure"]) - 30) / 12))
    churn_probability = 0.12 + 0.72 * churn_probability
    data["Churn"] = np.where(rng.random(n) < churn_probability, "Yes", "No")
    if len(set(data["Churn"])) < 2:
        data["Churn"][0:2] = ["Yes", "No"]
    return pd.DataFrame(data)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mlp_training_and_inference_round_trip(tmp_path: Path):
    pytest.importorskip("torch")
    dataset = make_telco_sample()
    data_path = tmp_path / "telco.csv"
    dataset.to_csv(data_path, index=False)
    output_dir = tmp_path / "models" / "deep_learning"
    config = TrainingConfig(max_epochs=2, patience=2, batch_size=32, hidden_dims=(16, 8))
    result = train_from_csv(
        data_path=data_path,
        output_dir=output_dir,
        config=config,
        include_rf_baseline=False,
    )
    assert "DeepLearningMLP_Structured" in result["metrics"]
    assert (output_dir / "structured" / "model_state.pt").exists()
    assert (output_dir / "comparison_metrics.json").exists()
    metrics = json.loads((output_dir / "comparison_metrics.json").read_text())
    assert metrics["evaluation_protocol"]["preprocessing_fit_on_training_only"] is True

    inference_data = dataset.drop(columns=["Churn"]).head(8)
    probabilities = predict_deep_learning(inference_data, output_dir / "structured")
    assert probabilities.shape == (8,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
