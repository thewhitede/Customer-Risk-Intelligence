"""Generate deterministic, fully synthetic telecom churn + support-text demo data.

The generated records contain no real customer information. They are intended to
smoke-test the add-on pipeline, not to support research or business performance claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "demo"
SEED = 20261028
SNAPSHOT_DATE = pd.Timestamp("2026-10-01")


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def _yes_no(rng: np.random.Generator, p_yes: np.ndarray | float, n: int) -> np.ndarray:
    return np.where(rng.random(n) < p_yes, "Yes", "No")


def generate_datasets(n_customers: int = 1200, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_customers < 100:
        raise ValueError("Use at least 100 customers for a useful demo split.")
    rng = np.random.default_rng(seed)
    n = int(n_customers)
    customer_ids = np.array([f"DEMO{i:05d}" for i in range(1, n + 1)])

    gender = rng.choice(["Female", "Male"], n)
    senior = rng.binomial(1, 0.16, n)
    partner = _yes_no(rng, 0.48, n)
    dependents = np.where(partner == "Yes", _yes_no(rng, 0.42, n), _yes_no(rng, 0.16, n))

    # A mixture of short-, medium-, and long-tenure accounts gives the demo a
    # plausible range of customer lifecycle stages without reproducing real people.
    tenure = np.clip(np.round(rng.gamma(shape=2.15, scale=15.5, size=n)), 0, 72).astype(int)
    tenure[tenure == 0] = rng.integers(1, 3, size=(tenure == 0).sum())

    phone_service = _yes_no(rng, 0.91, n)
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        _yes_no(rng, 0.44, n),
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.36, 0.44, 0.20])

    # A latent dissatisfaction signal influences both churn likelihood and the
    # tone/content of simulated ticket text. It is not derived from the final
    # Churn label, avoiding direct label-copying into the NLP data.
    preliminary_risk = (
        0.60 * (tenure < 12).astype(float)
        + 0.48 * (tenure < 24).astype(float)
        + 0.55 * (internet_service == "Fiber optic").astype(float)
        + 0.25 * (senior == 1).astype(float)
        + rng.normal(0, 0.55, n)
    )
    dissatisfaction = _sigmoid(preliminary_risk - 0.60)

    def internet_service_choice(p_yes: np.ndarray) -> np.ndarray:
        result = _yes_no(rng, p_yes, n)
        result[internet_service == "No"] = "No internet service"
        return result

    online_security = internet_service_choice(0.30 - 0.06 * dissatisfaction)
    online_backup = internet_service_choice(0.42 - 0.04 * dissatisfaction)
    device_protection = internet_service_choice(0.39 - 0.04 * dissatisfaction)
    tech_support = internet_service_choice(0.31 - 0.12 * dissatisfaction)
    streaming_tv = internet_service_choice(0.48)
    streaming_movies = internet_service_choice(0.47)

    contract = []
    for months in tenure:
        if months < 12:
            probabilities = [0.83, 0.14, 0.03]
        elif months < 36:
            probabilities = [0.60, 0.27, 0.13]
        else:
            probabilities = [0.30, 0.35, 0.35]
        contract.append(rng.choice(["Month-to-month", "One year", "Two year"], p=probabilities))
    contract = np.asarray(contract)

    paperless_billing = _yes_no(rng, 0.61, n)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.34, 0.18, 0.25, 0.23],
    )

    internet_base = np.select(
        [internet_service == "No", internet_service == "DSL", internet_service == "Fiber optic"],
        [20.0, 45.0, 72.0],
        default=45.0,
    )
    service_addons = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly_charges = np.clip(
        18 + internet_base + 3.0 * service_addons + 8.0 * (phone_service == "Yes").astype(int)
        + rng.normal(0, 6.0, n),
        18.0,
        125.0,
    ).round(2)
    total_charges = np.maximum(0.0, monthly_charges * tenure + rng.normal(0, 22.0, n)).round(2)
    total_charges = np.where(tenure <= 1, monthly_charges * tenure, total_charges)

    contract_risk = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [0.95, -0.10, -0.80],
        default=0.0,
    )
    churn_logit = (
        -1.90
        + contract_risk
        + 0.52 * (tenure < 12).astype(float)
        + 0.26 * (tenure < 24).astype(float)
        + 0.31 * (tech_support == "No").astype(float)
        + 0.20 * (online_security == "No").astype(float)
        + 0.24 * (internet_service == "Fiber optic").astype(float)
        + 0.24 * (payment_method == "Electronic check").astype(float)
        + 0.20 * (monthly_charges > 85).astype(float)
        - 0.20 * (dependents == "Yes").astype(float)
        + 0.60 * (dissatisfaction - 0.5)
        + rng.normal(0, 0.25, n)
    )
    churn = np.where(rng.random(n) < _sigmoid(churn_logit), "Yes", "No")
    if len(np.unique(churn)) < 2:
        churn[:2] = ["Yes", "No"]

    customers = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    negative_templates = {
        "billing": [
            "I was charged twice this month and the billing issue is still unresolved.",
            "The latest invoice has an unexpected charge and I am unhappy with the explanation.",
            "My bill keeps increasing and nobody has explained the extra fees.",
            "I contacted support about an incorrect payment but the problem remains.",
        ],
        "connectivity": [
            "The internet connection keeps dropping and the service is very frustrating.",
            "My broadband is slow again and video calls keep disconnecting.",
            "The connection has been unreliable for days and I need this fixed.",
            "I have repeated network problems and the service is not meeting expectations.",
        ],
        "support": [
            "I have contacted support several times but nobody has resolved my issue.",
            "The agent was unhelpful and I am disappointed with the response.",
            "I waited a long time for help and the problem is still open.",
            "The support process has been confusing and I am not satisfied.",
        ],
        "price_contract": [
            "The monthly price is too high for the service I receive.",
            "I want to cancel because this plan costs more than I expected.",
            "The contract terms are unclear and the price increase is frustrating.",
            "I am considering switching because there are cheaper plans elsewhere.",
        ],
    }
    positive_templates = {
        "billing": [
            "The bill was clear this month and the payment was processed correctly.",
            "The billing team corrected my invoice quickly and explained the charges.",
            "My payment went through smoothly and the account details look right.",
        ],
        "connectivity": [
            "The internet has been reliable lately and streaming works smoothly.",
            "My connection is fast and video calls have been stable this week.",
            "The service is working well and I have had no network interruptions.",
        ],
        "support": [
            "The support agent was helpful and resolved my issue quickly.",
            "Customer service explained the options clearly and followed up as promised.",
            "I received friendly support and the issue was fixed on the first contact.",
        ],
        "price_contract": [
            "The plan is good value and the monthly price suits my needs.",
            "The contract options were clear and I am happy with my current plan.",
            "The service is affordable and I am satisfied with what is included.",
        ],
    }
    ticket_rows: list[dict[str, object]] = []
    start_date = pd.Timestamp("2026-01-01")
    date_span = int((SNAPSHOT_DATE - pd.Timedelta(days=1) - start_date).days) + 1

    for i, customer_id in enumerate(customer_ids):
        # Shared latent dissatisfaction drives text sentiment independently of the
        # generated churn draw. Some customers have no support contacts.
        contact_probability = float(_sigmoid(-0.35 + 1.1 * dissatisfaction[i]))
        if rng.random() > contact_probability:
            n_tickets = 0
        else:
            n_tickets = int(np.clip(rng.poisson(0.75 + 1.7 * dissatisfaction[i]) + 1, 1, 5))
        for ticket_number in range(n_tickets):
            negative_probability = float(np.clip(0.18 + 0.66 * dissatisfaction[i], 0.08, 0.92))
            sentiment = "negative" if rng.random() < negative_probability else "positive"
            if contract[i] == "Month-to-month" and monthly_charges[i] > 80:
                topic_weights = [0.22, 0.15, 0.17, 0.46]
            elif internet_service[i] in {"Fiber optic", "DSL"}:
                topic_weights = [0.22, 0.43, 0.22, 0.13]
            else:
                topic_weights = [0.28, 0.12, 0.32, 0.28]
            topic = str(rng.choice(["billing", "connectivity", "support", "price_contract"], p=topic_weights))
            templates = negative_templates if sentiment == "negative" else positive_templates
            message = str(rng.choice(templates[topic]))
            # A small deterministic-seeming surface variation adds realistic token variation.
            prefix = rng.choice(["", "Hello, ", "Hi, ", "Regarding my account: "])
            suffix = rng.choice(["", " Please advise.", " Thanks.", " This matters to me."])
            message = f"{prefix}{message}{suffix}".strip()
            ticket_day = start_date + pd.Timedelta(days=int(rng.integers(0, max(date_span, 1))))
            ticket_rows.append({
                "ticket_id": f"TKT{i + 1:05d}_{ticket_number + 1:02d}",
                "customerID": customer_id,
                "ticket_date": ticket_day.date().isoformat(),
                "ticket_text": message,
                "true_sentiment": sentiment,
                "demo_note": "synthetic_label_not_manual_annotation",
            })

    tickets = pd.DataFrame(ticket_rows).sort_values(["ticket_date", "customerID", "ticket_id"]).reset_index(drop=True)
    if tickets.empty:
        raise RuntimeError("The synthetic support dataset unexpectedly contains no tickets.")

    # Demo-only aggregates let users smoke-test the hybrid MLP without first
    # downloading a Transformer checkpoint. They are derived from synthetic
    # sentiment annotations, not from a trained or pretrained NLP model.
    score = np.where(tickets["true_sentiment"].eq("negative"), -0.75, 0.75)
    tickets = tickets.assign(_demo_score=score, _demo_negative=tickets["true_sentiment"].eq("negative").astype(int))
    feature_rows = []
    for customer_id, group in tickets.groupby("customerID", sort=False):
        latest = group.sort_values("ticket_date").iloc[-1]
        feature_rows.append({
            "customerID": customer_id,
            "ticket_count": int(len(group)),
            "negative_ticket_count": int(group["_demo_negative"].sum()),
            "negative_ticket_ratio": float(group["_demo_negative"].mean()),
            "mean_sentiment_score": float(group["_demo_score"].mean()),
            "latest_sentiment_score": float(latest["_demo_score"]),
            "latest_ticket_date": str(latest["ticket_date"]),
            "demo_note": "synthetic_precomputed_features_not_model_output",
        })
    demo_features = pd.DataFrame(feature_rows)

    return customers, tickets.drop(columns=["_demo_score", "_demo_negative"]), demo_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory to write the three demo CSVs.")
    parser.add_argument("--customers", type=int, default=1200, help="Number of synthetic customers (minimum 100).")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, tickets, features = generate_datasets(args.customers, args.seed)
    paths = {
        "churn_data": output_dir / "telco_customer_churn_demo.csv",
        "support_text": output_dir / "customer_support_text_demo.csv",
        "precomputed_demo_features": output_dir / "customer_text_features_demo.csv",
    }
    customers.to_csv(paths["churn_data"], index=False)
    tickets.to_csv(paths["support_text"], index=False)
    features.to_csv(paths["precomputed_demo_features"], index=False)

    print(f"Synthetic churn rows: {len(customers):,} -> {paths['churn_data']}")
    print(f"Synthetic support tickets: {len(tickets):,} -> {paths['support_text']}")
    print(f"Customers with synthetic ticket aggregates: {len(features):,} -> {paths['precomputed_demo_features']}")
    print(f"Synthetic churn prevalence: {customers['Churn'].eq('Yes').mean():.1%}")
    print(f"All ticket dates precede the churn snapshot date {SNAPSHOT_DATE.date()}.")


if __name__ == "__main__":
    main()
