from pathlib import Path
import pandas as pd
from src.db import upsert_scores

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data" / "processed" / "customer_risk_scores.csv"

if __name__ == "__main__":
    if not SCORES.exists():
        raise SystemExit("Run scripts/score_customers.py first.")
    df = pd.read_csv(SCORES)
    count = upsert_scores(df)
    print(f"Upserted {count:,} score records into MySQL.")
