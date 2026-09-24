from pathlib import Path
import pandas as pd
from src.predict import CustomerRiskScorer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
MODEL = ROOT / "models" / "calibrated_research_rf.joblib"
OUT = ROOT / "data" / "processed" / "customer_risk_scores.csv"

if __name__ == "__main__":
    if not DATA.exists() or not MODEL.exists():
        raise SystemExit("Run scripts/train_production_model.py after placing the raw CSV in data/raw/.")
    df = pd.read_csv(DATA)
    scored = CustomerRiskScorer(MODEL).score(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT, index=False)
    print(f"Saved {len(scored):,} scored customers to {OUT}")
