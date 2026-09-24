from pathlib import Path
from src.modeling import train_from_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
MODEL = ROOT / "models" / "calibrated_research_rf.joblib"
META = ROOT / "models" / "model_metadata.json"

if __name__ == "__main__":
    if not DATA.exists():
        raise SystemExit(f"Missing dataset: {DATA}\nPlace the Telco Customer Churn CSV there first.")
    meta = train_from_csv(DATA, MODEL, META)
    print("Production refit saved:", MODEL)
    print("Metadata saved:", META)
    print(meta)
