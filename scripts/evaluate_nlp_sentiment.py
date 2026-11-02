"""Evaluate general sentiment predictions against a manually labelled sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp import SupportSentimentAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with ticket_text and true_sentiment columns.")
    parser.add_argument("--text-column", default="ticket_text")
    parser.add_argument("--label-column", default="true_sentiment")
    parser.add_argument("--output", default="results/nlp_sentiment_evaluation.json")
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise SystemExit(f"Labelled evaluation file not found: {source}")
    df = pd.read_csv(source)
    required = {args.text_column, args.label_column}
    if not required.issubset(df.columns):
        raise SystemExit(f"Input needs columns: {sorted(required)}")
    label_map = {"POSITIVE": "positive", "NEGATIVE": "negative", "POS": "positive", "NEG": "negative"}
    true_labels = df[args.label_column].astype(str).str.strip().str.upper().map(label_map)
    valid = true_labels.notna() & df[args.text_column].notna()
    df = df.loc[valid].copy()
    true_labels = true_labels.loc[valid].tolist()
    if len(df) < 10:
        raise SystemExit("Use at least 10 manually labelled, non-empty examples for a basic evaluation; more is better.")
    analyzer = SupportSentimentAnalyzer(model_name=args.model)
    scored = analyzer.predict(df[args.text_column].tolist())
    predicted = [row["sentiment_label"].lower() for row in scored]
    labels = ["negative", "positive"]
    result = {
        "model": args.model,
        "labelled_rows_evaluated": len(df),
        "accuracy": float(accuracy_score(true_labels, predicted)),
        "precision_macro": float(precision_score(true_labels, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predicted, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(true_labels, predicted, labels=labels).tolist(),
        "classification_report": classification_report(true_labels, predicted, labels=labels, output_dict=True, zero_division=0),
        "note": "Evaluate on manually labelled telecom/customer-support text; SST-2 validation scores are not domain validation.",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
