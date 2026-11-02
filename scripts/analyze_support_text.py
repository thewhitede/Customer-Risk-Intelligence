"""Generate sentiment scores and exploratory topics for customer support text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp import DEFAULT_SENTIMENT_MODEL, SupportSentimentAnalyzer, analyze_support_tickets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/customer_support_text.csv")
    parser.add_argument("--output-dir", default="data/processed/customer_voice")
    parser.add_argument("--id-column", default="customerID")
    parser.add_argument("--text-column", default="ticket_text")
    parser.add_argument("--date-column", default="ticket_date")
    parser.add_argument("--model", default=DEFAULT_SENTIMENT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-topics", type=int, default=5, help="Set to 0 to skip topic discovery.")
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Optional cutoff YYYY-MM-DD; tickets after the cutoff are excluded before aggregating customer features.",
    )
    return parser.parse_args()



def normalize_support_input(tickets: pd.DataFrame, id_column: str = "customerID",
                           text_column: str = "ticket_text",
                           date_column: str = "ticket_date") -> pd.DataFrame:
    """Accept either ticket-level rows or the project's wide master customer CSV.

    Wide format is converted to one row per non-empty ticket. Customer provenance
    is retained so downstream outputs can be labelled as synthetic.
    """
    if text_column in tickets.columns:
        return tickets.copy()

    text_columns = [
        column for column in tickets.columns
        if column.startswith("ticket_text_")
    ]
    if not text_columns:
        raise ValueError(
            "Input must contain ticket_text or wide columns such as "
            "ticket_text_1, ticket_text_2, ticket_text_3."
        )

    rows = []
    for text_col in text_columns:
        suffix = text_col.rsplit("_", 1)[-1]
        date_col = f"ticket_date_{suffix}"
        keep_columns = [id_column]
        for optional in ("record_origin",):
            if optional in tickets.columns:
                keep_columns.append(optional)
        part = tickets[keep_columns].copy()
        part[text_column] = tickets[text_col]
        if date_col in tickets.columns:
            part[date_column] = tickets[date_col]
        else:
            part[date_column] = pd.NaT
        part["ticket_source_column"] = text_col
        rows.append(part)

    long_tickets = pd.concat(rows, ignore_index=True)
    long_tickets[text_column] = long_tickets[text_column].astype("string").str.strip()
    long_tickets = long_tickets[
        long_tickets[text_column].notna()
        & long_tickets[text_column].ne("")
    ].copy()
    long_tickets["is_synthetic_customer"] = (
        long_tickets[id_column].astype(str).str.upper().str.startswith(("SYNTH_", "DEMO"))
    )
    return long_tickets.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = (ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not input_path.exists():
        raise SystemExit(
            f"Support-text data not found: {input_path}\n"
            "Create a CSV with customerID and ticket_text columns; ticket_date is recommended."
        )
    source_df = pd.read_csv(input_path)
    tickets = normalize_support_input(
        source_df,
        id_column=args.id_column,
        text_column=args.text_column,
        date_column=args.date_column,
    )
    date_column = args.date_column if args.date_column in tickets.columns else None
    if tickets.empty:
        raise SystemExit("No non-empty support ticket text was found in the input CSV.")
    if args.as_of_date:
        if not date_column:
            raise SystemExit("--as-of-date was supplied, but the input has no configured date column.")
        dates = pd.to_datetime(tickets[date_column], errors="coerce", utc=True)
        cutoff = pd.to_datetime(args.as_of_date, errors="raise", utc=True) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        tickets = tickets[dates <= cutoff].copy()
        if tickets.empty:
            raise SystemExit("No tickets remain after the specified as-of cutoff.")

    analyzer = SupportSentimentAnalyzer(model_name=args.model, batch_size=args.batch_size)
    scored, features, topic_artifacts = analyze_support_tickets(
        tickets=tickets,
        id_column=args.id_column,
        text_column=args.text_column,
        date_column=date_column,
        analyzer=analyzer,
        n_topics=max(0, args.n_topics),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ticket_output = output_dir / "customer_tickets_scored.csv"
    feature_output = output_dir / "customer_text_features.csv"
    scored.to_csv(ticket_output, index=False)
    features.to_csv(feature_output, index=False)

    if topic_artifacts is not None:
        joblib.dump(
            {
                "vectorizer": topic_artifacts["vectorizer"],
                "model": topic_artifacts["model"],
                "random_state": topic_artifacts["random_state"],
            },
            output_dir / "topic_model.joblib",
        )
        (output_dir / "topics.json").write_text(
            json.dumps(topic_artifacts["topic_info"], indent=2), encoding="utf-8"
        )

    summary = {
        "input_path": str(input_path),
        "sentiment_model": args.model,
        "tickets_scored": int(len(scored)),
        "customers_aggregated": int(features[args.id_column].nunique()),
        "synthetic_customers_in_ticket_input": int(
            tickets.loc[
                tickets[args.id_column].astype(str).str.upper().str.startswith(("SYNTH_", "DEMO")),
                args.id_column,
            ].nunique()
        ),
        "input_format": (
            "wide_customer_master" if any(c.startswith("ticket_text_") for c in source_df.columns)
            else "ticket_level"
        ),
        "as_of_date": args.as_of_date,
        "topic_discovery_enabled": topic_artifacts is not None,
        "ticket_output": str(ticket_output),
        "feature_output": str(feature_output),
        "warning": (
            "General English sentiment model; validate on manually labelled telecom examples. "
            "Only use time-safe customer aggregates for churn prediction."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
