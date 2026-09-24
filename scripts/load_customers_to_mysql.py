"""
Load the Telco customer CSV into MySQL without relying on LOAD DATA LOCAL INFILE.

Usage from the project root:
    python scripts/load_customers_to_mysql.py

The script:
- reads data/raw/telco_customer_churn.csv by default
- validates the expected 21 source columns
- converts numeric fields safely
- derives the same three research features used by the project
- upserts customers into customer_risk_intelligence.customers
- prints row-count and quality checks after loading

No password is stored in the project. The MySQL password is requested securely.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import Iterable

import mysql.connector
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

INSERT_COLUMNS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn",
    "annual_revenue",
    "tenure_group",
    "monthly_charge_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Telco churn customers into MySQL."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--database", default="customer_risk_intelligence")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows written per batch (default: 500)",
    )
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def engineer_research_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["AnnualRevenue"] = out["MonthlyCharges"] * 12

    out["TenureGroup"] = pd.cut(
        out["tenure"],
        bins=[-float("inf"), 6, 12, 24, 48, 72],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
        ],
        right=True,
        include_lowest=True,
    ).astype("string")

    out["MonthlyChargeGroup"] = pd.Series(
        pd.NA, index=out.index, dtype="string"
    )
    out.loc[out["MonthlyCharges"] < 40, "MonthlyChargeGroup"] = "Low"
    out.loc[
        (out["MonthlyCharges"] >= 40) & (out["MonthlyCharges"] < 70),
        "MonthlyChargeGroup",
    ] = "Medium-Low"
    out.loc[
        (out["MonthlyCharges"] >= 70) & (out["MonthlyCharges"] <= 90),
        "MonthlyChargeGroup",
    ] = "Medium-High"
    out.loc[out["MonthlyCharges"] > 90, "MonthlyChargeGroup"] = "High"

    return out


def validate_source(df: pd.DataFrame) -> None:
    actual = list(df.columns)

    missing = [col for col in EXPECTED_COLUMNS if col not in actual]
    unexpected = [col for col in actual if col not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if unexpected:
        print(
            "Note: extra CSV columns will be ignored: "
            + ", ".join(unexpected)
        )

    if df["customerID"].isna().any() or (normalize_text(df["customerID"]) == "").any():
        raise ValueError("customerID contains blank or missing values.")

    duplicate_ids = normalize_text(df["customerID"]).duplicated().sum()
    if duplicate_ids:
        raise ValueError(
            f"Found {duplicate_ids} duplicate customer IDs; load aborted."
        )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df[EXPECTED_COLUMNS].copy()

    # Text cleanup.
    for col in EXPECTED_COLUMNS:
        if col not in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}:
            out[col] = normalize_text(out[col])

    # Numeric conversion.
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # TotalCharges has known blank values in the source dataset.
    # Preserve them as NULL; the ML pipeline already handles missing values.
    out.loc[out["customerID"].eq(""), "customerID"] = pd.NA

    out = engineer_research_features(out)

    return out


def to_rows(df: pd.DataFrame) -> Iterable[tuple]:
    export = df.copy()

    # Match MySQL column names.
    export = export.rename(
        columns={
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "contract",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churn",
            "AnnualRevenue": "annual_revenue",
            "TenureGroup": "tenure_group",
            "MonthlyChargeGroup": "monthly_charge_group",
            "gender": "gender",
            "tenure": "tenure",
        }
    )

    # MySQL connector handles None as SQL NULL.
    export = export.where(pd.notna(export), None)

    for row in export[INSERT_COLUMNS].itertuples(index=False, name=None):
        yield row


def load(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    print(f"CSV:      {csv_path}")
    print(f"MySQL:    {args.user}@{args.host}:{args.port}")
    print(f"Database: {args.database}")

    source = pd.read_csv(csv_path)
    validate_source(source)

    prepared = prepare(source)

    print(f"Source rows: {len(prepared):,}")
    print(
        f"Blank TotalCharges: "
        f"{int(prepared['TotalCharges'].isna().sum()):,}"
    )

    password = getpass.getpass("MySQL password: ")

    connection = mysql.connector.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=password,
        database=args.database,
    )

    insert_sql = f"""
        INSERT INTO customers ({", ".join(INSERT_COLUMNS)})
        VALUES ({", ".join(["%s"] * len(INSERT_COLUMNS))})
        ON DUPLICATE KEY UPDATE
            gender = VALUES(gender),
            senior_citizen = VALUES(senior_citizen),
            partner = VALUES(partner),
            dependents = VALUES(dependents),
            tenure = VALUES(tenure),
            phone_service = VALUES(phone_service),
            multiple_lines = VALUES(multiple_lines),
            internet_service = VALUES(internet_service),
            online_security = VALUES(online_security),
            online_backup = VALUES(online_backup),
            device_protection = VALUES(device_protection),
            tech_support = VALUES(tech_support),
            streaming_tv = VALUES(streaming_tv),
            streaming_movies = VALUES(streaming_movies),
            contract = VALUES(contract),
            paperless_billing = VALUES(paperless_billing),
            payment_method = VALUES(payment_method),
            monthly_charges = VALUES(monthly_charges),
            total_charges = VALUES(total_charges),
            churn = VALUES(churn),
            annual_revenue = VALUES(annual_revenue),
            tenure_group = VALUES(tenure_group),
            monthly_charge_group = VALUES(monthly_charge_group)
    """

    cursor = connection.cursor()

    try:
        rows = list(to_rows(prepared))

        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            cursor.executemany(insert_sql, batch)
            connection.commit()
            print(
                f"Loaded batch {start + 1:,}-{start + len(batch):,} "
                f"of {len(rows):,}"
            )

        cursor.execute("SELECT COUNT(*) FROM customers")
        db_count = int(cursor.fetchone()[0])

        cursor.execute(
            """
            SELECT
                COUNT(*) AS rows_total,
                COUNT(DISTINCT customer_id) AS unique_customers,
                SUM(churn = 'Yes') AS churners,
                SUM(total_charges IS NULL) AS blank_total_charges
            FROM customers
            """
        )
        stats = cursor.fetchone()

        print("\nLOAD COMPLETE")
        print(f"Database rows:       {db_count:,}")
        print(f"Unique customers:    {int(stats[1]):,}")
        print(f"Observed churners:   {int(stats[2]):,}")
        print(f"NULL TotalCharges:   {int(stats[3]):,}")

        if db_count < len(prepared):
            raise RuntimeError(
                "Database row count is smaller than the source row count."
            )

    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def main() -> int:
    args = parse_args()

    try:
        load(args)
        return 0
    except KeyboardInterrupt:
        print("\nLoad cancelled.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
