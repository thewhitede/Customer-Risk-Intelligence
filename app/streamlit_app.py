
from __future__ import annotations

from pathlib import Path
import json
import sys
from io import BytesIO

import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import CustomerRiskScorer
from src.research_summary import RESEARCH_METRICS, RISK_VALIDATION, TARGETING, RETENTION_PRIORITY
from src.explainability import explain_customer
from src.db import load_customer_risk_scores
from src.voice_risk_dashboard import render_customer_voice_risk_section

MODEL_PATH = ROOT / "models" / "calibrated_research_rf.joblib"
DEFAULT_DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"

st.set_page_config(
    page_title="Customer Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero { padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,.22); border-radius: 18px;
            background: linear-gradient(135deg, rgba(120,120,120,.10), rgba(120,120,120,.035)); margin-bottom: 1rem; }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.25rem; }
    .hero p { margin: 0; color: rgba(120,120,120,.95); font-size: 1rem; }
    .section-title { font-size: 1.05rem; font-weight: 700; margin: .4rem 0 .6rem 0; }
    .mini-card { border: 1px solid rgba(128,128,128,.20); border-radius: 14px; padding: .9rem 1rem; min-height: 112px; }
    .mini-label { font-size: .78rem; color: rgba(120,120,120,.95); text-transform: uppercase; letter-spacing: .04em; }
    .mini-value { font-size: 1.65rem; font-weight: 750; margin-top: .2rem; }
    .risk-pill { display: inline-block; padding: .25rem .65rem; border-radius: 999px; font-weight: 700; font-size: .85rem; border: 1px solid rgba(128,128,128,.18); }
    .muted { color: rgba(120,120,120,.95); font-size: .9rem; }
    .finding { border-left: 4px solid currentColor; padding: .55rem .85rem; background: rgba(120,120,120,.06); border-radius: 0 10px 10px 0; margin: .5rem 0; }
    .footer-note { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.18); color: rgba(120,120,120,.85); font-size: .8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    """Production-safe normalization before scoring."""
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    if "MonthlyCharges" in out.columns:
        out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    if "tenure" in out.columns:
        out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce")
    return out


def prepare_explain_input(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the same engineered feature space used by the research model for SHAP."""
    out = clean_input(df)
    if "MonthlyCharges" in out.columns:
        out["AnnualRevenue"] = out["MonthlyCharges"] * 12
    if "tenure" in out.columns:
        out["TenureGroup"] = pd.cut(
            out["tenure"],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=[
                "0-6 months",
                "7-12 months",
                "13-24 months",
                "25-48 months",
                "49-72 months",
            ],
        )
    if "MonthlyCharges" in out.columns:
        out["MonthlyChargeGroup"] = pd.cut(
            out["MonthlyCharges"],
            bins=[-float("inf"), 40, 70, 90, float("inf")],
            labels=["Low", "Medium-Low", "Medium-High", "High"],
        )
    return out


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">{label}</div>'
        f'<div class="mini-value">{value}</div><div class="muted">{detail}</div></div>',
        unsafe_allow_html=True,
    )


RAW_CUSTOMER_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

EXPECTED_OPTIONAL_COLUMNS = ["Churn"]

NUMERIC_CUSTOMER_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]


def assess_data_quality(raw_df: pd.DataFrame) -> dict:
    """Validate customer-level input without changing the scoring pipeline."""
    missing_required = [
        col for col in RAW_CUSTOMER_COLUMNS
        if col not in raw_df.columns
    ]

    duplicate_rows = int(raw_df.duplicated().sum())

    duplicate_ids = 0
    if "customerID" in raw_df.columns:
        duplicate_ids = int(raw_df["customerID"].duplicated().sum())

    numeric_invalid = {}
    for col in NUMERIC_CUSTOMER_COLUMNS:
        if col in raw_df.columns:
            converted = pd.to_numeric(raw_df[col], errors="coerce")
            # Count values that are non-empty in the raw file but cannot be
            # interpreted numerically. Genuine blanks are reported separately.
            raw_nonempty = raw_df[col].notna() & raw_df[col].astype(str).str.strip().ne("")
            numeric_invalid[col] = int((raw_nonempty & converted.isna()).sum())

    blank_totalcharges = 0
    if "TotalCharges" in raw_df.columns:
        blank_totalcharges = int(
            raw_df["TotalCharges"].isna().sum()
            + raw_df["TotalCharges"].astype(str).str.strip().eq("").sum()
        )

    unexpected = [
        col for col in raw_df.columns
        if col not in RAW_CUSTOMER_COLUMNS + EXPECTED_OPTIONAL_COLUMNS
    ]

    blocking_issues = []
    if missing_required:
        blocking_issues.append("missing required scoring columns")
    if "customerID" in raw_df.columns and duplicate_ids > 0:
        blocking_issues.append("duplicate customer IDs")

    warning_count = (
        duplicate_rows
        + blank_totalcharges
        + sum(numeric_invalid.values())
    )

    status = "READY"
    if blocking_issues:
        status = "BLOCKED"
    elif warning_count > 0 or unexpected:
        status = "CHECK"

    return {
        "status": status,
        "rows": len(raw_df),
        "missing_required": missing_required,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "blank_totalcharges": blank_totalcharges,
        "numeric_invalid": numeric_invalid,
        "unexpected": unexpected,
    }


# -----------------------------------------------------------------------------
# Header and data loading
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>Customer Risk Intelligence</h1>'
    '<p>Explainable churn prediction • calibrated risk • revenue exposure • retention prioritization</p></div>',
    unsafe_allow_html=True,
)

# The model is required for CSV scoring and local SHAP explanations, but not
# when MySQL already contains persisted risk scores.

with st.sidebar:
    st.header("Data source")
    data_source = st.radio(
        "Scoring data source",
        ["CSV", "MySQL"],
        index=0,
    )

    if data_source == "CSV":
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        source = uploaded if uploaded is not None else (
            str(DEFAULT_DATA) if DEFAULT_DATA.exists() else None
        )

        if source is None:
            st.error("No customer CSV available.")
            st.stop()

        st.caption(
            "CSV mode scores customers with the calibrated research Random Forest artifact. "
            "For the combined customer-and-voice file, upload "
            "`data/raw/telco_customer_risk_and_voice_master.csv`. Extra ticket columns are "
            "kept out of the production model's feature inputs."
        )

    else:
        st.caption("MySQL mode loads persisted customer risk scores; it does not rescore customers.")

if data_source == "MySQL":
    try:
        scored = load_customer_risk_scores()
    except Exception as e:
        st.error(f"MySQL connection failed: {e}")
        st.stop()

    if scored.empty:
        st.warning("No customer risk scores found in MySQL.")
        st.stop()

    # Normalize the database view into the column names used by the existing UI.
    scored = scored.rename(columns={
        "customer_id": "customerID",
        "monthly_charges": "MonthlyCharges",
        "calibrated_churn_probability": "CalibratedChurnProbability",
        "calibrated_risk_level": "CalibratedRiskLevel",
        "revenue_at_risk": "RevenueAtRisk",
        "retention_priority_score": "RetentionPriorityScore",
        "retention_priority": "RetentionPriority",
        "retention_recommendation": "RetentionRecommendation",
        "contract": "Contract",
        "internet_service": "InternetService",
        "payment_method": "PaymentMethod",
    })

    # Keep the existing customer-profile/SHAP experience when the project's
    # default raw CSV is available, but never rescore it in MySQL mode.
    if DEFAULT_DATA.exists():
        try:
            raw_customers = pd.read_csv(DEFAULT_DATA)
            quality = assess_data_quality(raw_customers)
            profile_columns = RAW_CUSTOMER_COLUMNS + (
                ["Churn"] if "Churn" in raw_customers.columns else []
            )
            customers = clean_input(raw_customers[profile_columns].copy())
        except Exception:
            raw_customers = None
            customers = None
    else:
        raw_customers = None
        customers = None

    st.success(f"Loaded {len(scored):,} persisted customer risk scores from MySQL.")

else:
    try:
        raw_customers = pd.read_csv(source)
        quality = assess_data_quality(raw_customers)

        with st.sidebar.expander("Data quality", expanded=False):
            if quality["status"] == "READY":
                st.success("Input passed validation.")
            elif quality["status"] == "CHECK":
                st.warning("Input loaded with review items.")
            else:
                st.error("Input cannot be scored safely.")

            q1, q2 = st.columns(2)
            with q1:
                st.metric("Rows", f"{quality['rows']:,}")
                st.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
            with q2:
                st.metric("Duplicate IDs", f"{quality['duplicate_ids']:,}")
                st.metric("Blank TotalCharges", f"{quality['blank_totalcharges']:,}")

            if quality["missing_required"]:
                st.error(
                    "Missing required columns: "
                    + ", ".join(quality["missing_required"])
                )
            if quality["unexpected"]:
                st.caption(
                    "Extra columns ignored by the scorer: "
                    + ", ".join(quality["unexpected"])
                )
            invalid_items = [
                f"{col}: {count}"
                for col, count in quality["numeric_invalid"].items()
                if count > 0
            ]
            if invalid_items:
                st.caption(
                    "Non-numeric values requiring review: "
                    + ", ".join(invalid_items)
                )
            if quality["blank_totalcharges"]:
                st.caption(
                    "Blank TotalCharges values are allowed; the existing "
                    "model pipeline handles them through numeric conversion "
                    "and imputation."
                )

        if quality["status"] == "BLOCKED":
            st.error(
                "Please fix the required schema or duplicate customer IDs "
                "before scoring this customer-level dataset."
            )
            st.stop()

        # The master customer-and-voice CSV contains extra provenance and ticket
        # columns. The production Random Forest must receive only its original
        # Telco customer feature schema.
        scoring_columns = RAW_CUSTOMER_COLUMNS + (
            ["Churn"] if "Churn" in raw_customers.columns else []
        )
        customers = clean_input(raw_customers[scoring_columns].copy())

    except Exception as exc:
        st.error(f"Could not read or validate CSV: {exc}")
        st.stop()

    if not MODEL_PATH.exists():
        st.warning("Production model artifact is not present yet.")
        st.info(
            "Place the Telco CSV at data/raw/telco_customer_churn.csv and train "
            "the production artifact with scripts/train_production_model.py."
        )
        st.stop()

    scorer = CustomerRiskScorer(MODEL_PATH)
    with st.spinner("Scoring customers..."):
        scored = scorer.score(customers)


# Useful app-only helper columns
scored = scored.copy()
scored["CustomerRank"] = scored["RetentionPriorityScore"].rank(method="first", ascending=False).astype(int)

# -----------------------------------------------------------------------------
# Global filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    risk_options = ["Low", "Medium", "High"]
    selected_risks = st.multiselect("Risk level", risk_options, default=risk_options)
    priority_options = ["Low", "Moderate", "High", "Critical"]
    selected_priorities = st.multiselect("Retention priority", priority_options, default=priority_options)
    min_probability = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)

filtered = scored[
    scored["CalibratedRiskLevel"].astype(str).isin(selected_risks)
    & scored["RetentionPriority"].astype(str).isin(selected_priorities)
    & (scored["CalibratedChurnProbability"] >= min_probability)
].copy()

st.caption(f"Showing {len(filtered):,} of {len(scored):,} customers after filters.")

risk_tab, customer_tab, retention_tab, research_tab, model_tab, voice_tab = st.tabs(
    ["Risk Overview", "Customer Scoring", "Retention Command Center", "Research Evidence", "Model Comparison", "Customer Voice (NLP)"]
)

# -----------------------------------------------------------------------------
# TAB 1 — Risk Overview
# -----------------------------------------------------------------------------
with risk_tab:
    total = len(filtered)
    high = int((filtered["CalibratedRiskLevel"].astype(str) == "High").sum())
    critical = int((filtered["RetentionPriority"].astype(str) == "Critical").sum())
    exposure = float(filtered["RevenueAtRisk"].sum())
    avg_prob = float(filtered["CalibratedChurnProbability"].mean()) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers analyzed", f"{total:,}", "Current filtered population")
    with c2:
        metric_card("High risk", f"{high:,}", f"{(high/total):.1%} of filtered customers" if total else "No customers")
    with c3:
        metric_card("Monthly revenue at risk", fmt_money(exposure), "Expected exposure, not guaranteed loss")
    with c4:
        metric_card("Critical priority", f"{critical:,}", "Highest retention-priority tier")

    st.markdown('<div class="section-title">Risk concentration</div>', unsafe_allow_html=True)
    left, right = st.columns(2)

    risk_order = ["Low", "Medium", "High"]
    risk_counts = (
        filtered["CalibratedRiskLevel"]
        .astype(str)
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk")
        .reset_index(name="Customers")
    )
    risk_counts["Risk"] = pd.Categorical(risk_counts["Risk"], categories=risk_order, ordered=True)

    with left:
        st.write("Customers by calibrated risk")
        risk_count_chart = (
            alt.Chart(risk_counts)
            .mark_bar()
            .encode(
                x=alt.X("Risk:N", sort=risk_order, title=None),
                y=alt.Y("Customers:Q", title="Customers"),
                tooltip=[alt.Tooltip("Risk:N", title="Risk"), alt.Tooltip("Customers:Q", title="Customers")],
            )
            .properties(height=280)
        )
        st.altair_chart(risk_count_chart, width="stretch")

    risk_churn = (
        filtered.groupby(filtered["CalibratedRiskLevel"].astype(str), observed=False)["CalibratedChurnProbability"]
        .mean()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk")
        .reset_index(name="Probability")
    )
    risk_churn["Risk"] = pd.Categorical(risk_churn["Risk"], categories=risk_order, ordered=True)

    with right:
        st.write("Average predicted churn probability")
        risk_probability_chart = (
            alt.Chart(risk_churn)
            .mark_bar()
            .encode(
                x=alt.X("Risk:N", sort=risk_order, title=None),
                y=alt.Y("Probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("Risk:N", title="Risk"), alt.Tooltip("Probability:Q", title="Probability", format=".1%")],
            )
            .properties(height=280)
        )
        st.altair_chart(risk_probability_chart, width="stretch")

    st.markdown('<div class="section-title">Highest-value opportunities</div>', unsafe_allow_html=True)
    top_cols = [
        "CustomerRank", "customerID", "CalibratedChurnProbability", "CalibratedRiskLevel",
        "MonthlyCharges", "RevenueAtRisk", "RetentionPriority", "RetentionRecommendation"
    ]
    top = filtered.sort_values(["RetentionPriorityScore", "RevenueAtRisk"], ascending=False)[top_cols].head(15)
    display = top.rename(columns={
        "CustomerRank": "Rank",
        "customerID": "Customer",
        "CalibratedChurnProbability": "Churn probability",
        "CalibratedRiskLevel": "Risk",
        "MonthlyCharges": "Monthly charges",
        "RevenueAtRisk": "Revenue at risk",
        "RetentionPriority": "Priority",
        "RetentionRecommendation": "Recommended action",
    }).copy()
    st.dataframe(
        display.style.format({
            "Churn probability": "{:.1%}",
            "Monthly charges": "${:,.2f}",
            "Revenue at risk": "${:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

# -----------------------------------------------------------------------------
# TAB 2 — Customer Scoring
# -----------------------------------------------------------------------------
with customer_tab:
    st.markdown('<div class="section-title">Individual customer assessment</div>', unsafe_allow_html=True)
    if "customerID" not in scored.columns:
        st.info("customerID is required for individual customer lookup.")
    elif filtered.empty:
        st.info("No customers match the current filters.")
    else:
        ids = filtered["customerID"].astype(str).tolist()
        customer_id = st.selectbox("Select customer", ids)
        row = scored[scored["customerID"].astype(str) == customer_id].iloc[0]
        raw_row = None
        if customers is not None and "customerID" in customers.columns:
            matches = customers[customers["customerID"].astype(str) == customer_id]
            if not matches.empty:
                raw_row = matches.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Churn probability", f"{row['CalibratedChurnProbability']:.1%}", "Calibrated probability")
        with m2:
            metric_card("Risk level", str(row["CalibratedRiskLevel"]), "Operational risk band")
        with m3:
            metric_card("Revenue at risk", fmt_money(float(row["RevenueAtRisk"])), "Expected monthly exposure")
        with m4:
            metric_card("Retention priority", str(row["RetentionPriority"]), "Revenue-aware ranking")

        profile_left, profile_right = st.columns([1.05, 1])
        with profile_left:
            st.markdown("**Customer profile**")
            profile_fields = [
                ("Contract", "Contract"),
                ("Tenure", "tenure"),
                ("Internet service", "InternetService"),
                ("Monthly charges", "MonthlyCharges"),
                ("Payment method", "PaymentMethod"),
                ("Tech support", "TechSupport"),
                ("Partner", "Partner"),
                ("Dependents", "Dependents"),
            ]
            profile_rows = []
            for label, col in profile_fields:
                value = row.get(col, "—")
                if col == "MonthlyCharges" and pd.notna(value):
                    value = fmt_money(float(value))
                elif col == "tenure" and pd.notna(value):
                    value = f"{int(value)} months"
                profile_rows.append({"Field": label, "Value": str(value)})
            st.dataframe(pd.DataFrame(profile_rows), hide_index=True, width="stretch")

        with profile_right:
            st.markdown("**Retention recommendation**")
            st.info(str(row["RetentionRecommendation"]))
            st.markdown("**Business interpretation**")
            st.write(
                "The score combines calibrated churn probability with normalized revenue exposure. "
                "It is designed for resource allocation, not as a guarantee that a retention action will prevent churn."
            )

        st.markdown("**Why the model is concerned**")
        try:
            if data_source == "MySQL":
                if raw_row is None:
                    raise RuntimeError("Raw customer data is unavailable for a local SHAP explanation in MySQL mode.")
                if not MODEL_PATH.exists():
                    raise RuntimeError("The production model artifact is unavailable for a local SHAP explanation.")
                shap_scorer = CustomerRiskScorer(MODEL_PATH)
                expl_input = prepare_explain_input(pd.DataFrame([raw_row]))
                shap_df = explain_customer(
                    shap_scorer.model,
                    expl_input.drop(columns=["customerID", "Churn"], errors="ignore"),
                    top_n=8,
                )
            else:
                expl_input = prepare_explain_input(pd.DataFrame([raw_row]))
                shap_df = explain_customer(
                    scorer.model,
                    expl_input.drop(columns=["customerID", "Churn"], errors="ignore"),
                    top_n=8,
                )
            shap_df = shap_df.rename(columns={"SHAPValue": "Contribution", "AbsoluteSHAP": "Magnitude"})
            shap_df["Direction"] = shap_df["Contribution"].apply(lambda x: "Raises risk" if x > 0 else "Lowers risk")
            st.dataframe(
                shap_df[["Feature", "Contribution", "Magnitude", "Direction"]].style.format({
                    "Contribution": "{:+.4f}",
                    "Magnitude": "{:.4f}",
                }),
                hide_index=True,
                width="stretch",
            )
            st.caption("SHAP explains the underlying Random Forest output; the displayed churn probability is calibrated separately. In MySQL mode, persisted scores remain the source of displayed risk values.")
        except Exception as exc:
            st.warning(f"Local SHAP explanation is unavailable in this environment: {exc}")

# -----------------------------------------------------------------------------
# TAB 3 — Retention Command Center
# -----------------------------------------------------------------------------
with retention_tab:
    st.markdown('<div class="section-title">Campaign planning</div>', unsafe_allow_html=True)
    cap = st.slider("Campaign capacity", min_value=5, max_value=50, value=10, step=5, format="%d%%")
    budget_count = max(1, round(len(filtered) * cap / 100)) if len(filtered) else 0
    target_pool = filtered.sort_values("RetentionPriorityScore", ascending=False).head(budget_count).copy()

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        metric_card("Campaign capacity", f"{cap}%", f"Top {budget_count:,} customers")
    with r2:
        metric_card("Revenue exposure targeted", fmt_money(float(target_pool["RevenueAtRisk"].sum())) if budget_count else "$0.00", "Within selected capacity")
    with r3:
        metric_card("Average customer exposure", fmt_money(float(target_pool["RevenueAtRisk"].mean())) if budget_count else "$0.00", "Per targeted customer")
    with r4:
        metric_card("High / critical targeted", f"{int(target_pool['RetentionPriority'].astype(str).isin(['High','Critical']).sum()):,}", "Priority tiers")

    st.markdown("**Top intervention queue**")
    queue_cols = [
        "CustomerRank", "customerID", "CalibratedChurnProbability", "CalibratedRiskLevel",
        "MonthlyCharges", "RevenueAtRisk", "RetentionPriorityScore", "RetentionPriority",
        "RetentionRecommendation"
    ]
    queue = target_pool[queue_cols].rename(columns={
        "CustomerRank": "Rank",
        "customerID": "Customer",
        "CalibratedChurnProbability": "Churn probability",
        "CalibratedRiskLevel": "Risk",
        "MonthlyCharges": "Monthly charges",
        "RevenueAtRisk": "Revenue at risk",
        "RetentionPriorityScore": "Priority score",
        "RetentionPriority": "Priority",
        "RetentionRecommendation": "Recommended action",
    })

    # Add an operational action label without changing the research score/model.
    queue["Action"] = queue["Priority"].map({
        "Critical": "Immediate outreach",
        "High": "Priority outreach",
        "Moderate": "Standard outreach",
        "Low": "Monitor",
    }).astype("object").fillna("Review")

    queue_display = queue[
        [
            "Rank", "Customer", "Risk", "Priority", "Churn probability",
            "Monthly charges", "Revenue at risk", "Priority score",
            "Action", "Recommended action"
        ]
    ]

    st.dataframe(
        queue_display.style.format({
            "Churn probability": "{:.1%}",
            "Monthly charges": "${:,.2f}",
            "Revenue at risk": "${:,.2f}",
            "Priority score": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

    q1, q2, q3 = st.columns(3)
    with q1:
        critical_n = int((target_pool["RetentionPriority"].astype(str) == "Critical").sum())
        metric_card("Immediate outreach", f"{critical_n:,}", "Critical priority customers")
    with q2:
        high_n = int(target_pool["RetentionPriority"].astype(str).isin(["High", "Critical"]).sum())
        metric_card("Priority outreach", f"{high_n:,}", "High + Critical customers")
    with q3:
        monitor_n = int((target_pool["RetentionPriority"].astype(str) == "Low").sum())
        metric_card("Monitor", f"{monitor_n:,}", "Low-priority customers in queue")

    csv_bytes = queue_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download retention queue (CSV)",
        data=csv_bytes,
        file_name="retention_intervention_queue.csv",
        mime="text/csv",
    )

    st.markdown("**Priority distribution**")
    priority_order = ["Low", "Moderate", "High", "Critical"]
    priority_counts = (
        filtered["RetentionPriority"].astype(str).value_counts()
        .reindex(priority_order, fill_value=0)
        .rename_axis("Priority")
        .reset_index(name="Customers")
    )
    priority_counts["Priority"] = pd.Categorical(priority_counts["Priority"], categories=priority_order, ordered=True)
    priority_chart = (
        alt.Chart(priority_counts)
        .mark_bar()
        .encode(
            x=alt.X("Priority:N", sort=priority_order, title=None),
            y=alt.Y("Customers:Q", title="Customers"),
            tooltip=[alt.Tooltip("Priority:N", title="Priority"), alt.Tooltip("Customers:Q", title="Customers")],
        )
        .properties(height=280)
    )
    st.altair_chart(priority_chart, width="stretch")
    st.caption("Retention recommendations are rule-based decision-support heuristics, not causal treatment guarantees.")

# -----------------------------------------------------------------------------
# TAB 4 — Research Evidence
# -----------------------------------------------------------------------------
with research_tab:
    st.markdown('<div class="section-title">Established research evidence</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Baseline Logistic ROC-AUC", f"{RESEARCH_METRICS['baseline_logistic_roc_auc']:.4f}", "Baseline model")
    with c2:
        metric_card("Research RF ROC-AUC", f"{RESEARCH_METRICS['research_rf_roc_auc']:.4f}", "Held-out research result")
    with c3:
        metric_card("Calibrated ROC-AUC", f"{RESEARCH_METRICS['roc_auc_calibrated']:.4f}", "After sigmoid calibration")
    with c4:
        improvement = RESEARCH_METRICS["brier_original"] - RESEARCH_METRICS["brier_calibrated"]
        metric_card("Brier improvement", f"{improvement:.4f}", "Lower Brier is better")

    st.markdown("**Key finding**")
    st.markdown(
        '<div class="finding">High-risk customers represent 13.13% of the evaluation population while containing '
        '34.49% of observed churners. Expanding to High + Medium covers 34.92% of customers and captures 71.12% of observed churners.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.write("Risk validation")
        st.dataframe(pd.DataFrame(RISK_VALIDATION).rename(columns={
            "risk": "Risk", "customers": "Customers", "observed_churn_pct": "Observed churn %"
        }).style.format({"Observed churn %": "{:.2f}%"}), width="stretch", hide_index=True)

    with right:
        st.write("Targeting efficiency")
        st.dataframe(pd.DataFrame(TARGETING).rename(columns={
            "target": "Target", "customers": "Customers", "customer_pct": "Customer share %",
            "capture_pct": "Churner capture %", "efficiency": "Efficiency"
        }).style.format({
            "Customer share %": "{:.2f}%", "Churner capture %": "{:.2f}%", "Efficiency": "{:.2f}"
        }), width="stretch", hide_index=True)

    st.write("Retention-priority evidence")
    st.dataframe(pd.DataFrame(RETENTION_PRIORITY).rename(columns={
        "priority": "Priority", "customers": "Customers", "churners": "Churners",
        "churn_rate_pct": "Observed churn %", "revenue_at_risk": "Revenue at risk"
    }).style.format({
        "Observed churn %": "{:.2f}%", "Revenue at risk": "${:,.2f}"
    }), width="stretch", hide_index=True)

    st.write("Probability-quality metrics")
    calibration = pd.DataFrame([
        ["Brier Score", RESEARCH_METRICS["brier_original"], RESEARCH_METRICS["brier_calibrated"]],
        ["ROC-AUC", RESEARCH_METRICS["research_rf_roc_auc"], RESEARCH_METRICS["roc_auc_calibrated"]],
        ["Log Loss", RESEARCH_METRICS["log_loss_original"], RESEARCH_METRICS["log_loss_calibrated"]],
    ], columns=["Metric", "Original", "Calibrated"])
    st.dataframe(calibration.style.format({"Original": "{:.4f}", "Calibrated": "{:.4f}"}), hide_index=True, width="stretch")

    st.markdown(
        '<div class="footer-note">'
        'Research evidence is presented as established project results. '
        'Production refitting is separate from the held-out research evaluation. '
        'Uploaded customer data is validated for schema and customer-level quality '
        'before scoring; validation does not modify research point estimates.'
        '</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# ADD-ON TAB 5 — Deep Learning / Model Comparison
# Reads artifacts produced by scripts/train_deep_learning_model.py. Training is
# deliberately not triggered from the dashboard.
# -----------------------------------------------------------------------------
with model_tab:
    st.markdown('<div class="section-title">Shared-split model comparison</div>', unsafe_allow_html=True)

    # Prefer the newest comparison trained on real data. Synthetic demo runs
    # remain available as a fallback, but should not silently replace the real
    # IBM Telco experiment in the main dashboard.
    real_data_comparisons = []
    synthetic_comparisons = []

    for candidate in (ROOT / "models").glob("*/comparison_metrics.json"):
        try:
            candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
            candidate_dataset = candidate_data.get("dataset") or {}
            if not isinstance(candidate_dataset, dict):
                candidate_dataset = {}

            candidate_data_path = str(candidate_dataset.get("path") or "").replace("\\", "/").lower()
            candidate_text_path = str(candidate_dataset.get("text_features_path") or "").replace("\\", "/").lower()
            is_synthetic_candidate = (
                "data/demo/" in candidate_data_path
                or "customer_text_features_demo.csv" in candidate_text_path
            )

            if is_synthetic_candidate:
                synthetic_comparisons.append(candidate)
            else:
                real_data_comparisons.append(candidate)
        except (OSError, json.JSONDecodeError, TypeError):
            # Ignore unreadable experiment files and continue with valid ones.
            continue

    preferred_comparisons = real_data_comparisons or synthetic_comparisons
    comparison_path = max(
        preferred_comparisons,
        key=lambda candidate: candidate.stat().st_mtime,
        default=ROOT / "models" / "deep_learning" / "comparison_metrics.json",
    )

    if not comparison_path.exists():
        st.info(
            "No Deep Learning comparison has been saved yet. Install requirements-addon.txt, "
            "then run `python -m scripts.train_deep_learning_model` from the project root."
        )
        st.caption("This add-on does not change the existing production Random Forest scoring path.")
    else:
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            st.caption(
                f"Metrics loaded from: {comparison_path.parent.relative_to(ROOT)}"
            )
            metrics = comparison.get("metrics", {})
            dataset_info = comparison.get("dataset", {})
            dataset_path_label = str(dataset_info.get("path") or "")
            text_path_label = str(dataset_info.get("text_features_path") or "")
            normalized_dataset_path = dataset_path_label.replace("\\", "/").lower()

            if (
                "data/demo/" in normalized_dataset_path
                or "customer_text_features_demo.csv" in text_path_label.lower()
            ):
                st.warning(
                    "SYNTHETIC DEMO EXPERIMENT: This comparison uses simulated customer data "
                    "and/or precomputed synthetic text features. It verifies the workflow only; "
                    "it is not evidence of real-world predictive performance or an NLP-driven churn lift."
                )

            if "customer_text_features_demo.csv" in text_path_label:
                st.warning(
                    "The hybrid comparison uses precomputed synthetic sentiment features, not outputs from the Transformer NLP pipeline. "
                    "Use real, linked support text and run the NLP script before drawing conclusions about NLP value."
                )

            if not metrics:
                st.warning("The comparison file exists but contains no metrics. Re-run the training script.")
            else:
                st.caption(
                    "These add-on metrics use a new held-out split. They are separate from the frozen research results "
                    "shown in the Research Evidence tab."
                )
                protocol = comparison.get("evaluation_protocol", {})
                if protocol:
                    st.caption(
                        f"Rows: {protocol.get('train_rows', '—'):,} train · "
                        f"{protocol.get('validation_rows', '—'):,} validation · "
                        f"{protocol.get('test_rows', '—'):,} test · "
                        f"random_state={protocol.get('random_state', '—')}"
                    )

                metric_rows = []
                for model_name, values in metrics.items():
                    metric_rows.append({
                        "Model": model_name,
                        "ROC-AUC": values.get("roc_auc"),
                        "PR-AUC": values.get("pr_auc"),
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1": values.get("f1"),
                        "Brier score ↓": values.get("brier_score"),
                        "Log loss ↓": values.get("log_loss"),
                    })

                metric_df = pd.DataFrame(metric_rows)
                st.dataframe(
                    metric_df.style.format({
                        "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "Precision": "{:.4f}",
                        "Recall": "{:.4f}", "F1": "{:.4f}", "Brier score ↓": "{:.4f}",
                        "Log loss ↓": "{:.4f}",
                    }, na_rep="—"),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("Higher ROC-AUC / PR-AUC / F1 may be useful; lower Brier score and log loss are better. Review trade-offs rather than selecting by one metric.")

                roc_frames = []
                for model_name, curve in comparison.get("roc_curves", {}).items():
                    fprs, tprs = curve.get("fpr", []), curve.get("tpr", [])
                    for fpr_value, tpr_value in zip(fprs, tprs):
                        roc_frames.append({
                            "Model": model_name,
                            "False positive rate": fpr_value,
                            "True positive rate": tpr_value,
                        })

                if roc_frames:
                    st.write("ROC curves")
                    roc_chart = alt.Chart(pd.DataFrame(roc_frames)).mark_line().encode(
                        x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Model:N"),
                        tooltip=["Model:N", "False positive rate:Q", "True positive rate:Q"],
                    ).properties(height=300)
                    st.altair_chart(roc_chart, width="stretch")

                pr_frames = []
                for model_name, curve in comparison.get("precision_recall_curves", {}).items():
                    recalls, precisions = curve.get("recall", []), curve.get("precision", [])
                    for recall_value, precision_value in zip(recalls, precisions):
                        pr_frames.append({
                            "Model": model_name,
                            "Recall": recall_value,
                            "Precision": precision_value,
                        })

                if pr_frames:
                    st.write("Precision-recall curves")
                    pr_chart = alt.Chart(pd.DataFrame(pr_frames)).mark_line().encode(
                        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Model:N"),
                        tooltip=["Model:N", "Recall:Q", "Precision:Q"],
                    ).properties(height=300)
                    st.altair_chart(pr_chart, width="stretch")

                calibration_frames = []
                for model_name, curve in comparison.get("calibration_curves", {}).items():
                    predicted_values = curve.get("mean_predicted_probability", [])
                    observed_values = curve.get("observed_positive_rate", [])
                    for predicted_value, observed_value in zip(predicted_values, observed_values):
                        calibration_frames.append({
                            "Model": model_name,
                            "Mean predicted probability": predicted_value,
                            "Observed positive rate": observed_value,
                        })

                if calibration_frames:
                    st.write("Probability calibration")
                    calibration_df = pd.DataFrame(calibration_frames)
                    ideal_df = pd.DataFrame({
                        "Model": ["Perfect calibration", "Perfect calibration"],
                        "Mean predicted probability": [0.0, 1.0],
                        "Observed positive rate": [0.0, 1.0],
                    })
                    calibration_chart = alt.Chart(
                        pd.concat([calibration_df, ideal_df], ignore_index=True)
                    ).mark_line(point=True).encode(
                        x=alt.X("Mean predicted probability:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("Observed positive rate:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Model:N"),
                        tooltip=["Model:N", "Mean predicted probability:Q", "Observed positive rate:Q"],
                    ).properties(height=300)
                    st.altair_chart(calibration_chart, width="stretch")

        except Exception as exc:
            st.error(f"Could not read the Deep Learning comparison file: {exc}")

# -----------------------------------------------------------------------------
# ADD-ON TAB 6 — Customer Voice / NLP
# Reads aggregated data from scripts/analyze_support_text.py. Raw ticket text is
# not rendered in this dashboard tab.
# -----------------------------------------------------------------------------
with voice_tab:
    st.markdown('<div class="section-title">Customer voice and support sentiment</div>', unsafe_allow_html=True)

    # Prefer the dedicated master-dataset output so the existing demo NLP
    # artifacts remain untouched. Fall back to the prior customer_voice folder.
    voice_path = (
        ROOT / "data" / "processed" / "customer_voice_master" / "customer_text_features.csv"
    )
    using_precomputed_synthetic_voice_demo = False

    if not voice_path.exists():
        previous_voice_path = (
            ROOT / "data" / "processed" / "customer_voice" / "customer_text_features.csv"
        )
        if previous_voice_path.exists():
            voice_path = previous_voice_path

    if not voice_path.exists():
        demo_voice_path = ROOT / "data" / "demo" / "customer_text_features_demo.csv"
        if demo_voice_path.exists():
            voice_path = demo_voice_path
            using_precomputed_synthetic_voice_demo = True

    if not voice_path.exists():
        st.info(
            "NLP aggregates are not available yet. Add a customer support CSV at "
            "`data/raw/customer_support_text.csv` with customerID, ticket_text, and ideally ticket_date, "
            "then run `python -m scripts.analyze_support_text`."
        )
        st.caption(
            "The default sentiment model is general English sentiment, not telecom-specific. "
            "Validate it on manually labelled support messages before operational use."
        )
    else:
        try:
            voice_df = pd.read_csv(voice_path)
            required_voice = {
                "customerID", "ticket_count", "negative_ticket_count", "negative_ticket_ratio",
                "mean_sentiment_score", "latest_sentiment_score"
            }

            missing_voice = sorted(required_voice - set(voice_df.columns))

            if missing_voice:
                st.warning(f"The NLP aggregate file is missing expected columns: {missing_voice}")
            elif voice_df.empty:
                st.info("The customer-voice aggregate file has no rows.")
            else:
                customer_ids = voice_df["customerID"].dropna().astype(str).str.upper()
                synthetic_ticket_ids = (
                    not customer_ids.empty
                    and customer_ids.str.startswith(("DEMO", "SYNTH")).any()
                )

                if using_precomputed_synthetic_voice_demo:
                    st.warning(
                        "SYNTHETIC DEMO FEATURES: This view is using precomputed simulated "
                        "customer aggregates. They were not generated by running the Transformer "
                        "on real customer messages and must not be treated as real customer evidence."
                    )
                elif synthetic_ticket_ids:
                    st.warning(
                        "SYNTHETIC CUSTOMER VOICE INPUT: The Transformer sentiment model was run "
                        "on generated support messages for synthetic customers. The displayed scores "
                        "are model predictions on synthetic text, not genuine customer feedback or "
                        "evidence of churn improvement."
                    )

                total_tickets = int(
                    pd.to_numeric(
                        voice_df["ticket_count"],
                        errors="coerce",
                    ).fillna(0).sum()
                )

                avg_negative = float(
                    pd.to_numeric(
                        voice_df["negative_ticket_ratio"],
                        errors="coerce",
                    ).fillna(0).mean()
                )

                sentiment_values = pd.to_numeric(
                    voice_df["mean_sentiment_score"],
                    errors="coerce",
                ).dropna()

                mean_sentiment = (
                    float(sentiment_values.mean())
                    if not sentiment_values.empty
                    else float("nan")
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "Customers with tickets",
                        f"{voice_df['customerID'].nunique():,}",
                        "Customer-level aggregates",
                    )

                with c2:
                    metric_card(
                        "Support interactions",
                        f"{total_tickets:,}",
                        "Messages included in this export",
                    )

                with c3:
                    metric_card(
                        "Average negative-ticket ratio",
                        f"{avg_negative:.1%}",
                        "Model-derived indicator, not ground truth",
                    )

                st.caption(
                    f"Mean aggregate sentiment score: {mean_sentiment:.3f} "
                    "(range −1 to +1 where available)."
                )

                hist_data = voice_df[["negative_ticket_ratio"]].copy()
                hist_data["negative_ticket_ratio"] = pd.to_numeric(
                    hist_data["negative_ticket_ratio"],
                    errors="coerce",
                )
                hist_data = hist_data.dropna()

                if not hist_data.empty:
                    st.write("Distribution of negative-ticket ratios")
                    histogram = alt.Chart(hist_data).mark_bar().encode(
                        x=alt.X(
                            "negative_ticket_ratio:Q",
                            bin=alt.Bin(maxbins=20),
                            title="Negative-ticket ratio",
                        ),
                        y=alt.Y("count():Q", title="Customers"),
                        tooltip=[
                            alt.Tooltip("count():Q", title="Customers")
                        ],
                    ).properties(height=260)

                    st.altair_chart(
                        histogram,
                        width="stretch",
                    )

                st.write("Customers with highest negative-ticket ratio")

                display_cols = [
                    "customerID", "ticket_count", "negative_ticket_count",
                    "negative_ticket_ratio", "mean_sentiment_score",
                    "latest_sentiment_score",
                ]

                top_voice = voice_df.sort_values(
                    ["negative_ticket_ratio", "ticket_count"],
                    ascending=[False, False],
                )[display_cols].head(20)

                st.dataframe(
                    top_voice.style.format({
                        "negative_ticket_ratio": "{:.1%}",
                        "mean_sentiment_score": "{:.3f}",
                        "latest_sentiment_score": "{:.3f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Use these signals for review and prioritization only. "
                    "Automated sentiment may be wrong and should not be the "
                    "sole basis for an individual customer decision."
                )

        except Exception as exc:
            st.error(f"Could not read the customer-voice features: {exc}")

    render_customer_voice_risk_section(ROOT)
