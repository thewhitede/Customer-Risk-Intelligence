"""Small Streamlit component for the synthetic Customer Voice × Risk experiment."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt


def render_customer_voice_risk_section(project_root: Path) -> None:
    output_dir = Path(project_root) / "data" / "processed" / "customer_voice_risk"
    joined_path = output_dir / "customer_voice_risk_joined.csv"
    churn_path = output_dir / "voice_risk_summary_by_churn.csv"
    corr_path = output_dir / "voice_risk_correlations.csv"
    summary_path = output_dir / "run_summary.json"

    st.markdown("### Experimental analysis: Customer Voice × Churn Risk")
    st.warning(
        "SYNTHETIC-ONLY EXPERIMENT: This analysis links generated support messages to generated "
        "customer records. It is for pipeline demonstration and descriptive exploration only. "
        "It is not real-customer evidence, a valid holdout evaluation, or proof that NLP improves churn prediction."
    )

    if not joined_path.exists():
        st.info(
            "Run `python -m scripts.analyze_customer_voice_risk` from the project root "
            "to generate the joined customer-level analysis."
        )
        return

    try:
        joined = pd.read_csv(joined_path)
        if joined.empty:
            st.info("The joined Customer Voice × Risk file contains no rows.")
            return
        churn_summary = pd.read_csv(churn_path) if churn_path.exists() else pd.DataFrame()
        correlations = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()

        a, b, c, d = st.columns(4)
        a.metric("Synthetic customers joined", f"{joined['customerID'].nunique():,}")
        b.metric("Support tickets", f"{int(joined['ticket_count'].sum()):,}")
        c.metric("Mean negative-ticket ratio", f"{joined['negative_ticket_ratio'].mean():.1%}")
        d.metric("Mean existing model risk", f"{joined['CalibratedChurnProbability'].mean():.1%}")

        if not churn_summary.empty:
            st.markdown("**Sentiment and model-risk summary by synthetic churn label**")
            st.dataframe(
                churn_summary.style.format({
                    "mean_ticket_count": "{:.2f}",
                    "mean_negative_ticket_ratio": "{:.1%}",
                    "median_negative_ticket_ratio": "{:.1%}",
                    "mean_sentiment_score": "{:.3f}",
                    "mean_calibrated_churn_probability": "{:.1%}",
                    "high_risk_customer_pct": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            chart = alt.Chart(churn_summary).mark_bar().encode(
                x=alt.X("Churn:N", title="Synthetic churn label", sort=["No", "Yes"]),
                y=alt.Y("mean_negative_ticket_ratio:Q", title="Mean negative-ticket ratio", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("Churn:N", title="Synthetic churn label"),
                    alt.Tooltip("customers:Q", title="Customers"),
                    alt.Tooltip("mean_negative_ticket_ratio:Q", title="Mean negative-ticket ratio", format=".1%"),
                    alt.Tooltip("mean_calibrated_churn_probability:Q", title="Mean existing model risk", format=".1%"),
                ],
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        if not correlations.empty:
            st.markdown("**Exploratory correlations**")
            st.dataframe(
                correlations.style.format({"correlation": "{:.4f}"}, na_rep="Not available"),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The Random Forest was loaded read-only and was not retrained. Correlations are descriptive, "
            "and synthetic labels/messages are not independent real-world validation data."
        )
        st.download_button(
            "Download joined Customer Voice × Risk data",
            data=joined.to_csv(index=False).encode("utf-8"),
            file_name="customer_voice_risk_joined.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Could not load Customer Voice × Risk analysis: {exc}")
