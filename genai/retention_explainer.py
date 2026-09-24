"""Optional Hugging Face explanation layer with deterministic fallback.

The tabular model remains the source of churn probability/risk. This module only
translates already-computed signals into natural-language decision support.
"""
from __future__ import annotations

import os
from typing import Any


def _fallback(profile: dict[str, Any]) -> str:
    priority = profile.get("RetentionPriority", "Unknown")
    risk = profile.get("CalibratedRiskLevel", "Unknown")
    prob = float(profile.get("CalibratedChurnProbability", 0.0)) * 100
    revenue = float(profile.get("RevenueAtRisk", 0.0))
    rec = profile.get("RetentionRecommendation", "Review customer profile.")
    return (
        f"Risk assessment: {risk} risk with calibrated churn probability {prob:.1f}%. "
        f"Retention priority: {priority}. Expected monthly revenue exposure is ${revenue:,.2f}. "
        f"Recommended action: {rec} This is decision support, not a causal claim that the action will prevent churn."
    )


def generate_explanation(profile: dict[str, Any]) -> str:
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    if not token:
        return _fallback(profile)

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        prompt = (
            "You are a telecom retention analyst. Convert the following model output into a concise, "
            "business-safe explanation. Do not claim causality, certainty, or that a recommendation will "
            "prevent churn. State that the probability is calibrated and that revenue at risk is expected exposure.\n\n"
            f"Customer profile: {profile}\n\n"
            "Return: risk summary, key business signals, recommended next action."
        )
        result = client.text_generation(prompt, model=model, max_new_tokens=220, temperature=0.2)
        return result.strip() if isinstance(result, str) else _fallback(profile)
    except Exception:
        return _fallback(profile)
