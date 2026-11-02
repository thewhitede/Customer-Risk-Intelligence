import pandas as pd

from src.nlp import aggregate_customer_features, analyze_support_tickets, discover_topics


class FakeSentimentAnalyzer:
    def predict(self, texts):
        rows = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                rows.append({
                    "sentiment_label": "UNKNOWN",
                    "sentiment_confidence": float("nan"),
                    "positive_probability": float("nan"),
                    "negative_probability": float("nan"),
                    "sentiment_score": float("nan"),
                })
            elif any(word in text.lower() for word in ["bad", "problem", "charged twice"]):
                rows.append({
                    "sentiment_label": "NEGATIVE",
                    "sentiment_confidence": 0.9,
                    "positive_probability": 0.1,
                    "negative_probability": 0.9,
                    "sentiment_score": -0.8,
                })
            else:
                rows.append({
                    "sentiment_label": "POSITIVE",
                    "sentiment_confidence": 0.9,
                    "positive_probability": 0.9,
                    "negative_probability": 0.1,
                    "sentiment_score": 0.8,
                })
        return rows


def test_customer_feature_aggregation():
    scored = pd.DataFrame({
        "customerID": ["C1", "C1", "C2"],
        "ticket_date": ["2026-01-01", "2026-01-05", "2026-01-03"],
        "sentiment_label": ["POSITIVE", "NEGATIVE", "NEGATIVE"],
        "sentiment_score": [0.8, -0.8, -0.7],
    })
    result = aggregate_customer_features(scored)
    c1 = result.set_index("customerID").loc["C1"]
    assert c1["ticket_count"] == 2
    assert c1["negative_ticket_count"] == 1
    assert c1["negative_ticket_ratio"] == 0.5
    assert c1["latest_sentiment_score"] == -0.8
    assert len(result) == 2


def test_analyze_support_tickets_uses_injected_analyzer():
    tickets = pd.DataFrame({
        "customerID": ["C1", "C1", "C2"],
        "ticket_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "ticket_text": ["The support was good", "I was charged twice", "Bad connection"],
    })
    scored, features, topics = analyze_support_tickets(tickets, analyzer=FakeSentimentAnalyzer())
    assert list(scored["sentiment_label"]) == ["POSITIVE", "NEGATIVE", "NEGATIVE"]
    assert features["customerID"].nunique() == 2
    assert topics is None


def test_topic_discovery_returns_assignments_and_keywords():
    texts = [
        "billing invoice wrong charge",
        "charged twice billing invoice",
        "internet connection slow network",
        "network internet keeps dropping",
        "cancel service contract billing",
        "contract cancellation account close",
    ]
    assignments, artifacts = discover_topics(texts, n_topics=2, random_state=42)
    assert len(assignments) == len(texts)
    assert len(artifacts["topic_info"]) == 2
    assert all(topic["keywords"] for topic in artifacts["topic_info"])
