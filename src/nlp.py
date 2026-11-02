"""NLP add-on for customer-support text.

The default sentiment checkpoint is a general English SST-2 model, not a
telecommunications-specific model. Validate it on a manually labelled sample
before using its sentiment outputs in a business decision or churn model.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ID_COLUMN = "customerID"
TEXT_COLUMN = "ticket_text"
DATE_COLUMN = "ticket_date"


class SupportSentimentAnalyzer:
    """Batched sentiment inference using a Hugging Face sequence classifier."""

    def __init__(
        self,
        model_name: str = DEFAULT_SENTIMENT_MODEL,
        batch_size: int = 16,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The NLP add-on requires PyTorch and Transformers. Install with: "
                "pip install -r requirements-addon.txt"
            ) from exc

        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        label_map = {int(index): str(label).lower() for index, label in self.model.config.id2label.items()}
        positives = [index for index, label in label_map.items() if "positive" in label or label == "pos"]
        negatives = [index for index, label in label_map.items() if "negative" in label or label == "neg"]
        if not positives or not negatives:
            raise ValueError(
                f"Model {model_name!r} must expose positive and negative labels; got {label_map}. "
                "Use a binary sentiment checkpoint with known label semantics."
            )
        self.positive_index = positives[0]
        self.negative_index = negatives[0]

    def predict(self, texts: Sequence[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(texts)
        valid_pairs = [
            (index, text.strip())
            for index, text in enumerate(texts)
            if isinstance(text, str) and text.strip()
        ]
        for start in range(0, len(valid_pairs), self.batch_size):
            batch = valid_pairs[start : start + self.batch_size]
            batch_texts = [text for _, text in batch]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for (original_index, _), row in zip(batch, probabilities):
                positive_probability = float(row[self.positive_index])
                negative_probability = float(row[self.negative_index])
                score = positive_probability - negative_probability
                results[original_index] = {
                    "sentiment_label": "POSITIVE" if score >= 0 else "NEGATIVE",
                    "sentiment_confidence": float(max(positive_probability, negative_probability)),
                    "positive_probability": positive_probability,
                    "negative_probability": negative_probability,
                    "sentiment_score": float(score),
                }
        for index, value in enumerate(results):
            if value is None:
                results[index] = {
                    "sentiment_label": "UNKNOWN",
                    "sentiment_confidence": np.nan,
                    "positive_probability": np.nan,
                    "negative_probability": np.nan,
                    "sentiment_score": np.nan,
                }
        return [value for value in results if value is not None]


def discover_topics(texts: Sequence[Any], n_topics: int = 5, random_state: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    """Discover exploratory complaint themes with TF-IDF + NMF.

    Topic IDs are corpus-specific and do not carry fixed business meaning.
    Review the top words and example tickets before assigning business labels.
    """
    usable_positions = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
    usable_texts = [str(texts[i]).strip() for i in usable_positions]
    if len(usable_texts) < 2:
        raise ValueError("At least two non-empty support messages are required for topic discovery.")
    min_df = 2 if len(usable_texts) >= 20 else 1
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.98 if len(usable_texts) >= 10 else 1.0,
    )
    matrix = vectorizer.fit_transform(usable_texts)
    if matrix.shape[1] == 0:
        raise ValueError("No usable vocabulary found in support text after preprocessing.")
    component_count = max(1, min(int(n_topics), len(usable_texts), matrix.shape[1]))
    init = "nndsvda" if component_count <= min(matrix.shape) else "random"
    model = NMF(n_components=component_count, init=init, random_state=random_state, max_iter=400)
    document_topic = model.fit_transform(matrix)
    local_assignments = document_topic.argmax(axis=1)
    assignments = np.full(len(texts), -1, dtype=int)
    assignments[np.asarray(usable_positions, dtype=int)] = local_assignments

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    topic_info = []
    for topic_index, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:8]
        keywords = [str(feature_names[index]) for index in top_indices if component[index] > 0]
        topic_info.append({
            "topic_id": int(topic_index),
            "topic_label": " / ".join(keywords[:4]) or f"Topic {topic_index + 1}",
            "keywords": keywords,
            "message_count": int((local_assignments == topic_index).sum()),
        })
    return assignments, {
        "vectorizer": vectorizer,
        "model": model,
        "topic_info": topic_info,
        "random_state": random_state,
    }


def aggregate_customer_features(
    scored_tickets: pd.DataFrame,
    id_column: str = ID_COLUMN,
    date_column: str | None = DATE_COLUMN,
) -> pd.DataFrame:
    """Aggregate ticket-level sentiment into one row per customer."""
    required = {id_column, "sentiment_label", "sentiment_score"}
    missing = sorted(required - set(scored_tickets.columns))
    if missing:
        raise ValueError(f"Scored tickets are missing required columns: {missing}")
    usable = scored_tickets[scored_tickets[id_column].notna()].copy()
    if usable.empty:
        raise ValueError("No support tickets have a usable customer ID.")
    usable["sentiment_score"] = pd.to_numeric(usable["sentiment_score"], errors="coerce")
    usable["_is_negative"] = usable["sentiment_label"].astype(str).str.upper().eq("NEGATIVE").astype(int)
    records = []
    has_date = bool(date_column and date_column in usable.columns)
    if has_date:
        usable[date_column] = pd.to_datetime(usable[date_column], errors="coerce", utc=True)
    for customer_id, group in usable.groupby(id_column, sort=False, dropna=True):
        valid_scores = group["sentiment_score"].dropna()
        record: dict[str, Any] = {
            id_column: customer_id,
            "ticket_count": int(len(group)),
            "negative_ticket_count": int(group["_is_negative"].sum()),
            "negative_ticket_ratio": float(group["_is_negative"].mean()),
            "mean_sentiment_score": float(valid_scores.mean()) if len(valid_scores) else 0.0,
            "latest_sentiment_score": 0.0,
        }
        if has_date:
            dated = group.dropna(subset=[date_column]).sort_values(date_column)
            if not dated.empty:
                latest = dated.iloc[-1]
                latest_score = pd.to_numeric(latest["sentiment_score"], errors="coerce")
                record["latest_sentiment_score"] = float(latest_score) if pd.notna(latest_score) else 0.0
                record["latest_ticket_date"] = latest[date_column].isoformat()
            else:
                record["latest_ticket_date"] = None
        else:
            record["latest_sentiment_score"] = float(valid_scores.iloc[-1]) if len(valid_scores) else 0.0
        records.append(record)
    return pd.DataFrame(records)


def analyze_support_tickets(
    tickets: pd.DataFrame,
    id_column: str = ID_COLUMN,
    text_column: str = TEXT_COLUMN,
    date_column: str | None = DATE_COLUMN,
    analyzer: SupportSentimentAnalyzer | None = None,
    n_topics: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """Score support messages and produce customer-level NLP features."""
    missing = sorted({id_column, text_column} - set(tickets.columns))
    if missing:
        raise ValueError(f"Support dataset is missing required columns: {missing}")
    if tickets[id_column].isna().any():
        raise ValueError(f"Support dataset contains missing {id_column} values.")
    scored = tickets.copy()
    sentiment_analyzer = analyzer or SupportSentimentAnalyzer()
    predictions = sentiment_analyzer.predict(scored[text_column].tolist())
    sentiment_frame = pd.DataFrame(predictions, index=scored.index)
    for column in sentiment_frame.columns:
        scored[column] = sentiment_frame[column]

    topic_artifacts = None
    if n_topics > 0:
        assignments, topic_artifacts = discover_topics(scored[text_column].tolist(), n_topics=n_topics)
        topic_lookup = {item["topic_id"]: item["topic_label"] for item in topic_artifacts["topic_info"]}
        scored["topic_id"] = [int(value) if value >= 0 else -1 for value in assignments]
        scored["topic_label"] = [topic_lookup.get(int(value), "UNASSIGNED") if value >= 0 else "UNASSIGNED" for value in assignments]

    used_date_column = date_column if date_column and date_column in scored.columns else None
    customer_features = aggregate_customer_features(scored, id_column=id_column, date_column=used_date_column)
    return scored, customer_features, topic_artifacts
