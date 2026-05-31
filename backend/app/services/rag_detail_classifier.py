from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import re

from typing import Sequence

from app.core.config import settings
from app.services.domain_taxonomy import CANONICAL_DETAIL_LABELS


RAG_DETAIL_LABELS: tuple[str, ...] = CANONICAL_DETAIL_LABELS
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagDetailPrediction:
    detail: str
    score: float


def classify_rag_detail_with_klue_bert(user_input: str) -> RagDetailPrediction | None:
    predictions = classify_rag_details_with_klue_bert(user_input)
    return predictions[0] if predictions else None


def classify_rag_details_with_klue_bert(user_input: str) -> tuple[RagDetailPrediction, ...]:
    if not settings.rag_detail_classifier_model_name:
        return ()

    classifier = _get_classifier()
    if classifier is None:
        return ()

    output = classifier(user_input, truncation=True)
    predictions = _predictions_from_output(output)
    threshold = settings.rag_detail_classifier_confidence_threshold
    top_k = max(settings.rag_detail_classifier_top_k, 1)
    return tuple(
        prediction
        for prediction in sorted(predictions, key=lambda item: item.score, reverse=True)
        if prediction.score >= threshold
    )[:top_k]


def _prediction_from_output(output: object) -> RagDetailPrediction | None:
    predictions = _predictions_from_output(output)
    return max(predictions, key=lambda item: item.score) if predictions else None


def _predictions_from_output(output: object) -> list[RagDetailPrediction]:
    if isinstance(output, list):
        if not output:
            return []
        if isinstance(output[0], list):
            return _predictions_from_output(output[0])
        rows = output
    else:
        rows = [output]

    predictions: list[RagDetailPrediction] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _normalize_label(str(row.get("label", "")))
        if label not in RAG_DETAIL_LABELS:
            continue
        predictions.append(RagDetailPrediction(detail=label, score=round(float(row.get("score", 0.0)), 6)))
    return predictions


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("label_", "")
    try:
        label_index = int(normalized)
    except ValueError:
        return normalized
    if 0 <= label_index < len(RAG_DETAIL_LABELS):
        return RAG_DETAIL_LABELS[label_index]
    return normalized


@lru_cache(maxsize=1)
def _get_classifier():
    try:
        from transformers import pipeline
    except ImportError:
        return None

    model_name = _resolve_local_model_name(settings.rag_detail_classifier_model_name)
    try:
        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=settings.rag_detail_classifier_device,
            top_k=None,
            function_to_apply="sigmoid",
        )
    except Exception as exc:
        logger.warning("RAG detail classifier unavailable: %s", exc)
        return None


def _resolve_local_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return model_name
    path = Path(model_name)
    if path.exists():
        return model_name
    if path.parent == Path("models"):
        base_name = re.sub(r"-v\d+$", "", path.name)
        candidates = sorted(path.parent.glob(f"{base_name}-v*"))
        if candidates:
            return str(candidates[-1])
    return model_name


def labels_to_multihot(details: Sequence[str]) -> list[float]:
    detail_set = set(details)
    return [1.0 if detail in detail_set else 0.0 for detail in RAG_DETAIL_LABELS]
