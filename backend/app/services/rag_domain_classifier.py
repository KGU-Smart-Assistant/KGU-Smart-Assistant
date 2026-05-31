from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import Sequence

from app.core.config import settings
from app.services.domain_taxonomy import CANONICAL_DOMAIN_LABELS, normalize_domain


RAG_DOMAIN_LABELS: tuple[str, ...] = tuple(
    domain for domain in CANONICAL_DOMAIN_LABELS if domain != "unknown"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagDomainPrediction:
    domain: str
    score: float


def classify_rag_domains_with_klue_bert(user_input: str) -> tuple[RagDomainPrediction, ...]:
    """Return top-k RAG domain predictions from a multi-label KLUE-BERT model.

    The model is optional. When it is not configured or dependencies are not
    installed, callers receive an empty tuple. RAG domain classification then
    becomes unknown rather than falling back to keyword rules.
    """
    if not settings.rag_domain_classifier_model_name:
        return ()

    classifier = _get_classifier()
    if classifier is None:
        return ()

    output = classifier(user_input, truncation=True)
    predictions = _predictions_from_output(output)
    threshold = settings.rag_domain_classifier_confidence_threshold
    top_k = max(settings.rag_domain_classifier_top_k, 1)
    return tuple(
        prediction
        for prediction in sorted(predictions, key=lambda item: item.score, reverse=True)
        if prediction.score >= threshold
    )[:top_k]


def _predictions_from_output(output: object) -> list[RagDomainPrediction]:
    if isinstance(output, list):
        if output and isinstance(output[0], list):
            return _predictions_from_output(output[0])
        rows = output
    else:
        rows = [output]

    predictions: list[RagDomainPrediction] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = normalize_domain(_normalize_label(str(row.get("label", ""))))
        if label not in RAG_DOMAIN_LABELS:
            continue
        predictions.append(
            RagDomainPrediction(
                domain=label,
                score=round(float(row.get("score", 0.0)), 6),
            )
        )
    return predictions


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("label_", "")
    try:
        label_index = int(normalized)
    except ValueError:
        return normalized
    if 0 <= label_index < len(RAG_DOMAIN_LABELS):
        return RAG_DOMAIN_LABELS[label_index]
    return normalized


@lru_cache(maxsize=1)
def _get_classifier():
    try:
        from transformers import pipeline
    except ImportError:
        return None

    model_name = _resolve_local_model_name(settings.rag_domain_classifier_model_name)
    try:
        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=settings.rag_domain_classifier_device,
            top_k=None,
            function_to_apply="sigmoid",
        )
    except Exception as exc:
        logger.warning("RAG domain classifier unavailable: %s", exc)
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


def labels_to_multihot(domains: Sequence[str]) -> list[float]:
    domain_set = set(domains)
    return [1.0 if domain in domain_set else 0.0 for domain in RAG_DOMAIN_LABELS]
