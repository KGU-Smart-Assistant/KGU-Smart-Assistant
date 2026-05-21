from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from app.core.config import settings

if TYPE_CHECKING:
    from app.services.chat_orchestrator import ChatDecision, ChatRoute, DbIntent


ClassifierRoute = Literal["llm", "relational_db", "rag", "weather"]
ClassifierDbIntent = Literal["map", "phone", "unknown"]

_ROUTES: set[str] = {"llm", "relational_db", "rag", "weather"}
_DB_INTENTS: set[str] = {"map", "phone", "unknown"}

_LEGACY_LABELS: dict[str, tuple[ClassifierRoute, ClassifierDbIntent]] = {
    "llm": ("llm", "unknown"),
    "general": ("llm", "unknown"),
    "db": ("relational_db", "unknown"),
    "map": ("relational_db", "map"),
    "phone": ("relational_db", "phone"),
    "rag": ("rag", "unknown"),
    "weather": ("weather", "unknown"),
}


@dataclass(frozen=True)
class IntentClassifierPrediction:
    route: ClassifierRoute
    db_intent: ClassifierDbIntent = "unknown"
    confidence: float = 0.0
    label: str = ""


def classify_with_klue_bert(user_input: str) -> IntentClassifierPrediction | None:
    """Classify chat route with a fine-tuned KLUE-BERT model when configured."""
    if not settings.intent_classifier_model_name:
        return None

    classifier = _get_classifier()
    if classifier is None:
        return None

    output = classifier(user_input, truncation=True)
    if isinstance(output, list):
        if not output:
            return None
        output = output[0]
        if isinstance(output, list):
            if not output:
                return None
            output = max(output, key=lambda item: float(item.get("score", 0.0)))

    label = str(output.get("label", "")).lower()
    score = float(output.get("score", 0.0))
    route_payload = _parse_classifier_label(label)
    if route_payload is None:
        return None

    route, db_intent = route_payload
    return IntentClassifierPrediction(
        route=route,
        db_intent=db_intent,
        confidence=score,
        label=label,
    )


def _parse_classifier_label(label: str) -> tuple[ClassifierRoute, ClassifierDbIntent] | None:
    """Map model labels into the public route/db_intent decision shape."""
    normalized = label.strip().lower().replace("__", ":").replace("/", ":")
    if normalized in _LEGACY_LABELS:
        return _LEGACY_LABELS[normalized]

    route, separator, db_intent = normalized.partition(":")
    if route not in _ROUTES:
        return None
    if route != "relational_db":
        return route, "unknown"
    if not separator:
        return "relational_db", "unknown"
    if db_intent not in _DB_INTENTS:
        return None
    return "relational_db", db_intent


@lru_cache(maxsize=1)
def _get_classifier():
    try:
        from transformers import pipeline
    except ImportError:
        return None

    return pipeline(
        "text-classification",
        model=settings.intent_classifier_model_name,
        tokenizer=settings.intent_classifier_model_name,
        device=settings.intent_classifier_device,
        top_k=None,
    )
