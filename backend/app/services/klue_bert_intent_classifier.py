from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal

from app.core.config import settings

if TYPE_CHECKING:
    from app.services.chat_orchestrator import ChatRoute, DbIntent


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
logger = logging.getLogger(__name__)


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
    """Map model labels into route-only decisions.

    Older route-only models may emit relational_db without a subtype. Newer
    routing models can emit map/phone or relational_db:* labels so DB subtype
    selection stays in the learned classifier path.
    """
    normalized = label.strip().lower().replace("__", ":").replace("/", ":")
    if normalized in _LEGACY_LABELS:
        return _LEGACY_LABELS[normalized]

    route, separator, db_intent = normalized.partition(":")
    if route not in _ROUTES:
        return None
    if route == "relational_db" and separator:
        if db_intent not in _DB_INTENTS:
            return None
        return route, db_intent
    return route, "unknown"


@lru_cache(maxsize=1)
def _get_classifier():
    try:
        from transformers import pipeline
    except ImportError:
        return None

    model_name = _resolve_local_model_name(settings.intent_classifier_model_name)
    try:
        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=settings.intent_classifier_device,
            top_k=None,
        )
    except Exception as exc:
        logger.warning("KLUE-BERT intent classifier unavailable: %s", exc)
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
