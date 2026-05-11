from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Literal


Route = Literal["llm", "relational_db", "rag", "weather"]
DbIntent = Literal["map", "phone", "unknown"]

ROUTES: set[str] = {"llm", "relational_db", "rag", "weather"}
DB_INTENTS: set[str] = {"map", "phone", "unknown"}
LEGACY_LABELS: dict[str, tuple[Route, DbIntent]] = {
    "llm": ("llm", "unknown"),
    "general": ("llm", "unknown"),
    "db": ("relational_db", "unknown"),
    "map": ("relational_db", "map"),
    "phone": ("relational_db", "phone"),
    "rag": ("rag", "unknown"),
    "weather": ("weather", "unknown"),
}


@dataclass(frozen=True)
class IntentValidationResult:
    text: str
    route: Route
    db_intent: DbIntent
    label: str
    confidence: float
    accepted: bool
    expected_route: str | None = None
    expected_db_intent: str | None = None
    matched_expected: bool | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one input against a trained KLUE-BERT intent classifier."
    )
    parser.add_argument("--text", help="User input to classify. Reads stdin when omitted.")
    parser.add_argument(
        "--model",
        default=os.getenv("INTENT_CLASSIFIER_MODEL_NAME") or "models/intent-klue-bert",
        help="Local model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("INTENT_CLASSIFIER_CONFIDENCE_THRESHOLD", "0.7")),
        help="Minimum confidence required to accept the classifier decision.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.getenv("INTENT_CLASSIFIER_DEVICE", "-1")),
        help="Transformers pipeline device. Use -1 for CPU, 0 for first GPU.",
    )
    parser.add_argument("--expected-route", choices=sorted(ROUTES))
    parser.add_argument("--expected-db-intent", choices=sorted(DB_INTENTS))
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("Input text is required. Pass --text or pipe text through stdin.")

    result = validate_text(
        text=text,
        model_name=args.model,
        threshold=args.threshold,
        device=args.device,
        expected_route=args.expected_route,
        expected_db_intent=args.expected_db_intent,
    )
    print(
        json.dumps(
            result.__dict__,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )

    if result.matched_expected is False:
        raise SystemExit(1)


def validate_text(
    text: str,
    model_name: str,
    threshold: float,
    device: int,
    expected_route: str | None = None,
    expected_db_intent: str | None = None,
) -> IntentValidationResult:
    output = _classify(text=text, model_name=model_name, device=device)
    label = str(output.get("label", "")).lower()
    confidence = float(output.get("score", 0.0))
    route, db_intent = parse_model_label(label)

    matched_expected = None
    if expected_route is not None or expected_db_intent is not None:
        expected_db_intent = expected_db_intent or "unknown"
        matched_expected = route == expected_route and db_intent == expected_db_intent

    return IntentValidationResult(
        text=text,
        route=route,
        db_intent=db_intent,
        label=label,
        confidence=confidence,
        accepted=confidence >= threshold,
        expected_route=expected_route,
        expected_db_intent=expected_db_intent,
        matched_expected=matched_expected,
    )


def parse_model_label(label: str) -> tuple[Route, DbIntent]:
    normalized = label.strip().lower().replace("__", ":").replace("/", ":")
    if normalized in LEGACY_LABELS:
        return LEGACY_LABELS[normalized]

    route, separator, db_intent = normalized.partition(":")
    if route not in ROUTES:
        raise ValueError(f"Unsupported classifier label: {label!r}")
    if route != "relational_db":
        return route, "unknown"  # type: ignore[return-value]
    if not separator:
        return "relational_db", "unknown"
    if db_intent not in DB_INTENTS:
        raise ValueError(f"Unsupported relational_db intent label: {label!r}")
    return "relational_db", db_intent  # type: ignore[return-value]


def _classify(text: str, model_name: str, device: int) -> dict[str, object]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install ML dependencies first: pip install -r requirements-ml.txt") from exc

    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        top_k=None,
    )
    output = classifier(text, truncation=True)
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Classifier returned no predictions.")
        output = output[0]
        if isinstance(output, list):
            if not output:
                raise RuntimeError("Classifier returned no predictions.")
            output = max(output, key=lambda item: float(item.get("score", 0.0)))
    return output


if __name__ == "__main__":
    main()
