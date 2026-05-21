from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_intent_classifier import DB_INTENTS, ROUTES, parse_model_label


Route = Literal["llm", "relational_db", "rag", "weather"]
DbIntent = Literal["map", "phone", "unknown"]


@dataclass(frozen=True)
class EvalExample:
    text: str
    route: Route
    db_intent: DbIntent

    @property
    def label(self) -> str:
        if self.route == "relational_db":
            return f"{self.route}:{self.db_intent}"
        return self.route


@dataclass(frozen=True)
class EvalPrediction:
    text: str
    expected_route: Route
    expected_db_intent: DbIntent
    predicted_route: Route
    predicted_db_intent: DbIntent
    predicted_label: str
    confidence: float
    accepted: bool

    @property
    def expected_label(self) -> str:
        if self.expected_route == "relational_db":
            return f"{self.expected_route}:{self.expected_db_intent}"
        return self.expected_route

    @property
    def predicted_public_label(self) -> str:
        if self.predicted_route == "relational_db":
            return f"{self.predicted_route}:{self.predicted_db_intent}"
        return self.predicted_route

    @property
    def correct(self) -> bool:
        return (
            self.expected_route == self.predicted_route
            and self.expected_db_intent == self.predicted_db_intent
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained KLUE-BERT intent classifier on a JSONL dataset."
    )
    parser.add_argument("--data", default="app/data/intent_eval.jsonl")
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
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Include incorrect predictions in output.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with code 1 only when accuracy is below this value.",
    )
    args = parser.parse_args()

    examples = load_examples(Path(args.data))
    predictions = evaluate_examples(
        examples=examples,
        model_name=args.model,
        threshold=args.threshold,
        device=args.device,
    )
    report = build_report(predictions, threshold=args.threshold, include_errors=args.show_errors)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))

    if args.fail_under is not None and report["accuracy"] < args.fail_under:
        raise SystemExit(1)


def load_examples(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "")).strip()
        route = str(payload.get("route", "")).strip()
        db_intent = str(payload.get("db_intent", "unknown")).strip()
        if route in {"map", "phone"}:
            db_intent = route
            route = "relational_db"
        if not text:
            raise ValueError(f"Missing text at {path}:{line_number}")
        if route not in ROUTES:
            raise ValueError(f"Unsupported route {route!r} at {path}:{line_number}")
        if db_intent not in DB_INTENTS:
            raise ValueError(f"Unsupported db_intent {db_intent!r} at {path}:{line_number}")
        if route != "relational_db":
            db_intent = "unknown"
        examples.append(EvalExample(text=text, route=route, db_intent=db_intent))  # type: ignore[arg-type]

    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def evaluate_examples(
    examples: list[EvalExample],
    model_name: str,
    threshold: float,
    device: int,
) -> list[EvalPrediction]:
    classifier = _load_classifier(model_name=model_name, device=device)
    raw_outputs = classifier([example.text for example in examples], truncation=True)
    return [
        _prediction_from_output(example=example, output=output, threshold=threshold)
        for example, output in zip(examples, raw_outputs, strict=True)
    ]


def build_report(
    predictions: list[EvalPrediction],
    threshold: float,
    include_errors: bool = False,
) -> dict[str, object]:
    total = len(predictions)
    correct = sum(prediction.correct for prediction in predictions)
    accepted = [prediction for prediction in predictions if prediction.accepted]
    accepted_correct = sum(prediction.correct for prediction in accepted)
    expected_counts = Counter(prediction.expected_label for prediction in predictions)
    predicted_counts = Counter(prediction.predicted_public_label for prediction in predictions)

    by_expected: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[EvalPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.expected_label].append(prediction)

    for label, label_predictions in sorted(grouped.items()):
        label_total = len(label_predictions)
        label_correct = sum(prediction.correct for prediction in label_predictions)
        by_expected[label] = {
            "total": label_total,
            "correct": label_correct,
            "accuracy": label_correct / label_total,
        }

    report: dict[str, object] = {
        "threshold": threshold,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "accepted_total": len(accepted),
        "accepted_correct": accepted_correct,
        "accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
        "expected_counts": dict(sorted(expected_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
        "by_expected": by_expected,
    }

    if include_errors:
        report["errors"] = [
            {
                "text": prediction.text,
                "expected": prediction.expected_label,
                "predicted": prediction.predicted_public_label,
                "raw_label": prediction.predicted_label,
                "confidence": prediction.confidence,
                "accepted": prediction.accepted,
            }
            for prediction in predictions
            if not prediction.correct
        ]

    return report


def _load_classifier(model_name: str, device: int):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install ML dependencies first: pip install -r requirements-ml.txt") from exc

    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        top_k=None,
    )


def _prediction_from_output(
    example: EvalExample,
    output: object,
    threshold: float,
) -> EvalPrediction:
    normalized_output = _pick_top_output(output)
    label = str(normalized_output.get("label", "")).lower()
    confidence = float(normalized_output.get("score", 0.0))
    route, db_intent = parse_model_label(label)
    return EvalPrediction(
        text=example.text,
        expected_route=example.route,
        expected_db_intent=example.db_intent,
        predicted_route=route,
        predicted_db_intent=db_intent,
        predicted_label=label,
        confidence=confidence,
        accepted=confidence >= threshold,
    )


def _pick_top_output(output: object) -> dict[str, object]:
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Classifier returned no predictions.")
        if isinstance(output[0], list):
            return _pick_top_output(output[0])
        return max(output, key=lambda item: float(item.get("score", 0.0)))
    if not isinstance(output, dict):
        raise RuntimeError(f"Unexpected classifier output: {output!r}")
    return output


if __name__ == "__main__":
    main()
