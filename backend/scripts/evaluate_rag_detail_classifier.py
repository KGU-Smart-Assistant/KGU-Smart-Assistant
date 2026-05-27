from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag_detail_classifier import RAG_DETAIL_LABELS, _predictions_from_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG detail classifier.")
    parser.add_argument("--data", default="app/data/rag_detail_test.jsonl")
    parser.add_argument("--model", default=os.getenv("RAG_DETAIL_CLASSIFIER_MODEL_NAME") or "models/rag-detail-klue-bert-v5")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("RAG_DETAIL_CLASSIFIER_CONFIDENCE_THRESHOLD", "0.45")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_DETAIL_CLASSIFIER_TOP_K", "3")))
    parser.add_argument("--device", type=int, default=int(os.getenv("RAG_DETAIL_CLASSIFIER_DEVICE", "-1")))
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--fail-under", type=float, default=None)
    args = parser.parse_args()

    examples = load_examples(Path(args.data))
    classifier = load_classifier(args.model, args.device)
    outputs = classifier([example["text"] for example in examples], truncation=True)
    predictions = [predict_details(output, threshold=args.threshold, top_k=args.top_k) for output in outputs]
    report = build_report(examples, predictions, include_errors=args.show_errors)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    if args.fail_under is not None and report["primary_accuracy"] < args.fail_under:
        raise SystemExit(1)


def load_examples(path: Path) -> list[dict[str, str]]:
    resolved = path
    if not resolved.exists() and not resolved.is_absolute():
        backend_relative = Path(__file__).resolve().parents[1] / resolved
        if backend_relative.exists():
            resolved = backend_relative
    examples = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        expected_details = tuple(row.get("expected_details") or [row.get("rag_detail", "")])
        if not text:
            raise ValueError(f"Missing text at {resolved}:{line_number}")
        unknown = [detail for detail in expected_details if detail not in RAG_DETAIL_LABELS]
        if unknown:
            raise ValueError(f"Unsupported detail {unknown!r} at {resolved}:{line_number}")
        examples.append({"text": text, "expected_details": tuple(expected_details)})
    if not examples:
        raise ValueError(f"No examples found in {resolved}")
    return examples


def load_classifier(model: str, device: int):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install ML dependencies first.") from exc
    return pipeline("text-classification", model=model, tokenizer=model, device=device, top_k=None)


def predict_details(output: object, *, threshold: float, top_k: int) -> list[str]:
    predictions = sorted(_predictions_from_output(output), key=lambda item: item.score, reverse=True)
    accepted = [prediction.detail for prediction in predictions if prediction.score >= threshold][:top_k]
    return accepted or ["unknown"]


def build_report(examples: list[dict[str, object]], predictions: list[list[str]], *, include_errors: bool) -> dict[str, object]:
    total = len(examples)
    primary_hits = 0
    any_hits = 0
    exact = 0
    expected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    errors = []
    for example, predicted in zip(examples, predictions, strict=True):
        expected = set(example["expected_details"])
        expected_counts.update(expected)
        predicted_counts.update(predicted)
        primary_hits += bool(predicted and predicted[0] in expected)
        any_hits += bool(expected.intersection(predicted))
        exact += set(predicted) == expected
        if include_errors and not bool(predicted and predicted[0] in expected):
            errors.append({"text": example["text"], "expected": sorted(expected), "predicted": predicted})
    report: dict[str, object] = {
        "total": total,
        "primary_hits": primary_hits,
        "primary_accuracy": primary_hits / total,
        "topk_hits": any_hits,
        "topk_recall": any_hits / total,
        "exact_matches": exact,
        "exact_match_rate": exact / total,
        "expected_counts": dict(sorted(expected_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
    }
    if include_errors:
        report["errors"] = errors
    return report


if __name__ == "__main__":
    main()
