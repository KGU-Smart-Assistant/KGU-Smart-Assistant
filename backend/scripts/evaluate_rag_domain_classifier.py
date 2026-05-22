from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag_domain_classifier import RAG_DOMAIN_LABELS, _predictions_from_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG domain multi-label classifier.")
    parser.add_argument("--data", default="app/data/rag_domain_test.jsonl")
    parser.add_argument("--model", default=os.getenv("RAG_DOMAIN_CLASSIFIER_MODEL_NAME") or "models/rag-domain-klue-bert-v2")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("RAG_DOMAIN_CLASSIFIER_CONFIDENCE_THRESHOLD", "0.5")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_DOMAIN_CLASSIFIER_TOP_K", "3")))
    parser.add_argument("--device", type=int, default=int(os.getenv("RAG_DOMAIN_CLASSIFIER_DEVICE", "-1")))
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--fail-under", type=float, default=None)
    args = parser.parse_args()

    examples = load_examples(Path(args.data))
    classifier = load_classifier(args.model, args.device)
    raw_outputs = classifier([example["text"] for example in examples], truncation=True)
    predictions = [
        predict_domains(output, threshold=args.threshold, top_k=args.top_k)
        for output in raw_outputs
    ]
    report = build_report(examples, predictions, include_errors=args.show_errors)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    if args.fail_under is not None and report["primary_accuracy"] < args.fail_under:
        raise SystemExit(1)


def load_examples(path: Path) -> list[dict[str, object]]:
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
        expected = tuple(row.get("expected_domains") or [row.get("rag_domain")])
        unknown = [domain for domain in expected if domain not in RAG_DOMAIN_LABELS]
        if not text:
            raise ValueError(f"Missing text at {resolved}:{line_number}")
        if unknown:
            raise ValueError(f"Unsupported domains {unknown!r} at {resolved}:{line_number}")
        examples.append({"text": text, "expected_domains": expected})
    if not examples:
        raise ValueError(f"No examples found in {resolved}")
    return examples


def load_classifier(model: str, device: int):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install ML dependencies first.") from exc
    return pipeline("text-classification", model=model, tokenizer=model, device=device, top_k=None, function_to_apply="sigmoid")


def predict_domains(output: object, *, threshold: float, top_k: int) -> list[str]:
    predictions = sorted(_predictions_from_output(output), key=lambda item: item.score, reverse=True)
    return [prediction.domain for prediction in predictions if prediction.score >= threshold][:top_k]


def build_report(examples: list[dict[str, object]], predictions: list[list[str]], *, include_errors: bool) -> dict[str, object]:
    total = len(examples)
    any_hits = 0
    primary_hits = 0
    exact_hits = 0
    expected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    errors = []
    for example, predicted in zip(examples, predictions, strict=True):
        expected = set(example["expected_domains"])
        expected_counts.update(expected)
        predicted_counts.update(predicted)
        any_hit = bool(expected.intersection(predicted))
        primary_hit = bool(predicted and predicted[0] in expected)
        exact_hit = set(predicted) == expected
        any_hits += any_hit
        primary_hits += primary_hit
        exact_hits += exact_hit
        if include_errors and not primary_hit:
            errors.append({"text": example["text"], "expected": sorted(expected), "predicted": predicted})
    report: dict[str, object] = {
        "total": total,
        "any_expected_in_topk": any_hits,
        "topk_recall": any_hits / total,
        "primary_hits": primary_hits,
        "primary_accuracy": primary_hits / total,
        "exact_matches": exact_hits,
        "exact_match_rate": exact_hits / total,
        "expected_counts": dict(sorted(expected_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
    }
    if include_errors:
        report["errors"] = errors
    return report


if __name__ == "__main__":
    main()
