from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_orchestrator import decide_chat_route


DEFAULT_DATA = Path("app/data/intent_failure_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate route/domain/detail decisions on collected production-like failure cases."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--fail-under", type=float, default=None)
    args = parser.parse_args()

    rows = _load_rows(Path(args.data))
    results = [_evaluate_row(row) for row in rows]
    correct = sum(result["correct"] for result in results)
    report: dict[str, object] = {
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "errors": [result for result in results if not result["correct"]] if args.show_errors else [],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))

    if args.fail_under is not None and report["accuracy"] < args.fail_under:
        raise SystemExit(1)


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No failure cases found in {path}")
    return rows


def _evaluate_row(row: dict[str, object]) -> dict[str, object]:
    text = str(row["text"])
    decision = decide_chat_route(text)
    expected_route = str(row["expected_route"])
    expected_db_intent = str(row.get("expected_db_intent", "unknown"))
    expected_domains = tuple(row.get("expected_rag_domains") or [])
    expected_domain = str(row.get("expected_rag_domain", ""))
    expected_detail = str(row.get("expected_rag_detail", ""))

    route_ok = decision.route == expected_route
    db_ok = expected_route != "relational_db" or decision.db_intent == expected_db_intent
    domain_ok = expected_route != "rag" or (
        expected_domain == decision.rag_domain
        or bool(expected_domains and expected_domain in decision.rag_domains)
    )
    detail_ok = expected_route != "rag" or not expected_detail or decision.rag_detail == expected_detail
    correct = route_ok and db_ok and domain_ok and detail_ok

    return {
        "text": text,
        "correct": correct,
        "expected": {
            "route": expected_route,
            "db_intent": expected_db_intent,
            "rag_domain": expected_domain or None,
            "rag_detail": expected_detail or None,
        },
        "actual": {
            "route": decision.route,
            "db_intent": decision.db_intent,
            "rag_domain": decision.rag_domain,
            "rag_domains": list(decision.rag_domains),
            "rag_detail": decision.rag_detail,
            "rag_ambiguity": decision.rag_ambiguity,
            "rag_confidence": decision.rag_confidence,
        },
    }


if __name__ == "__main__":
    main()
