from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.services import chat_orchestrator


@dataclass(frozen=True)
class ExpectedAction:
    route: str
    db_intent: str = "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate how the chat planner decomposes and routes one user question."
    )
    parser.add_argument("--text", help="User input to plan. Reads stdin when omitted.")
    parser.add_argument(
        "--expected-action",
        action="append",
        default=[],
        help=(
            "Expected action in route or route:db_intent form. "
            "Repeat for compound questions, e.g. relational_db:map --expected-action weather."
        ),
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("Input text is required. Pass --text or pipe text through stdin.")

    expected_actions = (
        [parse_expected_action(action) for action in args.expected_action]
        if args.expected_action
        else None
    )
    report = validate_plan(text=text, expected_actions=expected_actions)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))

    if report["matched_expected"] is False:
        raise SystemExit(1)


def validate_plan(
    text: str,
    expected_actions: list[ExpectedAction] | None = None,
) -> dict[str, object]:
    plan = chat_orchestrator.decide_chat_plan(text)
    actual_actions = [
        {
            "route": action.route,
            "db_intent": action.db_intent,
            "query": action.query or text,
            "reason": action.reason,
        }
        for action in plan.actions
    ]

    matched_expected = None
    if expected_actions is not None:
        actual_pairs = [(action["route"], action["db_intent"]) for action in actual_actions]
        expected_pairs = [(action.route, action.db_intent) for action in expected_actions]
        matched_expected = actual_pairs == expected_pairs

    return {
        "text": text,
        "reason": plan.reason,
        "actions": actual_actions,
        "expected_actions": [
            {"route": action.route, "db_intent": action.db_intent}
            for action in expected_actions or []
        ],
        "matched_expected": matched_expected,
    }


def parse_expected_action(raw: str) -> ExpectedAction:
    route, separator, db_intent = raw.strip().partition(":")
    if not route:
        raise ValueError("Expected action route is required.")
    if not separator:
        db_intent = "unknown"
    if route != "relational_db":
        db_intent = "unknown"
    return ExpectedAction(route=route, db_intent=db_intent)


if __name__ == "__main__":
    main()
