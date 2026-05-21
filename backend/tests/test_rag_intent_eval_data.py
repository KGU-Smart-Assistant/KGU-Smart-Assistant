import json
from pathlib import Path

from app.services import chat_orchestrator


DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "rag_intent_eval.jsonl"


def test_rag_intent_eval_examples_match_expected_taxonomy() -> None:
    examples = [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for example in examples:
        decision = chat_orchestrator.decide_chat_route(example["text"])
        expected_domains = example.get("expected_domains", [example["rag_domain"]])
        assert decision.route == example["route"]
        if example.get("assert_primary", len(expected_domains) == 1):
            assert decision.rag_domain == example["rag_domain"]
            assert decision.rag_detail == example["rag_detail"]
        assert decision.source_scope == example["source_scope"]
        actual_domains = list(decision.rag_domains)
        for domain in expected_domains:
            assert domain in actual_domains
