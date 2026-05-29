import json
from collections import Counter
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "intent_failure_cases.jsonl"
VALID_ROUTES = {"llm", "relational_db", "rag", "weather"}
VALID_DB_INTENTS = {"unknown", "map", "phone", "info_link"}
VALID_RAG_DETAILS = {
    "period",
    "eligibility",
    "procedure",
    "required_documents",
    "benefit",
    "announcement_lookup",
    "summary",
    "unknown",
}
VALID_AMBIGUITIES = {"clear", "multi_domain", "low_confidence", "missing_detail", "needs_clarification"}


def _rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_intent_failure_cases_have_valid_schema() -> None:
    rows = _rows()
    assert rows

    seen_texts: set[str] = set()
    for row in rows:
        text = row.get("text")
        route = row.get("expected_route")
        db_intent = row.get("expected_db_intent", "unknown")
        assert isinstance(text, str) and text.strip()
        assert text not in seen_texts
        seen_texts.add(text)
        assert route in VALID_ROUTES
        assert db_intent in VALID_DB_INTENTS

        if route == "rag":
            domains = row.get("expected_rag_domains")
            assert isinstance(row.get("expected_rag_domain"), str)
            assert isinstance(domains, list) and domains
            assert row.get("expected_rag_domain") in domains
            assert row.get("expected_rag_detail") in VALID_RAG_DETAILS
            assert row.get("expected_ambiguity") in VALID_AMBIGUITIES


def test_intent_failure_cases_cover_service_risk_patterns() -> None:
    rows = _rows()
    routes = Counter(row["expected_route"] for row in rows)
    failure_types = Counter(row["failure_type"] for row in rows)

    assert routes["rag"] >= 20
    assert routes["relational_db"] >= 5
    assert routes["weather"] >= 1
    assert failure_types["where_rag_not_map"] >= 3
    assert failure_types["multi_domain"] >= 4
    assert failure_types["short_detail"] >= 4
