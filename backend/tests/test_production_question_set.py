import json
from collections import Counter
from pathlib import Path

from app.services.rag_detail_classifier import RAG_DETAIL_LABELS
from app.services.rag_domain_classifier import RAG_DOMAIN_LABELS


DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "production_question_set.jsonl"
VALID_ROUTES = {"llm", "relational_db", "rag", "weather"}
VALID_DB_INTENTS = {"unknown", "map", "phone"}
VALID_AMBIGUITIES = {"clear", "multi_domain", "low_confidence", "missing_detail", "needs_clarification"}


def _rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_production_question_set_has_valid_schema() -> None:
    rows = _rows()
    assert len(rows) >= 70

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
        assert isinstance(row.get("category"), str) and row["category"]
        assert row.get("difficulty") in {"short", "normal", "ambiguous"}

        if route == "rag":
            domains = row.get("expected_rag_domains")
            detail = row.get("expected_rag_detail")
            assert row.get("expected_rag_domain") in RAG_DOMAIN_LABELS
            assert isinstance(domains, list) and domains
            assert row.get("expected_rag_domain") in domains
            assert all(domain in RAG_DOMAIN_LABELS for domain in domains)
            assert detail in RAG_DETAIL_LABELS
            assert row.get("expected_ambiguity") in VALID_AMBIGUITIES


def test_production_question_set_covers_core_routes_and_rag_domains() -> None:
    rows = _rows()
    routes = Counter(row["expected_route"] for row in rows)
    rag_domains = Counter(row.get("expected_rag_domain") for row in rows if row["expected_route"] == "rag")
    difficulties = Counter(row["difficulty"] for row in rows)

    assert routes["rag"] >= 45
    assert routes["relational_db"] >= 10
    assert routes["weather"] >= 4
    assert routes["llm"] >= 4
    assert difficulties["ambiguous"] >= 12

    required_domains = {
        "scholarship",
        "tuition",
        "course_registration",
        "academic_calendar",
        "academic_status",
        "major_change",
        "multi_major",
        "graduation",
        "document_materials",
        "student_life",
        "career_support",
        "international_exchange",
        "department_notice",
        "general_notice",
    }
    assert required_domains.issubset(set(rag_domains))
