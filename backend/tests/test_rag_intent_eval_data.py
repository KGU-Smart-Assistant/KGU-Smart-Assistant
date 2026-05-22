import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import chat_orchestrator


DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "rag_intent_eval.jsonl"


@pytest.fixture(autouse=True)
def use_fake_rag_domain_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_domains_with_klue_bert",
        _fake_rag_domain_classifier,
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_details_with_klue_bert",
        _fake_rag_detail_classifier,
    )


def _fake_rag_domain_classifier(text: str):
    normalized = text.casefold()
    scores = []
    for domain, keywords in chat_orchestrator._RAG_ROUTE_TOPIC_KEYWORDS.items():
        matched = tuple(keyword for keyword in keywords if keyword.casefold() in normalized)
        if not matched:
            continue
        priority = chat_orchestrator._RAG_ROUTE_TOPIC_PRIORITY.get(domain, 0)
        score = min(0.35 + len(matched) * 0.2 + priority * 0.01, 0.99)
        scores.append(SimpleNamespace(domain=domain, score=round(score, 3)))
    if any(score.domain not in {"general_notice", "department_notice"} for score in scores):
        scores = [
            SimpleNamespace(
                domain=score.domain,
                score=round(score.score * 0.7, 3),
            )
            if score.domain in {"general_notice", "department_notice"}
            else score
            for score in scores
        ]
    return tuple(sorted(scores, key=lambda item: item.score, reverse=True)[:3])


def _fake_rag_detail_classifier(text: str):
    normalized = text.casefold()
    detail_keywords = {
        "period": ("기간", "일정", "언제", "마감", "시기"),
        "required_documents": ("서류", "제출", "제출서류", "증명", "첨부", "신청서", "양식", "서식", "자료", "파일"),
        "eligibility": ("대상", "자격", "조건", "가능", "지원", "받을 수"),
        "procedure": ("신청", "절차", "방법", "접수", "어떻게"),
        "benefit": ("금액", "혜택", "지원액", "감면"),
        "announcement_lookup": ("공지", "안내", "모집", "결과 발표", "확인"),
        "summary": ("요약", "정리"),
    }
    scored = [
        (detail, sum(keyword.casefold() in normalized for keyword in keywords))
        for detail, keywords in detail_keywords.items()
    ]
    detail, count = max(scored, key=lambda item: item[1])
    if count > 0:
        return (SimpleNamespace(detail=detail, score=0.99),)
    return ()


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
