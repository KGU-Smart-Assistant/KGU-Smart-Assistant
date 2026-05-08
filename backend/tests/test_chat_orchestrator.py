import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.schemas.search import SearchResult
from app.services import chat_orchestrator


@pytest.fixture(autouse=True)
def disable_configured_intent_classifier(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", None)


def test_decide_chat_route_uses_relational_db_for_phone_question() -> None:
    decision = chat_orchestrator.decide_chat_route("도서관 전화번호 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "phone"


def test_decide_chat_route_uses_relational_db_for_map_question() -> None:
    decision = chat_orchestrator.decide_chat_route("학생회관 위치 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "map"


def test_decide_chat_route_uses_rag_for_notice_question() -> None:
    decision = chat_orchestrator.decide_chat_route("장학 신청 기간 공지 알려줘")

    assert decision.route == "rag"
    assert "scholarship_support" in decision.reason


def test_decide_chat_route_uses_rag_for_department_question() -> None:
    decision = chat_orchestrator.decide_chat_route("청소년학과 전공이수자격원 접수 안내 알려줘")

    assert decision.route == "rag"


def test_decide_chat_route_uses_rag_for_materials_question() -> None:
    decision = chat_orchestrator.decide_chat_route("자료실 첨부파일 신청서 내용 알려줘")

    assert decision.route == "rag"
    assert "materials" in decision.reason


def test_decide_chat_route_uses_rag_for_graduation_question() -> None:
    decision = chat_orchestrator.decide_chat_route("졸업요건과 전공 학점 기준 알려줘")

    assert decision.route == "rag"
    assert "graduation_requirements" in decision.reason


def test_decide_chat_route_uses_weather_for_forecast_question() -> None:
    decision = chat_orchestrator.decide_chat_route("내일 수원 날씨 알려줘")

    assert decision.route == "weather"


def test_phone_keyword_has_priority_over_department_rag_keyword() -> None:
    decision = chat_orchestrator.decide_chat_route("청소년학과 전화번호 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "phone"


def test_decide_chat_route_parses_llm_json_when_heuristic_is_general(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: '{"route":"rag","db_intent":"unknown","reason":"notice question"}',
    )

    decision = chat_orchestrator.decide_chat_route("이번 비교과 프로그램 내용 알려줘")

    assert decision.route == "rag"
    assert decision.db_intent == "unknown"


def test_decide_chat_route_uses_high_confidence_klue_bert_prediction(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_confidence_threshold", 0.7)
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(
            route="relational_db",
            db_intent="map",
            confidence=0.93,
            label="map",
        ),
    )

    decision = chat_orchestrator.decide_chat_route("8강의동은 어디야?")

    assert decision.route == "relational_db"
    assert decision.db_intent == "map"
    assert decision.reason.startswith("klue-bert:map")


def test_decide_chat_route_falls_back_to_llm_for_low_confidence_klue_bert(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_confidence_threshold", 0.7)
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(
            route="llm",
            db_intent="unknown",
            confidence=0.42,
            label="general",
        ),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: '{"route":"rag","db_intent":"unknown","reason":"low confidence fallback"}',
    )

    decision = chat_orchestrator.decide_chat_route("성적향상 장학금은 어디에서 정보를 찾을 수 있어?")

    assert decision.route == "rag"
    assert decision.reason == "low confidence fallback"


def test_decide_chat_plan_detects_compound_map_and_phone_before_klue_bert(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_confidence_threshold", 0.7)
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(
            route="relational_db",
            db_intent="map",
            confidence=0.99,
            label="map",
        ),
    )

    plan = chat_orchestrator.decide_chat_plan("중앙도서관 위치랑 전화번호 알려줘")

    assert [(action.route, action.db_intent) for action in plan.actions] == [
        ("relational_db", "map"),
        ("relational_db", "phone"),
    ]


def test_decide_chat_plan_parses_llm_multiple_actions(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(
            route="llm",
            db_intent="unknown",
            confidence=0.2,
            label="general",
        ),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: (
            '{"actions":['
            '{"route":"rag","db_intent":"unknown"},'
            '{"route":"weather","db_intent":"unknown"}'
            '],"reason":"compound fallback"}'
        ),
    )

    plan = chat_orchestrator.decide_chat_plan("장학 신청 기간이랑 내일 날씨 알려줘")

    assert [(action.route, action.db_intent) for action in plan.actions] == [
        ("rag", "unknown"),
        ("weather", "unknown"),
    ]


def test_answer_chat_uses_relational_db_service(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "get_phone",
        lambda user_input, db: "도서관 전화번호는 031입니다.",
    )

    result = chat_orchestrator.answer_chat("도서관 전화번호 알려줘", db=None)

    assert result.route == "relational_db"
    assert result.intent == "전화"
    assert result.reply == "도서관 전화번호는 031입니다."
    assert result.sources[0].title == "kgu_contacts"


def test_answer_chat_combines_compound_map_and_phone(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "get_map_response",
        lambda user_input, db: "중앙도서관 위치는 위도/경도 37, 127입니다.",
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "get_phone",
        lambda user_input, db: "중앙도서관 전화번호는 031입니다.",
    )

    result = chat_orchestrator.answer_chat("중앙도서관 위치랑 전화번호 알려줘", db=None)

    assert result.route == "multi"
    assert result.intent == "복합"
    assert "중앙도서관 위치" in result.reply
    assert "중앙도서관 전화번호" in result.reply
    assert [source.title for source in result.sources] == ["kgu_places", "kgu_contacts"]


def test_answer_chat_uses_rag_results_as_context(monkeypatch) -> None:
    captured = {}
    search_result = SearchResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=0.91,
        text="장학 신청 기간은 5월 1일부터 5월 10일까지입니다.",
        title="장학 신청 안내",
        source_url="https://example.com/scholarship",
    )
    monkeypatch.setattr(chat_orchestrator, "search_documents", lambda query, top_k: [search_result])

    def fake_context_answer(user_input: str, context: str) -> str:
        captured["context"] = context
        return "장학 신청 기간은 5월 1일부터 5월 10일까지입니다."

    monkeypatch.setattr(chat_orchestrator, "get_gemini_response_with_context", fake_context_answer)

    result = chat_orchestrator.answer_chat("장학 신청 기간 알려줘", db=None)

    assert result.route == "rag"
    assert result.intent == "RAG"
    assert "장학 신청 안내" in captured["context"]
    assert "5월 1일부터 5월 10일" in captured["context"]
    assert result.sources[0].source_url == "https://example.com/scholarship"


def test_answer_chat_uses_weather_service(monkeypatch) -> None:
    report = SimpleNamespace(
        reply="내일 수원은 비 예보가 있습니다.",
        location_name="수원",
        source_url="https://api.open-meteo.com/v1/forecast",
    )
    monkeypatch.setattr(chat_orchestrator, "get_weather_response", lambda _: report)

    result = chat_orchestrator.answer_chat("내일 수원 날씨 알려줘", db=None)

    assert result.route == "weather"
    assert result.intent == "날씨"
    assert result.reply == "내일 수원은 비 예보가 있습니다."
    assert result.sources[0].type == "weather_api"
