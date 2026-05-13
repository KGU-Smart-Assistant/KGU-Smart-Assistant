import os
from types import SimpleNamespace

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.schemas.search import SearchResult
from app.services import chat_orchestrator


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
    assert decision.reason == "rag keyword"
    assert decision.rag_domain == "scholarship"
    assert decision.rag_detail == "period"


def test_decide_chat_route_uses_rag_for_department_question() -> None:
    decision = chat_orchestrator.decide_chat_route("청소년학과 전공이수자격원 접수 안내 알려줘")

    assert decision.route == "rag"
    assert decision.source_scope == "department"


def test_decide_chat_route_uses_topic_domain_and_department_scope_for_department_career_notice() -> None:
    decision = chat_orchestrator.decide_chat_route("컴퓨터공학과 취업 공지 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "career_support"
    assert decision.rag_detail == "announcement_lookup"
    assert decision.source_scope == "department"


def test_decide_chat_route_uses_rag_for_exchange_student_partner_school_question() -> None:
    decision = chat_orchestrator.decide_chat_route("교환학생 지원 가능한 학교를 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "international_exchange"
    assert decision.rag_detail == "eligibility"
    assert decision.source_scope == "unknown"


def test_decide_chat_route_uses_department_notice_only_when_topic_is_not_specific() -> None:
    decision = chat_orchestrator.decide_chat_route("컴퓨터공학과 공지 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "department_notice"
    assert decision.rag_detail == "announcement_lookup"
    assert decision.source_scope == "department"


def test_decide_chat_route_uses_rag_for_materials_question() -> None:
    decision = chat_orchestrator.decide_chat_route("자료실 첨부파일 신청서 내용 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "document_materials"


def test_decide_chat_route_uses_rag_for_graduation_question() -> None:
    decision = chat_orchestrator.decide_chat_route("졸업요건과 전공 학점 기준 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "graduation"


def test_decide_chat_route_separates_scholarship_period_from_academic_calendar() -> None:
    decision = chat_orchestrator.decide_chat_route("장학금 신청기간 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "scholarship"
    assert decision.rag_detail == "period"


def test_decide_chat_route_keeps_where_to_check_scholarship_as_rag() -> None:
    decision = chat_orchestrator.decide_chat_route("장학금 신청 공지 어디서 확인해?")

    assert decision.route == "rag"
    assert decision.rag_domain == "scholarship"
    assert decision.rag_detail == "announcement_lookup"


def test_decide_chat_route_separates_course_registration_period_from_scholarship() -> None:
    decision = chat_orchestrator.decide_chat_route("수강신청 기간 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "course_registration"
    assert decision.rag_detail == "period"


def test_decide_chat_route_keeps_leave_of_absence_in_academic_calendar_domain() -> None:
    decision = chat_orchestrator.decide_chat_route("휴학 신청 절차 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "academic_calendar"
    assert decision.rag_detail == "procedure"


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
    assert decision.rag_domain == "career_support"
    assert decision.rag_detail == "unknown"


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
    monkeypatch.setattr(
        chat_orchestrator,
        "search_documents",
        lambda query, top_k, rag_domain, rag_detail, source_scope: [search_result],
    )

    def fake_context_answer(user_input: str, context: str) -> str:
        captured["context"] = context
        return "장학 신청 기간은 5월 1일부터 5월 10일까지입니다."

    monkeypatch.setattr(chat_orchestrator, "get_gemini_response_with_context", fake_context_answer)

    result = chat_orchestrator.answer_chat("장학 신청 기간 알려줘", db=None)

    assert result.route == "rag"
    assert result.intent == "RAG"
    assert result.rag_domain == "scholarship"
    assert result.rag_detail == "period"
    assert result.source_scope == "unknown"
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
