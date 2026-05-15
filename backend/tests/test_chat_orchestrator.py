import os
from types import SimpleNamespace

from langchain_core.documents import Document

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.services import chat_orchestrator
from app.services.langchain_rag_service import LangChainRagResult


def test_decide_chat_route_uses_relational_db_for_phone_question() -> None:
    decision = chat_orchestrator.decide_chat_route("예술관 전화번호 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "phone"


def test_decide_chat_route_uses_relational_db_for_map_question() -> None:
    decision = chat_orchestrator.decide_chat_route("학생회관 위치 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "map"


def test_decide_chat_route_uses_rag_for_notice_question() -> None:
    decision = chat_orchestrator.decide_chat_route("장학금 신청 기간 공지 알려줘")

    assert decision.route == "rag"
    assert decision.rag_category == "scholarship"
    assert "scholarship" in decision.reason


def test_decide_chat_route_uses_rag_for_info_source_question() -> None:
    decision = chat_orchestrator.decide_chat_route("성적장학금 관련 정보는 어디서 찾을 수 있어?")

    assert decision.route == "rag"
    assert decision.rag_category == "scholarship"


def test_decide_chat_route_uses_rag_for_materials_question() -> None:
    decision = chat_orchestrator.decide_chat_route("자료실 첨부파일 신청서 내용 알려줘")

    assert decision.route == "rag"
    assert decision.rag_category == "document_materials"


def test_decide_chat_route_uses_rag_for_graduation_question() -> None:
    decision = chat_orchestrator.decide_chat_route("졸업요건과 전공 학점 기준 알려줘")

    assert decision.route == "rag"
    assert decision.rag_category == "graduation"


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
        lambda _: '{"route":"rag","db_intent":"unknown","reason":"notice question","rag_category":"general_notice"}',
    )

    decision = chat_orchestrator.decide_chat_route("오늘 기분은 좋은데 공지가 궁금해")

    assert decision.route == "rag"
    assert decision.db_intent == "unknown"
    assert decision.rag_category == "general_notice"


def test_decide_chat_route_falls_back_to_heuristic_when_llm_classifier_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: (_ for _ in ()).throw(RuntimeError("quota exceeded")),
    )

    decision = chat_orchestrator.decide_chat_route("안녕")

    assert decision.route == "llm"


def test_answer_chat_uses_relational_db_service(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator, "get_phone", lambda user_input, db: "예술관 전화번호는 031입니다.")

    result = chat_orchestrator.answer_chat("예술관 전화번호 알려줘", db=None)

    assert result.route == "relational_db"
    assert result.intent == "전화"
    assert result.reply == "예술관 전화번호는 031입니다."
    assert result.sources[0].title == "kgu_contacts"


def test_answer_chat_uses_langchain_rag_chain_with_category(monkeypatch) -> None:
    captured = {}
    document = Document(
        page_content="장학금 신청 기간은 5월 1일부터 5월 10일까지입니다.",
        metadata={"title": "장학금 신청 안내", "source_url": "https://example.com/scholarship", "score": 0.91},
    )

    def fake_rag(user_input, top_k, category, detail=None):
        captured["category"] = category
        return LangChainRagResult(
            reply="장학금 신청 기간은 5월 1일부터 5월 10일까지입니다.\n\n출처:\n- https://example.com/scholarship",
            documents=[document],
            context="formatted context",
            expanded_queries=[user_input],
            confidence=0.91,
        )

    monkeypatch.setattr(chat_orchestrator, "answer_with_langchain_rag", fake_rag)

    result = chat_orchestrator.answer_chat("장학금 신청 기간 알려줘", db=None)

    assert captured["category"] == "scholarship"
    assert result.route == "rag"
    assert result.intent == "RAG"
    assert result.sources[0].title == "장학금 신청 안내"


def test_answer_chat_falls_back_to_llm_when_langchain_rag_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "answer_with_langchain_rag",
        lambda user_input, top_k, category: (_ for _ in ()).throw(RuntimeError("vector store down")),
    )
    monkeypatch.setattr(chat_orchestrator, "get_gemini_response", lambda user_input: "일반 답변")

    result = chat_orchestrator.answer_chat("장학금 신청 기간 알려줘", db=None)

    assert result.route == "llm"
    assert result.intent == "일반"
    assert result.reply == "일반 답변"


def test_answer_chat_returns_no_context_message_when_langchain_rag_has_no_documents(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "answer_with_langchain_rag",
        lambda user_input, top_k, category, detail=None: LangChainRagResult(
            reply="관련 문서를 찾지 못했습니다.",
            documents=[],
            context="",
            expanded_queries=[user_input],
            low_confidence=True,
        ),
    )

    result = chat_orchestrator.answer_chat("장학금 신청 기간 알려줘", db=None)

    assert result.route == "rag"
    assert result.sources == []
    assert "관련 문서" in result.reply


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
