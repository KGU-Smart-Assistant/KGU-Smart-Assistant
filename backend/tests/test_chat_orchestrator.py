import os
from types import SimpleNamespace

from langchain_core.documents import Document

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.services import chat_orchestrator
from app.services.langchain_rag_service import LangChainRagResult


def test_decide_chat_route_uses_relational_db_for_phone_question() -> None:
    decision = chat_orchestrator.decide_chat_route("도서관 전화번호 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "phone"


def test_decide_chat_route_uses_relational_db_for_map_question() -> None:
    decision = chat_orchestrator.decide_chat_route("학생회관 위치 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "map"


def test_decide_chat_route_uses_rag_for_notice_question() -> None:
    decision = chat_orchestrator.decide_chat_route("장학금 신청 기간 공지 알려줘")

    assert decision.route == "rag"
    assert "scholarship_support" in decision.reason


def test_decide_chat_route_uses_rag_for_info_source_question() -> None:
    decision = chat_orchestrator.decide_chat_route("성적장학금 관련 정보는 어디서 찾을 수 있어?")

    assert decision.route == "rag"


def test_decide_chat_route_uses_rag_for_department_question() -> None:
    decision = chat_orchestrator.decide_chat_route("청소년학과 전공이수자격 접수 안내 알려줘")

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


def test_decide_chat_route_falls_back_to_heuristic_when_llm_classifier_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: (_ for _ in ()).throw(RuntimeError("quota exceeded")),
    )

    decision = chat_orchestrator.decide_chat_route("안녕")

    assert decision.route == "llm"


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


def test_answer_chat_uses_langchain_rag_chain(monkeypatch) -> None:
    document = Document(
        page_content="장학금 신청 기간은 5월 1일부터 5월 10일까지입니다.",
        metadata={
            "title": "장학금 신청 안내",
            "source_url": "https://example.com/scholarship",
            "score": 0.91,
        },
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "answer_with_langchain_rag",
        lambda user_input, top_k: LangChainRagResult(
            reply="장학금 신청 기간은 5월 1일부터 5월 10일까지입니다.",
            documents=[document],
            context="formatted context",
            expanded_queries=[user_input],
        ),
    )

    result = chat_orchestrator.answer_chat("장학금 신청 기간 알려줘", db=None)

    assert result.route == "rag"
    assert result.intent == "RAG"
    assert result.reply == "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다."
    assert result.sources[0].title == "장학금 신청 안내"
    assert result.sources[0].source_url == "https://example.com/scholarship"


def test_answer_chat_falls_back_to_llm_when_langchain_rag_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "answer_with_langchain_rag",
        lambda user_input, top_k: (_ for _ in ()).throw(RuntimeError("vector store down")),
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
        lambda user_input, top_k: LangChainRagResult(
            reply="관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요.",
            documents=[],
            context="",
            expanded_queries=[user_input],
        ),
    )

    result = chat_orchestrator.answer_chat("장학금 신청 기간 알려줘", db=None)

    assert result.route == "rag"
    assert result.sources == []
    assert "관련 문서를 찾지 못했습니다" in result.reply


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
