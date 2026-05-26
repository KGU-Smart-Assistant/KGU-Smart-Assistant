import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.schemas.search import SearchResult
from app.services import chat_orchestrator


@pytest.fixture(autouse=True)
def disable_configured_intent_classifier(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(chat_orchestrator, "classify_with_klue_bert", _fake_route_classifier)
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
    domain_keywords = {
        "scholarship": ("장학", "scholarship"),
        "tuition": ("등록금", "납부", "환불"),
        "course_registration": ("수강",),
        "academic_calendar": ("학사일정", "개강", "종강", "시험"),
        "academic_status": ("휴학", "복학", "자퇴", "재입학"),
        "major_change": ("전과", "전공변경"),
        "multi_major": ("다전공", "복수전공", "부전공"),
        "graduation": ("졸업",),
        "admission_transfer": ("편입", "입학", "모집요강"),
        "teaching_certification": ("교직", "교원자격"),
        "document_materials": ("양식", "서식", "신청서", "서류", "자료", "파일"),
        "student_life": ("학생증", "동아리", "상담", "기숙사"),
        "career_support": ("취업", "진로", "현장실습", "채용"),
        "international_exchange": ("교환학생", "국제교류", "해외"),
        "department_notice": ("학과", "전공", "컴퓨터공학과", "청소년학과"),
        "general_notice": ("공지", "안내", "모집", "발표"),
    }
    priority = {
        "scholarship": 5,
        "tuition": 5,
        "course_registration": 5,
        "academic_status": 5,
        "major_change": 5,
        "multi_major": 5,
        "graduation": 5,
        "admission_transfer": 5,
        "teaching_certification": 5,
        "document_materials": 4,
        "student_life": 4,
        "career_support": 4,
        "international_exchange": 4,
        "academic_calendar": 3,
        "department_notice": 2,
        "general_notice": 1,
    }
    scores = []
    for domain, keywords in domain_keywords.items():
        count = sum(keyword.casefold() in normalized for keyword in keywords)
        if count:
            scores.append(SimpleNamespace(domain=domain, score=round(min(0.35 + count * 0.2 + priority[domain] * 0.01, 0.99), 3)))
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
    return tuple(
        sorted(scores, key=lambda item: item.score, reverse=True)[:3]
    )


def _fake_route_classifier(text: str):
    normalized = text.casefold()
    if any(keyword in normalized for keyword in ("날씨", "우산", "비 올", "비올", "겉옷", "야외 행사", "weather")):
        return SimpleNamespace(route="weather", db_intent="unknown", confidence=0.99, label="weather")
    if any(keyword in normalized for keyword in ("전화", "전화번호", "연락처")):
        return SimpleNamespace(route="relational_db", db_intent="phone", confidence=0.99, label="relational_db:phone")
    if any(keyword in normalized for keyword in ("위치", "가는 길", "어디", "지도")) and not any(
        keyword in normalized for keyword in ("공지", "정보", "양식", "신청서", "졸업요건", "수강신청", "등록금")
    ):
        return SimpleNamespace(route="relational_db", db_intent="map", confidence=0.99, label="relational_db:map")
    if any(keyword in normalized for keyword in ("db", "데이터", "조회", "목록", "레코드")):
        return SimpleNamespace(route="relational_db", db_intent="unknown", confidence=0.99, label="relational_db")
    if any(
        keyword in normalized
        for keyword in (
            "장학",
            "등록금",
            "수강",
            "졸업",
            "휴학",
            "복학",
            "전과",
            "다전공",
            "편입",
            "교직",
            "공지",
            "양식",
            "자료",
            "교환학생",
            "취업",
            "학과",
            "현장실습",
            "학생증",
            "모집요강",
        )
    ):
        return SimpleNamespace(route="rag", db_intent="unknown", confidence=0.99, label="rag")
    return SimpleNamespace(route="llm", db_intent="unknown", confidence=0.99, label="llm")


def _fake_rag_detail_classifier(text: str):
    normalized = text.casefold()
    detail_keywords = {
        "period": ("기간", "일정", "언제", "마감", "시기", "deadline"),
        "required_documents": ("서류", "제출", "제출서류", "증명", "첨부", "신청서", "양식", "서식", "자료", "파일"),
        "eligibility": ("대상", "자격", "조건", "가능", "지원자격", "받을 수"),
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
    assert decision.reason == "klue-bert:rag:0.990"
    assert decision.rag_domain == "scholarship"
    assert decision.rag_detail == "period"
    assert decision.rag_confidence is not None
    assert decision.rag_confidence > 0.0
    assert decision.rag_domains[0] == "scholarship"
    assert decision.intent_scores[0].domain == "scholarship"
    assert decision.matched_keywords == ()


def test_information_lookup_where_question_uses_rag_not_map() -> None:
    decision = chat_orchestrator.decide_chat_route("장학금 정보는 어디에서 찾을 수 있어?")

    assert decision.route == "rag"
    assert decision.db_intent == "unknown"
    assert decision.reason == "klue-bert:rag:0.990"


def test_form_location_question_uses_rag_not_map() -> None:
    decision = chat_orchestrator.decide_chat_route("등록금 환불 신청서 양식 어디 있어?")

    assert decision.route == "rag"
    assert decision.db_intent == "unknown"
    assert decision.rag_domain == "tuition"
    assert "document_materials" in decision.rag_domains


@pytest.mark.parametrize(
    "question,reason_keyword",
    [
        ("졸업요건은 어디에서 볼 수 있어?", "graduation_requirements"),
        ("수강신청 공지는 어디서 확인해?", "academic_schedule"),
        ("신청서 양식은 어디에서 다운로드해?", "materials"),
        ("청소년학과 공지사항은 어디서 확인해?", "department_sources"),
        ("등록금 납부 기준은 어디에 나와 있어?", "scholarship_support"),
    ],
)
def test_information_lookup_exceptions_use_rag_not_map(
    question: str,
    reason_keyword: str,
) -> None:
    decision = chat_orchestrator.decide_chat_route(question)

    assert decision.route == "rag"
    assert decision.db_intent == "unknown"
    assert decision.reason == "klue-bert:rag:0.990"


def test_decide_chat_route_keeps_multiple_rag_intents_for_complex_question() -> None:
    decision = chat_orchestrator.decide_chat_route("휴학하면 등록금이랑 장학금은 어떻게 돼?")

    assert decision.route == "rag"
    assert decision.rag_domain == "scholarship"
    assert "scholarship" in decision.rag_domains
    assert "academic_status" in decision.rag_domains
    assert "tuition" in decision.rag_domains
    assert [score.domain for score in decision.intent_scores] == list(decision.rag_domains)


def test_decide_chat_route_uses_rag_for_department_question() -> None:
    decision = chat_orchestrator.decide_chat_route("청소년학과 전공이수자격원 접수 안내 알려줘")

    assert decision.route == "rag"
    assert decision.source_scope == "unknown"


def test_decide_chat_route_uses_topic_domain_and_department_scope_for_department_career_notice() -> None:
    decision = chat_orchestrator.decide_chat_route("컴퓨터공학과 취업 공지 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "career_support"
    assert decision.rag_detail == "announcement_lookup"
    assert decision.source_scope == "unknown"


def test_decide_chat_route_uses_rag_for_exchange_student_partner_school_question() -> None:
    decision = chat_orchestrator.decide_chat_route("교환학생 지원 가능한 학교를 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "international_exchange"
    assert decision.rag_detail == "eligibility"
    assert decision.source_scope == "unknown"


def test_decide_chat_route_normalizes_disallowed_detail_for_domain() -> None:
    decision = chat_orchestrator.decide_chat_route("졸업 혜택 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "graduation"
    assert decision.rag_detail == "benefit"


def test_decide_chat_route_uses_department_notice_only_when_topic_is_not_specific() -> None:
    decision = chat_orchestrator.decide_chat_route("컴퓨터공학과 공지 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "department_notice"
    assert decision.rag_detail == "announcement_lookup"
    assert decision.source_scope == "unknown"


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


def test_decide_chat_route_keeps_leave_of_absence_in_academic_status_domain() -> None:
    decision = chat_orchestrator.decide_chat_route("휴학 신청 절차 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "academic_status"
    assert decision.rag_detail == "procedure"


def test_decide_chat_route_uses_rag_for_major_change_question() -> None:
    decision = chat_orchestrator.decide_chat_route("전과 지원 조건 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "major_change"
    assert decision.rag_detail == "eligibility"


def test_decide_chat_route_uses_rag_for_multi_major_question() -> None:
    decision = chat_orchestrator.decide_chat_route("다전공 신청 방법 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "multi_major"
    assert decision.rag_detail == "procedure"


def test_decide_chat_route_uses_rag_for_transfer_admission_question() -> None:
    decision = chat_orchestrator.decide_chat_route("편입 지원 자격 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "admission_transfer"
    assert decision.rag_detail == "eligibility"


def test_decide_chat_route_uses_rag_for_teaching_certification_question() -> None:
    decision = chat_orchestrator.decide_chat_route("교직이수 신청 기간 알려줘")

    assert decision.route == "rag"
    assert decision.rag_domain == "teaching_certification"
    assert decision.rag_detail == "period"


def test_decide_chat_route_uses_weather_for_forecast_question() -> None:
    decision = chat_orchestrator.decide_chat_route("내일 수원 날씨 알려줘")

    assert decision.route == "weather"


@pytest.mark.parametrize(
    "question",
    [
        "오늘 우산 가져가야 해?",
        "오늘 비 올 가능성 있어?",
        "오늘 학교 갈 때 겉옷 필요해?",
        "내일 야외 행사하기 괜찮을까?",
    ],
)
def test_lifestyle_weather_questions_use_weather(question: str) -> None:
    decision = chat_orchestrator.decide_chat_route(question)

    assert decision.route == "weather"
    assert decision.db_intent == "unknown"


@pytest.mark.parametrize(
    "question",
    [
        "내부 DB에 있는 학교 데이터 조회해줘",
        "저장된 캠퍼스 장소 데이터 목록을 확인해줘",
        "서비스 DB의 캠퍼스 레코드 검색해줘",
    ],
)
def test_explicit_db_lookup_uses_relational_db_unknown(question: str) -> None:
    decision = chat_orchestrator.decide_chat_route(question)

    assert decision.route == "relational_db"
    assert decision.db_intent == "unknown"


def test_phone_keyword_has_priority_over_department_rag_keyword() -> None:
    decision = chat_orchestrator.decide_chat_route("청소년학과 전화번호 알려줘")

    assert decision.route == "relational_db"
    assert decision.db_intent == "phone"


def test_high_confidence_general_klue_bert_is_not_overridden_by_phone_keyword(
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
            confidence=0.99,
            label="general",
        ),
    )

    decision = chat_orchestrator.decide_chat_route("학사혁신팀 번호 알려줘")

    assert decision.route == "llm"
    assert decision.db_intent == "unknown"


def test_decide_chat_route_parses_llm_json_when_heuristic_is_general(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator, "classify_with_klue_bert", lambda _: None)
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: '{"route":"rag","db_intent":"unknown","reason":"notice question"}',
    )

    decision = chat_orchestrator.decide_chat_route("이번 비교과 프로그램 내용 알려줘")

    assert decision.route == "rag"
    assert decision.db_intent == "unknown"
    assert decision.rag_domain == "unknown"
    assert decision.rag_detail == "unknown"


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


def test_decide_chat_plan_uses_klue_bert_single_action_without_keyword_split(
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

    assert [(action.route, action.db_intent) for action in plan.actions] == [("relational_db", "map")]


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


def test_decide_chat_plan_splits_compound_question_into_atomic_queries(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator, "classify_with_klue_bert", lambda _: None)
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: (
            '{"actions":['
            '{"query":"weather tomorrow","route":"weather","db_intent":"unknown","reason":"planner"},'
            '{"query":"scholarship deadline","route":"rag","db_intent":"unknown","reason":"planner"}'
            '],"reason":"planner"}'
        ),
    )

    plan = chat_orchestrator.decide_chat_plan("weather tomorrow, scholarship deadline")

    assert [(action.route, action.db_intent, action.query) for action in plan.actions] == [
        ("weather", "unknown", "weather tomorrow"),
        ("rag", "unknown", "scholarship deadline"),
    ]


def test_answer_chat_uses_atomic_queries_for_compound_actions(monkeypatch) -> None:
    captured = {}
    search_result = SearchResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=0.91,
        text="Scholarship deadline is May 10.",
        title="Scholarship notice",
        source_url="https://example.com/scholarship",
    )
    monkeypatch.setattr(chat_orchestrator, "classify_with_klue_bert", lambda _: None)
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: (
            '{"actions":['
            '{"query":"weather tomorrow","route":"weather","db_intent":"unknown","reason":"planner"},'
            '{"query":"scholarship deadline","route":"rag","db_intent":"unknown","reason":"planner"}'
            '],"reason":"planner"}'
        ),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "get_weather_response",
        lambda user_input: captured.setdefault(
            "weather",
            SimpleNamespace(
                reply=f"weather query: {user_input}",
                location_name="Suwon",
                source_url="https://api.open-meteo.com/v1/forecast",
            ),
        ),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "search_documents",
        lambda query, top_k: captured.setdefault("rag_query", query) and [search_result],
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response_with_context",
        lambda user_input, context: f"rag query: {user_input}",
    )

    result = chat_orchestrator.answer_chat("weather tomorrow, scholarship deadline", db=None)

    assert result.route == "weather"
    assert "weather query: weather tomorrow" in result.reply
    assert "rag query: scholarship deadline" in result.reply
    assert captured["rag_query"] == "scholarship deadline"


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
    monkeypatch.setattr(chat_orchestrator, "classify_with_klue_bert", lambda _: None)
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response",
        lambda _: (
            '{"actions":['
            '{"query":"중앙도서관 위치","route":"relational_db","db_intent":"map","reason":"planner"},'
            '{"query":"중앙도서관 전화번호","route":"relational_db","db_intent":"phone","reason":"planner"}'
            '],"reason":"planner"}'
        ),
    )
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

    assert result.route == "relational_db"
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
    monkeypatch.setattr(
        chat_orchestrator,
        "search_documents",
        lambda query, top_k, rag_domain, rag_domains, rag_detail, source_scope: [search_result],
    )

    def fake_context_answer(user_input: str, context: str) -> str:
        captured["context"] = context
        return "장학 신청 기간은 5월 1일부터 5월 10일까지입니다."

    monkeypatch.setattr(chat_orchestrator, "get_gemini_response_with_context", fake_context_answer)

    result = chat_orchestrator.answer_chat("장학 신청 기간 알려줘", db=None)

    assert result.route == "rag"
    assert result.intent == "RAG"
    assert result.rag_domain == "scholarship"
    assert result.rag_domains[0] == "scholarship"
    assert result.rag_detail == "period"
    assert result.source_scope == "unknown"
    assert result.rag_confidence is not None
    assert result.rag_ambiguity == "clear"
    assert result.rewritten_queries
    assert result.matched_keywords == ()
    assert result.intent_scores
    assert result.answer_status == "answered"
    assert "장학 신청 안내" in captured["context"]
    assert "5월 1일부터 5월 10일" in captured["context"]
    assert result.sources[0].source_url == "https://example.com/scholarship"


def test_answer_chat_requests_clarification_when_rag_domain_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(route="rag", db_intent="unknown", confidence=0.99, label="rag"),
    )
    monkeypatch.setattr(chat_orchestrator, "classify_rag_domains_with_klue_bert", lambda _: ())
    monkeypatch.setattr(
        chat_orchestrator,
        "search_documents",
        lambda **kwargs: pytest.fail("RAG search should wait for clarification"),
    )

    result = chat_orchestrator.answer_chat("scholarship deadline", db=None)

    assert result.route == "rag"
    assert result.answer_status == "insufficient"
    assert result.rag_ambiguity == "needs_clarification"
    assert result.sources == []
    assert result.unverified


def test_answer_chat_requests_clarification_for_multi_domain_rag(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(route="rag", db_intent="unknown", confidence=0.99, label="rag"),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_domains_with_klue_bert",
        lambda _: (
            SimpleNamespace(domain="tuition", score=0.74),
            SimpleNamespace(domain="scholarship", score=0.69),
        ),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_details_with_klue_bert",
        lambda _: (SimpleNamespace(detail="period", score=0.99),),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "search_documents",
        lambda **kwargs: pytest.fail("RAG search should wait for clarification"),
    )

    result = chat_orchestrator.answer_chat("tuition scholarship deadline", db=None)

    assert result.route == "rag"
    assert result.answer_status == "insufficient"
    assert result.rag_ambiguity == "multi_domain"
    assert result.rag_domains[:2] == ("tuition", "scholarship")
    assert "tuition" in result.reply
    assert "scholarship" in result.reply


def test_answer_chat_allows_missing_rag_detail_by_default(monkeypatch) -> None:
    search_result = SearchResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=0.91,
        text="Scholarship notices are available on the scholarship board.",
        title="Scholarship notice",
        source_url="https://example.com/scholarship",
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: SimpleNamespace(route="rag", db_intent="unknown", confidence=0.99, label="rag"),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_domains_with_klue_bert",
        lambda _: (SimpleNamespace(domain="scholarship", score=0.92),),
    )
    monkeypatch.setattr(chat_orchestrator, "classify_rag_details_with_klue_bert", lambda _: ())
    monkeypatch.setattr(chat_orchestrator, "search_documents", lambda query, top_k: [search_result])
    monkeypatch.setattr(
        chat_orchestrator,
        "get_gemini_response_with_context",
        lambda user_input, context: "Scholarship answer",
    )

    result = chat_orchestrator.answer_chat("scholarship information", db=None)

    assert result.route == "rag"
    assert result.answer_status == "answered"
    assert result.rag_ambiguity == "missing_detail"
    assert result.rag_detail == "unknown"
    assert result.reply == "Scholarship answer"


def test_answer_chat_returns_insufficient_when_results_do_not_ground_answer(monkeypatch) -> None:
    search_result = SearchResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=0.05,
        text="휴학 신청은 포털에서 진행합니다.",
        title="휴학 신청 안내",
        source_url="https://example.com/leave",
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "search_documents",
        lambda query, top_k, rag_domain, rag_domains, rag_detail, source_scope: [search_result],
    )

    monkeypatch.setattr(chat_orchestrator.settings, "rag_clarify_on_multi_domain", False)
    result = chat_orchestrator.answer_chat("휴학하면 장학금은 어떻게 돼?", db=None)

    assert result.route == "rag"
    assert result.answer_status == "insufficient"
    assert result.unverified
    assert "근거를 확인할 수 없습니다" in result.reply


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
