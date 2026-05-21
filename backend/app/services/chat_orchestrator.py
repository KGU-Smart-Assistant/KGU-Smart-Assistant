from __future__ import annotations

import json
import re
import inspect
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.schemas.search import SearchResult
from app.services.call_service import get_phone
from app.services.gemini_service import (
    get_gemini_response,
    get_gemini_response_with_context,
)
from app.core.config import settings
from app.services.klue_bert_intent_classifier import classify_with_klue_bert
from app.services.map_service import get_map_response
from app.services.search_service import search_documents
from app.services.weather_service import get_weather_response

ChatRoute = Literal["llm", "relational_db", "rag", "weather", "multi"]
AtomicChatRoute = Literal["llm", "relational_db", "rag", "weather"]
DbIntent = Literal["map", "phone", "unknown"]


@dataclass(frozen=True)
class ChatDecision:
    route: AtomicChatRoute
    db_intent: DbIntent = "unknown"
    reason: str = ""
    query: str | None = None
    rag_domain: str | None = None
    rag_domains: tuple[str, ...] = ()
    rag_detail: str | None = None
    source_scope: str | None = None
    rag_confidence: float | None = None
    matched_keywords: tuple[str, ...] = ()
    intent_scores: tuple["RagIntentScore", ...] = ()


@dataclass(frozen=True)
class ChatPlan:
    actions: tuple[ChatDecision, ...]
    reason: str = ""


@dataclass(frozen=True)
class ChatSource:
    type: str
    title: str
    source_url: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class ChatResult:
    reply: str
    intent: str
    route: ChatRoute
    sources: list[ChatSource] = field(default_factory=list)
    rag_domain: str | None = None
    rag_domains: tuple[str, ...] = ()
    rag_detail: str | None = None
    source_scope: str | None = None
    rag_confidence: float | None = None
    matched_keywords: tuple[str, ...] = ()
    intent_scores: tuple["RagIntentScore", ...] = ()
    answer_status: Literal["answered", "partial", "insufficient"] = "answered"
    unverified: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagIntentScore:
    domain: str
    score: float
    matched_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagClassification:
    domain: str
    domains: tuple[str, ...] = ()
    detail: str = "unknown"
    source_scope: str = "unknown"
    confidence: float = 0.0
    matched_keywords: tuple[str, ...] = ()
    intent_scores: tuple[RagIntentScore, ...] = ()


_PHONE_KEYWORDS = (
    "전화",
    "전화번호",
    "연락처",
    "문의처",
    "사무실 번호",
    "행정실 번호",
    "문의 번호",
    "대표 번호",
    "담당 부서 번호",
    "통화 가능한 번호",
    "상담 번호",
    "어느 번호",
    "번호 알려",
    "번호가",
    "번호로",
)
_MAP_KEYWORDS = (
    "어디",
    "위치",
    "찾아가",
    "가는 길",
    "어떻게 가",
    "길찾기",
    "캠퍼스맵",
    "지도",
    "근처 건물",
    "캠퍼스 안",
    "현재 위치",
    "강의실",
    "도서관",
    "학생회관",
    "공학관",
    "복지관",
    "박물관",
    "정문",
    "후문",
)
_LOCATION_REQUEST_KEYWORDS = (
    "위치",
    "찾아가",
    "가는 길",
    "어떻게 가",
    "길찾기",
    "캠퍼스맵",
    "지도",
    "근처 건물",
    "캠퍼스 안",
    "현재 위치",
    "어디야",
    "어디 있",
    "호실",
)
_WEATHER_KEYWORDS = (
    "날씨",
    "기온",
    "강수",
    "우산",
    "비 올",
    "비 와",
    "비가",
    "비올",
    "눈 와",
    "더워",
    "더운",
    "추워",
    "춥",
    "겉옷",
    "야외 행사",
    "걸어다니기 괜찮",
    "학교 갈 때",
    "예보",
)
_DB_LOOKUP_KEYWORDS = (
    "db",
    "데이터베이스",
    "내부 db",
    "서비스 db",
    "백엔드 db",
    "내부 데이터",
    "저장된",
    "등록된",
    "레코드",
    "저장 데이터",
    "캠퍼스 데이터",
    "학교 데이터",
    "학교 항목",
    "학내 데이터",
    "관리 중인",
    "기본 데이터",
)
_INFORMATION_LOOKUP_KEYWORDS = (
    "정보",
    "공지",
    "공지사항",
    "안내",
    "내용",
    "자료",
    "문서",
    "홈페이지",
    "확인",
    "찾을 수",
    "볼 수",
    "볼수",
    "조회",
    "열람",
    "게시",
    "나와",
    "다운로드",
    "받을 수",
    "파일",
    "기준",
    "조건",
    "기간",
    "서류",
    "대상",
    "자격",
    "절차",
    "요건",
    "신청",
)

# RAG uses a two-axis taxonomy:
# - domain: what the user is asking about
# - detail: which aspect of that domain they need
_RAG_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "scholarship": (
        "장학",
        "장학금",
        "국가장학금",
        "교내장학",
        "성적향상장학금",
        "학자금",
        "수혜",
        "중복 수혜",
    ),
    "course_registration": (
        "수강신청",
        "수강 신청",
        "수강정정",
        "수강 정정",
        "수강취소",
        "수강 취소",
        "강의 신청",
    ),
    "academic_calendar": (
        "학사일정",
        "학사 일정",
        "개강",
        "종강",
        "시험 기간",
        "성적 확인",
        "성적 공시",
    ),
    "academic_status": (
        "학적",
        "휴학",
        "복학",
        "자퇴",
        "제적",
        "재입학",
        "재학",
        "학적변동",
        "학적 변동",
    ),
    "major_change": (
        "전과",
        "전부",
        "전공변경",
        "전공 변경",
        "소속변경",
        "소속 변경",
    ),
    "multi_major": (
        "다전공",
        "복수전공",
        "복수 전공",
        "부전공",
        "연계전공",
        "융합전공",
        "마이크로전공",
    ),
    "admission_transfer": (
        "편입",
        "편입학",
        "입학",
        "신입학",
        "모집요강",
        "입시",
        "입학전형",
        "전형",
    ),
    "teaching_certification": (
        "교직",
        "교직이수",
        "교원자격",
        "교원 자격",
        "교원자격증",
        "교직과정",
        "교직 과정",
    ),
    "graduation": (
        "졸업",
        "졸업요건",
        "졸업 요건",
        "졸업학점",
        "졸업 학점",
        "전공 학점",
        "교양 학점",
        "이수 학점",
        "필수 이수",
        "졸업인증",
    ),
    "tuition": (
        "등록금",
        "납부",
        "분납",
        "환불",
        "고지서",
    ),
    "document_materials": (
        "자료실",
        "자료",
        "첨부파일",
        "첨부 파일",
        "양식",
        "서식",
        "신청서",
        "제출서류",
        "제출 서류",
        "pdf",
        "hwp",
        "hwpx",
        "docx",
    ),
    "student_life": (
        "학생생활",
        "학생 생활",
        "학생증",
        "동아리",
        "상담",
        "통학",
        "셔틀",
        "기숙사",
        "복지",
    ),
    "career_support": (
        "취업",
        "진로",
        "커리어",
        "현장실습",
        "인턴",
        "채용",
        "비교과",
        "취업지원",
    ),
    "international_exchange": (
        "교환학생",
        "국제교류",
        "파견",
        "해외파견",
        "복수학위",
        "어학연수",
        "유학",
        "해외 대학",
        "해외대학",
    ),
    "department_notice": (
        "학과",
        "전공",
        "단과대",
        "대학 공지",
        "학과 공지",
        "전공 공지",
        "청소년학과",
        "경영학과",
        "호텔경영",
        "스포츠과학",
        "입체조형",
        "모빌리티소프트웨어",
        "컴퓨터공학",
        "인공지능",
        "관광",
        "미디어영상",
        "애니메이션",
    ),
    "general_notice": (
        "공지",
        "공지사항",
        "학교 공지",
        "전체 공지",
        "안내",
        "모집",
        "선발",
        "접수",
        "결과 발표",
    ),
}
_RAG_DOMAIN_PRIORITY = {
    "scholarship": 5,
    "course_registration": 5,
    "academic_status": 5,
    "major_change": 5,
    "multi_major": 5,
    "graduation": 5,
    "tuition": 5,
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
_RAG_DETAIL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "period": ("기간", "일정", "언제", "마감", "시기"),
    "eligibility": ("대상", "자격", "조건", "가능", "지원 대상", "할 수 있어", "받을 수"),
    "procedure": ("신청", "절차", "방법", "접수", "어떻게"),
    "required_documents": (
        "서류",
        "제출서류",
        "제출 서류",
        "증빙",
        "첨부",
        "신청서",
        "양식",
        "서식",
    ),
    "benefit": ("금액", "혜택", "지원액", "수혜", "감면"),
    "announcement_lookup": ("공지", "안내", "모집", "모집요강", "결과 발표", "확인"),
    "summary": ("요약", "정리", "핵심"),
}
_RAG_DOMAIN_ALLOWED_DETAILS: dict[str, set[str]] = {
    "scholarship": {
        "period",
        "eligibility",
        "procedure",
        "required_documents",
        "benefit",
        "announcement_lookup",
        "summary",
        "unknown",
    },
    "tuition": {"period", "procedure", "required_documents", "benefit", "announcement_lookup", "unknown"},
    "course_registration": {"period", "procedure", "announcement_lookup", "summary", "unknown"},
    "academic_calendar": {"period", "announcement_lookup", "summary", "unknown"},
    "academic_status": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "unknown"},
    "major_change": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "unknown"},
    "multi_major": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "summary", "unknown"},
    "graduation": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "summary", "unknown"},
    "admission_transfer": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "unknown"},
    "teaching_certification": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "unknown"},
    "document_materials": {"required_documents", "announcement_lookup", "summary", "unknown"},
    "student_life": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "summary", "unknown"},
    "career_support": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "summary", "unknown"},
    "international_exchange": {"period", "eligibility", "procedure", "required_documents", "announcement_lookup", "summary", "unknown"},
    "department_notice": {"announcement_lookup", "summary", "unknown"},
    "general_notice": {"announcement_lookup", "summary", "unknown"},
}
_DEPARTMENT_SCOPE_KEYWORDS = tuple(
    keyword
    for keyword in _RAG_DOMAIN_KEYWORDS["department_notice"]
    if keyword != "전공"
)
MIN_GROUNDED_RESULT_SCORE = 0.18
MIN_RAG_INTENT_SCORE = 0.30


_RAG_FORCE_GROUP_KEYWORDS = (
    ("scholarship_support", (*_RAG_DOMAIN_KEYWORDS["scholarship"], *_RAG_DOMAIN_KEYWORDS["tuition"])),
    ("graduation_requirements", _RAG_DOMAIN_KEYWORDS["graduation"]),
    ("materials", _RAG_DOMAIN_KEYWORDS["document_materials"]),
    (
        "academic_schedule",
        (*_RAG_DOMAIN_KEYWORDS["course_registration"], *_RAG_DOMAIN_KEYWORDS["academic_calendar"]),
    ),
    ("career_support", _RAG_DOMAIN_KEYWORDS["career_support"]),
    ("student_life", _RAG_DOMAIN_KEYWORDS["student_life"]),
    ("international_exchange", _RAG_DOMAIN_KEYWORDS["international_exchange"]),
    ("department_sources", _RAG_DOMAIN_KEYWORDS["department_notice"]),
    ("university_notices", _RAG_DOMAIN_KEYWORDS["general_notice"]),
)


def answer_chat(user_input: str, db: Session) -> ChatResult:
    plan = decide_chat_plan(user_input)
    if len(plan.actions) > 1:
        return _answer_from_multi(user_input, plan.actions, db)

    decision = plan.actions[0]
    return _answer_for_decision(user_input, decision, db)


def _answer_for_decision(user_input: str, decision: ChatDecision, db: Session) -> ChatResult:
    atomic_input = decision.query or user_input

    if decision.route == "relational_db":
        return _answer_from_relational_db(atomic_input, decision, db)

    if decision.route == "rag":
        return _answer_from_rag(atomic_input, decision)

    if decision.route == "weather":
        return _answer_from_weather(atomic_input)

    return ChatResult(
        reply=get_gemini_response(atomic_input),
        intent="일반",
        route="llm",
    )


def decide_chat_route(user_input: str) -> ChatDecision:
    return decide_chat_plan(user_input).actions[0]


def decide_chat_plan(user_input: str) -> ChatPlan:
    compound_decisions = _compound_decisions(user_input)
    if len(compound_decisions) > 1:
        return _attach_rag_classification_to_plan(
            user_input,
            ChatPlan(
                actions=tuple(compound_decisions),
                reason="compound keyword match",
            ),
        )

    heuristic = _heuristic_decision(user_input)
    if heuristic.route == "relational_db" and heuristic.db_intent == "phone":
        return _attach_rag_classification_to_plan(
            user_input,
            ChatPlan(actions=(heuristic,), reason=heuristic.reason),
        )

    bert_decision = _klue_bert_decision(user_input)
    if bert_decision is not None:
        return _attach_rag_classification_to_plan(
            user_input,
            ChatPlan(actions=(bert_decision,), reason=bert_decision.reason),
        )

    if not settings.intent_classifier_model_name and heuristic.route != "llm":
        return _attach_rag_classification_to_plan(
            user_input,
            ChatPlan(actions=(heuristic,), reason=heuristic.reason),
        )

    prompt = f"""
You classify a user question for a university assistant.
Return only valid JSON with this schema:
{{"actions":[{{"query":"atomic user question","route":"llm|relational_db|rag|weather","db_intent":"map|phone|unknown"}}],"reason":"short reason"}}

Routing rules:
- llm: basic general knowledge or casual conversation that does not need local data.
- relational_db: exact campus data stored in relational DB, such as place locations or phone numbers.
- rag: information that must be grounded in crawled documents, notices, policies, schedules, or other text sources.
- weather: current or forecast weather questions that need live weather API data.
- If the user asks for multiple independent things, split them into atomic queries and return multiple actions in the order they should be answered.
- Use relational_db with db_intent map for campus location/path requests.
- Use relational_db with db_intent phone for phone number/contact requests.

User question:
{user_input}
"""
    raw = get_gemini_response(prompt)
    parsed = _parse_decision_plan(raw)
    if parsed is None:
        return _attach_rag_classification_to_plan(
            user_input,
            ChatPlan(actions=(heuristic,), reason=heuristic.reason),
        )
    return _attach_rag_classification_to_plan(user_input, parsed)


def _klue_bert_decision(user_input: str) -> ChatDecision | None:
    prediction = classify_with_klue_bert(user_input)
    if prediction is None:
        return None

    if prediction.confidence < settings.intent_classifier_confidence_threshold:
        return None

    return ChatDecision(
        route=prediction.route,
        db_intent=prediction.db_intent,
        reason=f"klue-bert:{prediction.label}:{prediction.confidence:.3f}",
    )


def _answer_from_relational_db(
    user_input: str,
    decision: ChatDecision,
    db: Session,
) -> ChatResult:
    db_intent = decision.db_intent
    if db_intent == "unknown":
        db_intent = _infer_db_intent(user_input)

    if db_intent == "phone":
        reply = get_phone(user_input, db)
        source_title = "kgu_contacts"
        intent = "전화"
    elif db_intent == "map":
        reply = get_map_response(user_input, db)
        source_title = "kgu_places"
        intent = "지도"
    else:
        reply = (
            "정확한 DB 정보가 필요한 질문으로 판단되지만, 어떤 DB에서 찾을지 "
            "결정하지 못했습니다. 장소 위치나 전화번호처럼 더 구체적으로 질문해 주세요."
        )
        source_title = "relational_db"
        intent = "DB"

    return ChatResult(
        reply=reply,
        intent=intent,
        route="relational_db",
        sources=[ChatSource(type="relational_db", title=source_title)],
    )


def _answer_from_rag(user_input: str, decision: ChatDecision) -> ChatResult:
    try:
        search_parameters = inspect.signature(search_documents).parameters
        search_kwargs = {"query": user_input, "top_k": 5}
        if "rag_domain" in search_parameters:
            search_kwargs["rag_domain"] = decision.rag_domain
        if "rag_domains" in search_parameters:
            search_kwargs["rag_domains"] = list(decision.rag_domains)
        if "rag_detail" in search_parameters:
            search_kwargs["rag_detail"] = decision.rag_detail
        if "source_scope" in search_parameters:
            search_kwargs["source_scope"] = decision.source_scope
        results = search_documents(**search_kwargs)
    except NotImplementedError:
        return ChatResult(
            reply=(
                "이 질문은 문서 기반 검색(RAG)으로 답해야 하지만, 현재 검색 파이프라인이 "
                "연결되어 있지 않습니다."
            ),
            intent="RAG",
            route="rag",
            rag_domain=decision.rag_domain,
            rag_detail=decision.rag_detail,
            rag_domains=decision.rag_domains,
            source_scope=decision.source_scope,
            rag_confidence=decision.rag_confidence,
            matched_keywords=decision.matched_keywords,
            intent_scores=decision.intent_scores,
            answer_status="insufficient",
            unverified=(_unverified_reason(decision),),
        )

    if not results:
        return ChatResult(
            reply="관련 문서를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요.",
            intent="RAG",
            route="rag",
            rag_domain=decision.rag_domain,
            rag_domains=decision.rag_domains,
            rag_detail=decision.rag_detail,
            source_scope=decision.source_scope,
            rag_confidence=decision.rag_confidence,
            matched_keywords=decision.matched_keywords,
            intent_scores=decision.intent_scores,
            answer_status="insufficient",
            unverified=(_unverified_reason(decision),),
        )

    grounded_results = [
        result for result in results if result.score >= MIN_GROUNDED_RESULT_SCORE
    ]
    if not grounded_results:
        return ChatResult(
            reply=_insufficient_rag_reply(decision),
            intent="RAG",
            route="rag",
            sources=_chat_sources_from_results(results),
            rag_domain=decision.rag_domain,
            rag_domains=decision.rag_domains,
            rag_detail=decision.rag_detail,
            source_scope=decision.source_scope,
            rag_confidence=decision.rag_confidence,
            matched_keywords=decision.matched_keywords,
            intent_scores=decision.intent_scores,
            answer_status="insufficient",
            unverified=(_unverified_reason(decision),),
        )

    answer_status: Literal["answered", "partial", "insufficient"] = (
        "partial" if len(grounded_results) < len(results) else "answered"
    )
    context = _format_rag_context(grounded_results)
    reply = get_gemini_response_with_context(user_input=user_input, context=context)
    sources = _chat_sources_from_results(results)
    return ChatResult(
        reply=reply,
        intent="RAG",
        route="rag",
        sources=sources,
        rag_domain=decision.rag_domain,
        rag_domains=decision.rag_domains,
        rag_detail=decision.rag_detail,
        source_scope=decision.source_scope,
        rag_confidence=decision.rag_confidence,
        matched_keywords=decision.matched_keywords,
        intent_scores=decision.intent_scores,
        answer_status=answer_status,
        unverified=() if answer_status == "answered" else (_unverified_reason(decision),),
    )


def _answer_from_weather(user_input: str) -> ChatResult:
    report = get_weather_response(user_input)
    return ChatResult(
        reply=report.reply,
        intent="날씨",
        route="weather",
        sources=[
            ChatSource(
                type="weather_api",
                title=f"Open-Meteo forecast: {report.location_name}",
                source_url=report.source_url,
            )
        ],
    )


def _answer_from_multi(
    user_input: str,
    actions: tuple[ChatDecision, ...],
    db: Session,
) -> ChatResult:
    results = [_answer_for_decision(user_input, action, db) for action in actions]
    replies = [result.reply.strip() for result in results if result.reply.strip()]
    sources: list[ChatSource] = []
    seen_sources: set[tuple[str, str, str | None]] = set()

    for result in results:
        for source in result.sources:
            key = (source.type, source.title, source.source_url)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            sources.append(source)

    return ChatResult(
        reply="\n\n".join(replies),
        intent="복합",
        route="multi",
        sources=sources,
    )


def _heuristic_decision(user_input: str) -> ChatDecision:
    normalized = _normalize_query(user_input)
    rag_reason = _matched_rag_group(normalized)

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        return ChatDecision(route="weather", reason="weather keyword")
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="phone", reason="phone keyword")
    if _looks_like_db_lookup(normalized):
        return ChatDecision(route="relational_db", reason="db lookup keyword")
    if rag_reason and _looks_like_source_lookup(normalized):
        return ChatDecision(route="rag", reason=rag_reason)
    if _looks_like_rag_query(normalized):
        return ChatDecision(route="rag", reason="rag keyword")
    if _contains_any(normalized, _MAP_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="map", reason="map keyword")

    if rag_reason:
        return ChatDecision(route="rag", reason=rag_reason)
    return ChatDecision(route="llm", reason="default")


def _compound_decisions(user_input: str) -> list[ChatDecision]:
    segmented_decisions: list[ChatDecision] = []
    for atomic_query in _split_atomic_queries(user_input):
        segmented_decisions.extend(
            _compound_decisions_for_atomic_query(
                atomic_query,
                reason_prefix="compound segment",
            )
        )

    deduped_segmented_decisions = _dedupe_decisions(segmented_decisions)
    if len(deduped_segmented_decisions) > 1:
        return deduped_segmented_decisions

    return _dedupe_decisions(
        _compound_decisions_for_atomic_query(
            user_input,
            reason_prefix="compound",
        )
    )


def _compound_decisions_for_atomic_query(
    query: str,
    reason_prefix: str,
) -> list[ChatDecision]:
    normalized = _normalize_query(query)
    decisions: list[ChatDecision] = []
    rag_reason = _matched_rag_group(normalized)
    is_source_lookup = bool(rag_reason and _looks_like_source_lookup(normalized))

    if _contains_any(normalized, _LOCATION_REQUEST_KEYWORDS) and not is_source_lookup:
        decisions.append(
            ChatDecision(
                route="relational_db",
                db_intent="map",
                reason=f"{reason_prefix} map keyword",
                query=query,
            )
        )
    if _contains_any(normalized, _PHONE_KEYWORDS):
        decisions.append(
            ChatDecision(
                route="relational_db",
                db_intent="phone",
                reason=f"{reason_prefix} phone keyword",
                query=query,
            )
        )

    if rag_reason:
        decisions.append(ChatDecision(route="rag", reason=rag_reason, query=query))

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        decisions.append(
            ChatDecision(route="weather", reason=f"{reason_prefix} weather keyword", query=query)
        )

    return decisions


def _dedupe_decisions(decisions: list[ChatDecision]) -> list[ChatDecision]:
    deduped: list[ChatDecision] = []
    seen: set[tuple[str, str, str]] = set()
    for decision in decisions:
        key = (decision.route, decision.db_intent, _normalize_query(decision.query or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(decision)
    return deduped


def _infer_db_intent(user_input: str) -> DbIntent:
    normalized = _normalize_query(user_input)
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return "phone"
    if _contains_any(normalized, _MAP_KEYWORDS):
        return "map"
    return "unknown"


def _normalize_query(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input.strip().lower())


def _split_atomic_queries(user_input: str) -> list[str]:
    chunks = [
        chunk.strip()
        for chunk in re.split(
            r"(?:[?？!！]+|[,，;；]+|\s+(?:그리고|또|또한|및|겸|하고)\s+)",
            user_input,
        )
        if chunk.strip()
    ]
    return chunks or [user_input.strip()]


def _contains_any(normalized_text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in normalized_text for keyword in keywords)


def _looks_like_information_lookup(normalized_text: str) -> bool:
    return _contains_any(normalized_text, _INFORMATION_LOOKUP_KEYWORDS)


def _looks_like_db_lookup(normalized_text: str) -> bool:
    return _contains_any(normalized_text, _DB_LOOKUP_KEYWORDS)


def _looks_like_source_lookup(normalized_text: str) -> bool:
    return _contains_any(
        normalized_text,
        ("어디", "확인", "다운로드", "나와", "볼 수", "찾을 수"),
    )


def _matched_rag_group(normalized_text: str) -> str | None:
    for group, keywords in _RAG_FORCE_GROUP_KEYWORDS:
        if _contains_any(normalized_text, keywords):
            return f"rag keyword: {group}"


def _looks_like_rag_query(normalized_text: str) -> bool:
    return any(
        _contains_any(normalized_text, keywords)
        for keywords in _RAG_DOMAIN_KEYWORDS.values()
    )


def _attach_rag_classification_to_plan(user_input: str, plan: ChatPlan) -> ChatPlan:
    return ChatPlan(
        actions=tuple(
            _attach_rag_classification(action.query or user_input, action)
            for action in plan.actions
        ),
        reason=plan.reason,
    )


def _attach_rag_classification(user_input: str, decision: ChatDecision) -> ChatDecision:
    if decision.route != "rag":
        return decision

    rag_classification = _classify_rag_query(_normalize_query(user_input))
    if rag_classification is None:
        return ChatDecision(
            route=decision.route,
            db_intent=decision.db_intent,
            reason=decision.reason,
            query=decision.query,
            rag_domain="unknown",
            rag_domains=(),
            rag_detail="unknown",
            source_scope="unknown",
            rag_confidence=0.0,
            matched_keywords=(),
            intent_scores=(),
        )

    return ChatDecision(
        route=decision.route,
        db_intent=decision.db_intent,
        reason=decision.reason,
        query=decision.query,
        rag_domain=rag_classification.domain,
        rag_domains=rag_classification.domains,
        rag_detail=rag_classification.detail,
        source_scope=rag_classification.source_scope,
        rag_confidence=rag_classification.confidence,
        matched_keywords=rag_classification.matched_keywords,
        intent_scores=rag_classification.intent_scores,
    )


def _classify_rag_query(normalized_text: str) -> RagClassification | None:
    source_scope = _classify_source_scope(normalized_text)
    matches = [
        (
            domain,
            _matched_keywords(normalized_text, keywords),
            _RAG_DOMAIN_PRIORITY.get(domain, 0),
        )
        for domain, keywords in _RAG_DOMAIN_KEYWORDS.items()
    ]
    specific_matches = [
        match
        for match in matches
        if match[0] not in {"general_notice", "department_notice"} and len(match[1]) > 0
    ]
    if not specific_matches:
        specific_matches = [
            match for match in matches if match[0] != "general_notice" and len(match[1]) > 0
        ]
    if specific_matches:
        matches = specific_matches
    domain, domain_keywords, _priority = max(matches, key=lambda item: (len(item[1]), item[2]))
    count = len(domain_keywords)
    if count <= 0:
        return None
    intent_scores = _rank_rag_intents(matches)

    detail_matches = [
        (
            detail,
            _matched_keywords(normalized_text, keywords),
        )
        for detail, keywords in _RAG_DETAIL_KEYWORDS.items()
    ]
    detail, detail_keywords = max(detail_matches, key=lambda item: len(item[1]))
    detail_count = len(detail_keywords)
    raw_detail = detail if detail_count > 0 else "unknown"
    detail = _normalize_detail_for_domain(domain=domain, detail=raw_detail)
    if detail == "unknown":
        detail_keywords = ()
    confidence = _rag_confidence(
        selected_domain_count=count,
        selected_detail_count=detail_count if detail != "unknown" else 0,
        competing_domain_count=_second_highest_domain_count(matches, selected_domain=domain),
        source_scope=source_scope,
    )
    matched_keywords = tuple(dict.fromkeys((*domain_keywords, *detail_keywords)))
    if source_scope == "department" and "department_notice" not in {
        score.domain for score in intent_scores
    }:
        intent_scores.append(
            RagIntentScore(
                domain="department_notice",
                score=MIN_RAG_INTENT_SCORE,
                matched_keywords=_matched_keywords(
                    normalized_text,
                    _DEPARTMENT_SCOPE_KEYWORDS,
                ),
            )
        )
        intent_scores = sorted(intent_scores, key=lambda item: item.score, reverse=True)[:3]

    return RagClassification(
        domain=domain,
        domains=tuple(score.domain for score in intent_scores),
        detail=detail,
        source_scope=source_scope,
        confidence=confidence,
        matched_keywords=matched_keywords,
        intent_scores=tuple(intent_scores),
    )


def _rank_rag_intents(
    matches: list[tuple[str, tuple[str, ...], int]],
) -> list[RagIntentScore]:
    scored: list[RagIntentScore] = []
    max_priority = max(_RAG_DOMAIN_PRIORITY.values(), default=1)
    for domain, keywords, priority in matches:
        if not keywords:
            continue
        keyword_score = min(len(keywords) / 3, 1.0)
        priority_score = priority / max_priority
        score = round((keyword_score * 0.75) + (priority_score * 0.25), 3)
        if score < MIN_RAG_INTENT_SCORE:
            continue
        scored.append(
            RagIntentScore(
                domain=domain,
                score=score,
                matched_keywords=keywords,
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)[:3]


def _classify_source_scope(normalized_text: str) -> str:
    if _contains_any(normalized_text, _DEPARTMENT_SCOPE_KEYWORDS):
        return "department"
    if _contains_any(normalized_text, ("학교 전체", "전체 공지", "대학 공지", "경기대 공지")):
        return "university"
    return "unknown"


def _matched_keywords(normalized_text: str, keywords: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(keyword for keyword in keywords if keyword.lower() in normalized_text)


def _normalize_detail_for_domain(*, domain: str, detail: str) -> str:
    allowed_details = _RAG_DOMAIN_ALLOWED_DETAILS.get(domain)
    if allowed_details is None or detail in allowed_details:
        return detail
    return "unknown"


def _second_highest_domain_count(
    matches: list[tuple[str, tuple[str, ...], int]],
    *,
    selected_domain: str,
) -> int:
    counts = [len(keywords) for domain, keywords, _priority in matches if domain != selected_domain]
    return max(counts, default=0)


def _rag_confidence(
    *,
    selected_domain_count: int,
    selected_detail_count: int,
    competing_domain_count: int,
    source_scope: str,
) -> float:
    confidence = 0.45
    confidence += min(selected_domain_count, 3) * 0.12
    if selected_detail_count > 0:
        confidence += min(selected_detail_count, 2) * 0.08
    if source_scope != "unknown":
        confidence += 0.04
    if competing_domain_count >= selected_domain_count:
        confidence -= 0.12
    elif competing_domain_count > 0:
        confidence -= 0.05
    return round(min(max(confidence, 0.0), 0.95), 3)


def _parse_decision(raw: str) -> ChatDecision | None:
    if not raw:
        return None

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    route = payload.get("route")
    db_intent = payload.get("db_intent", "unknown")
    reason = str(payload.get("reason", ""))

    if route not in {"llm", "relational_db", "rag", "weather"}:
        return None
    if db_intent not in {"map", "phone", "unknown"}:
        db_intent = "unknown"

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        query = None

    return ChatDecision(route=route, db_intent=db_intent, reason=reason, query=query)


def _parse_decision_plan(raw: str) -> ChatPlan | None:
    if not raw:
        return None

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    reason = str(payload.get("reason", ""))
    raw_actions = payload.get("actions")
    if raw_actions is None:
        single_decision = _decision_from_payload(payload)
        if single_decision is None:
            return None
        return ChatPlan(actions=(single_decision,), reason=single_decision.reason)

    if not isinstance(raw_actions, list) or not raw_actions:
        return None

    actions: list[ChatDecision] = []
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            return None
        action = _decision_from_payload(raw_action, default_reason=reason)
        if action is None:
            return None
        actions.append(action)

    deduped = _dedupe_decisions(actions)
    if not deduped:
        return None
    return ChatPlan(actions=tuple(deduped), reason=reason)


def _decision_from_payload(
    payload: dict[str, object],
    default_reason: str = "",
) -> ChatDecision | None:
    route = payload.get("route")
    db_intent = payload.get("db_intent", "unknown")
    reason = str(payload.get("reason", default_reason))
    query = payload.get("query")

    if route not in {"llm", "relational_db", "rag", "weather"}:
        return None
    if db_intent not in {"map", "phone", "unknown"}:
        db_intent = "unknown"
    if route != "relational_db":
        db_intent = "unknown"
    if not isinstance(query, str) or not query.strip():
        query = None

    return ChatDecision(route=route, db_intent=db_intent, reason=reason, query=query)  # type: ignore[arg-type]


def _chat_sources_from_results(results: list[SearchResult]) -> list[ChatSource]:
    return [
        ChatSource(
            type="document",
            title=result.title,
            source_url=result.source_url,
            score=result.score,
        )
        for result in results
    ]


def _insufficient_rag_reply(decision: ChatDecision) -> str:
    domain_text = _domain_label(decision.rag_domain)
    detail_text = _detail_label(decision.rag_detail)
    if domain_text and detail_text:
        return (
            f"{domain_text} 관련 {detail_text} 질문으로 판단되지만, "
            "현재 검색된 자료에서는 답변에 필요한 근거를 확인할 수 없습니다."
        )
    if domain_text:
        return (
            f"{domain_text} 관련 질문으로 판단되지만, 현재 검색된 자료에서는 "
            "답변에 필요한 근거를 확인할 수 없습니다."
        )
    return "현재 검색된 자료에서는 답변에 필요한 근거를 확인할 수 없습니다."


def _unverified_reason(decision: ChatDecision) -> str:
    domain_text = _domain_label(decision.rag_domain) or "질문 의도"
    detail_text = _detail_label(decision.rag_detail)
    if detail_text:
        return f"{domain_text}의 {detail_text}에 대한 직접 근거"
    return f"{domain_text}에 대한 직접 근거"


def _domain_label(domain: str | None) -> str | None:
    labels = {
        "scholarship": "장학",
        "course_registration": "수강신청",
        "academic_calendar": "학사일정",
        "academic_status": "학적",
        "major_change": "전과",
        "multi_major": "다전공",
        "admission_transfer": "입학/편입",
        "teaching_certification": "교직",
        "graduation": "졸업",
        "tuition": "등록금",
        "document_materials": "자료/서식",
        "student_life": "학생생활",
        "career_support": "진로/취업",
        "international_exchange": "국제교류",
        "department_notice": "학과 공지",
        "general_notice": "일반 공지",
    }
    if domain is None or domain == "unknown":
        return None
    return labels.get(domain, domain)


def _detail_label(detail: str | None) -> str | None:
    labels = {
        "period": "기간",
        "eligibility": "대상/자격",
        "procedure": "신청 절차",
        "required_documents": "제출 서류",
        "benefit": "혜택",
        "announcement_lookup": "공지 확인",
        "summary": "요약",
    }
    if detail is None or detail == "unknown":
        return None
    return labels.get(detail, detail)


def _format_rag_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {result.title}",
                    f"source_url: {result.source_url}",
                    f"score: {result.score}",
                    result.text,
                ]
            )
        )
    return "\n\n".join(blocks)
