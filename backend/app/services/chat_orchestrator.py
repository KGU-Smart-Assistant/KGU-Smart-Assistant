from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.schemas.search import SearchResult
from app.services.call_service import get_phone
from app.services.gemini_service import (
    get_gemini_response,
    get_gemini_response_with_context,
)
from app.services.map_service import get_map_response
from app.services.search_service import search_documents
from app.services.weather_service import get_weather_response

ChatRoute = Literal["llm", "relational_db", "rag", "weather"]
DbIntent = Literal["map", "phone", "unknown"]


@dataclass(frozen=True)
class ChatDecision:
    route: ChatRoute
    db_intent: DbIntent = "unknown"
    reason: str = ""
    rag_domain: str | None = None
    rag_detail: str | None = None
    source_scope: str | None = None


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
    rag_detail: str | None = None
    source_scope: str | None = None


@dataclass(frozen=True)
class RagClassification:
    domain: str
    detail: str = "unknown"
    source_scope: str = "unknown"


_PHONE_KEYWORDS = (
    "전화",
    "전화번호",
    "연락처",
    "문의처",
    "사무실 번호",
    "행정실 번호",
)
_MAP_KEYWORDS = (
    "어디",
    "위치",
    "찾아가",
    "가는 길",
    "어떻게 가",
    "길찾기",
    "캠퍼스맵",
    "강의실",
    "도서관",
    "학생회관",
    "공학관",
    "복지관",
    "박물관",
    "정문",
    "후문",
)
_WEATHER_KEYWORDS = (
    "날씨",
    "기온",
    "강수",
    "비 와",
    "비가",
    "비올",
    "눈 와",
    "더워",
    "추워",
    "예보",
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
        "휴학",
        "복학",
        "휴학 일정",
        "복학 일정",
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
    "graduation": 5,
    "tuition": 5,
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
    "eligibility": ("대상", "자격", "조건", "가능", "지원 대상"),
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
    "announcement_lookup": ("공지", "안내", "모집", "결과 발표", "확인"),
    "summary": ("요약", "정리", "핵심"),
}
_DEPARTMENT_SCOPE_KEYWORDS = _RAG_DOMAIN_KEYWORDS["department_notice"]


def answer_chat(user_input: str, db: Session) -> ChatResult:
    decision = decide_chat_route(user_input)

    if decision.route == "relational_db":
        return _answer_from_relational_db(user_input, decision, db)

    if decision.route == "rag":
        return _answer_from_rag(user_input, decision)

    if decision.route == "weather":
        return _answer_from_weather(user_input)

    return ChatResult(
        reply=get_gemini_response(user_input),
        intent="일반",
        route="llm",
    )


def decide_chat_route(user_input: str) -> ChatDecision:
    heuristic = _heuristic_decision(user_input)
    if heuristic.route != "llm":
        return _attach_rag_classification(user_input, heuristic)

    prompt = f"""
You classify a user question for a university assistant.
Return only valid JSON with this schema:
{{"route":"llm|relational_db|rag|weather","db_intent":"map|phone|unknown","reason":"short reason"}}

Routing rules:
- llm: basic general knowledge or casual conversation that does not need local data.
- relational_db: exact campus data stored in relational DB, such as place locations or phone numbers.
- rag: information that must be grounded in crawled documents, notices, policies, schedules, or other text sources.
- weather: current or forecast weather questions that need live weather API data.

User question:
{user_input}
"""
    raw = get_gemini_response(prompt)
    parsed = _parse_decision(raw)
    if parsed is None:
        return _attach_rag_classification(user_input, heuristic)
    return _attach_rag_classification(user_input, parsed)


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
        results = search_documents(
            query=user_input,
            top_k=5,
            rag_domain=decision.rag_domain,
            rag_detail=decision.rag_detail,
            source_scope=decision.source_scope,
        )
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
            source_scope=decision.source_scope,
        )

    if not results:
        return ChatResult(
            reply="관련 문서를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요.",
            intent="RAG",
            route="rag",
            rag_domain=decision.rag_domain,
            rag_detail=decision.rag_detail,
            source_scope=decision.source_scope,
        )

    context = _format_rag_context(results)
    reply = get_gemini_response_with_context(user_input=user_input, context=context)
    sources = [
        ChatSource(
            type="document",
            title=result.title,
            source_url=result.source_url,
            score=result.score,
        )
        for result in results
    ]
    return ChatResult(
        reply=reply,
        intent="RAG",
        route="rag",
        sources=sources,
        rag_domain=decision.rag_domain,
        rag_detail=decision.rag_detail,
        source_scope=decision.source_scope,
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


def _heuristic_decision(user_input: str) -> ChatDecision:
    normalized = _normalize_query(user_input)

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        return ChatDecision(route="weather", reason="weather keyword")
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="phone", reason="phone keyword")

    if _looks_like_rag_query(normalized):
        return ChatDecision(route="rag", reason="rag keyword")
    if _contains_any(normalized, _MAP_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="map", reason="map keyword")
    return ChatDecision(route="llm", reason="default")


def _infer_db_intent(user_input: str) -> DbIntent:
    normalized = _normalize_query(user_input)
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return "phone"
    if _contains_any(normalized, _MAP_KEYWORDS):
        return "map"
    return "unknown"


def _normalize_query(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input.strip().lower())


def _contains_any(normalized_text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in normalized_text for keyword in keywords)


def _looks_like_rag_query(normalized_text: str) -> bool:
    return any(
        _contains_any(normalized_text, keywords)
        for keywords in _RAG_DOMAIN_KEYWORDS.values()
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
            rag_domain="unknown",
            rag_detail="unknown",
            source_scope="unknown",
        )

    return ChatDecision(
        route=decision.route,
        db_intent=decision.db_intent,
        reason=decision.reason,
        rag_domain=rag_classification.domain,
        rag_detail=rag_classification.detail,
        source_scope=rag_classification.source_scope,
    )


def _classify_rag_query(normalized_text: str) -> RagClassification | None:
    source_scope = _classify_source_scope(normalized_text)
    matches = [
        (
            domain,
            sum(keyword.lower() in normalized_text for keyword in keywords),
            _RAG_DOMAIN_PRIORITY.get(domain, 0),
        )
        for domain, keywords in _RAG_DOMAIN_KEYWORDS.items()
    ]
    specific_matches = [
        match
        for match in matches
        if match[0] not in {"general_notice", "department_notice"} and match[1] > 0
    ]
    if not specific_matches:
        specific_matches = [
            match for match in matches if match[0] != "general_notice" and match[1] > 0
        ]
    if specific_matches:
        matches = specific_matches
    domain, count, _priority = max(matches, key=lambda item: (item[1], item[2]))
    if count <= 0:
        return None

    detail_matches = [
        (
            detail,
            sum(keyword.lower() in normalized_text for keyword in keywords),
        )
        for detail, keywords in _RAG_DETAIL_KEYWORDS.items()
    ]
    detail, detail_count = max(detail_matches, key=lambda item: item[1])
    return RagClassification(
        domain=domain,
        detail=detail if detail_count > 0 else "unknown",
        source_scope=source_scope,
    )


def _classify_source_scope(normalized_text: str) -> str:
    if _contains_any(normalized_text, _DEPARTMENT_SCOPE_KEYWORDS):
        return "department"
    if _contains_any(normalized_text, ("학교 전체", "전체 공지", "대학 공지", "경기대 공지")):
        return "university"
    return "unknown"


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

    return ChatDecision(route=route, db_intent=db_intent, reason=reason)


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
