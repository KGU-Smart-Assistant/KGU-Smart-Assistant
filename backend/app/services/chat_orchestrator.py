from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.services.call_service import get_phone
from app.services.gemini_service import get_gemini_response
from app.services.langchain_rag_service import answer_with_langchain_rag
from app.services.map_service import get_map_response
from app.services.weather_service import get_weather_response

ChatRoute = Literal["llm", "relational_db", "rag", "weather"]
DbIntent = Literal["map", "phone", "unknown"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatDecision:
    route: ChatRoute
    db_intent: DbIntent = "unknown"
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


_PHONE_KEYWORDS = (
    "전화",
    "전화번호",
    "연락처",
    "문의처",
    "사무실",
    "행정실",
    "학과사무실",
)
_MAP_KEYWORDS = (
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
_INFO_SOURCE_PHRASES = (
    "어디서 찾",
    "어디에서 찾",
    "어디서 확인",
    "어디에서 확인",
    "어디에 나와",
    "어디에 있어",
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

_RAG_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "academic_schedule": (
        "학사일정",
        "수강신청",
        "수강 정정",
        "수강취소",
        "등록 기간",
        "개강",
        "종강",
        "시험 기간",
        "성적 확인",
        "성적 공시",
    ),
    "graduation_requirements": (
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
    "university_notices": (
        "공지",
        "공지사항",
        "학교 공지",
        "전체 공지",
        "안내",
        "모집",
        "선발",
        "신청",
        "접수",
        "제출",
        "대상자",
        "자격",
        "기간",
        "결과 발표",
    ),
    "scholarship_support": (
        "장학",
        "장학금",
        "국가장학금",
        "교내장학",
        "성적우수장학금",
        "학자금",
        "등록금",
        "혜택",
        "중복 혜택",
        "신청 불가",
    ),
    "materials": (
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
        "식당",
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
    "department_sources": (
        "학과",
        "전공",
        "학과별",
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
}
_RAG_GROUP_PRIORITY = {
    "academic_schedule": 3,
    "graduation_requirements": 3,
    "scholarship_support": 3,
    "materials": 3,
    "student_life": 3,
    "career_support": 3,
    "department_sources": 2,
    "university_notices": 1,
}
_RAG_FORCE_GROUP_KEYWORDS = (
    ("scholarship_support", ("장학", "장학금", "국가장학금", "성적우수장학금")),
    ("graduation_requirements", ("졸업", "졸업요건", "졸업학점")),
    ("materials", ("자료실", "첨부파일", "첨부 파일", "양식", "서식", "신청서")),
    ("academic_schedule", ("학사일정", "수강신청", "개강", "종강")),
    ("career_support", ("취업", "진로", "현장실습", "인턴", "채용")),
    ("student_life", ("학생생활", "학생증", "동아리", "기숙사", "식당")),
)


def answer_chat(user_input: str, db: Session) -> ChatResult:
    decision = decide_chat_route(user_input)

    if decision.route == "relational_db":
        return _answer_from_relational_db(user_input, decision, db)
    if decision.route == "rag":
        return _answer_from_rag(user_input)
    if decision.route == "weather":
        return _answer_from_weather(user_input)

    return ChatResult(reply=_safe_llm_reply(user_input), intent="일반", route="llm")


def decide_chat_route(user_input: str) -> ChatDecision:
    heuristic = _heuristic_decision(user_input)
    if heuristic.route != "llm":
        return heuristic

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
    try:
        raw = get_gemini_response(prompt)
    except Exception:
        logger.exception("Failed to classify chat route with LLM")
        return heuristic

    parsed = _parse_decision(raw)
    if parsed is None:
        return heuristic
    return parsed


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
            "정확한 DB 정보가 필요한 질문으로 판단했지만 어떤 DB에서 찾을지 결정하지 못했습니다. "
            "장소 위치나 전화번호처럼 더 구체적으로 질문해 주세요."
        )
        source_title = "relational_db"
        intent = "DB"

    return ChatResult(
        reply=reply,
        intent=intent,
        route="relational_db",
        sources=[ChatSource(type="relational_db", title=source_title)],
    )


def _answer_from_rag(user_input: str) -> ChatResult:
    try:
        rag_result = answer_with_langchain_rag(user_input, top_k=5)
    except NotImplementedError:
        logger.info("Document search pipeline is not implemented")
        return ChatResult(
            reply="문서 기반 검색 파이프라인이 아직 연결되어 있지 않습니다.",
            intent="RAG",
            route="rag",
        )
    except Exception:
        logger.exception("LangChain RAG chain failed; falling back to LLM")
        return ChatResult(
            reply=_safe_llm_reply(user_input),
            intent="일반",
            route="llm",
        )

    if not rag_result.documents:
        return ChatResult(reply=rag_result.reply, intent="RAG", route="rag")

    sources = [
        ChatSource(
            type="document",
            title=str(document.metadata.get("title", "Untitled")),
            source_url=document.metadata.get("source_url"),
            score=document.metadata.get("score"),
        )
        for document in rag_result.documents
    ]
    return ChatResult(reply=rag_result.reply, intent="RAG", route="rag", sources=sources)


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


def _safe_llm_reply(user_input: str) -> str:
    try:
        return get_gemini_response(user_input)
    except Exception:
        logger.exception("LLM answer failed")
        return "지금은 답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."


def _heuristic_decision(user_input: str) -> ChatDecision:
    normalized = _normalize_query(user_input)

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        return ChatDecision(route="weather", reason="weather keyword")
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="phone", reason="phone keyword")

    rag_reason = _matched_rag_group(normalized)
    if rag_reason and _contains_any(normalized, _INFO_SOURCE_PHRASES):
        return ChatDecision(route="rag", reason=f"{rag_reason}; info source phrase")

    if _contains_any(normalized, _MAP_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="map", reason="map keyword")

    if rag_reason:
        return ChatDecision(route="rag", reason=rag_reason)
    return ChatDecision(route="llm", reason="default")


def _infer_db_intent(user_input: str) -> DbIntent:
    normalized = _normalize_query(user_input)
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return "phone"
    if _contains_any(normalized, _MAP_KEYWORDS):
        return "map"
    return "unknown"


def _normalize_query(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input.strip().casefold())


def _contains_any(normalized_text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold() in normalized_text for keyword in keywords)


def _matched_rag_group(normalized_text: str) -> str | None:
    for group, keywords in _RAG_FORCE_GROUP_KEYWORDS:
        if _contains_any(normalized_text, keywords):
            return f"rag keyword: {group}"

    matches = [
        (
            group,
            sum(keyword.casefold() in normalized_text for keyword in keywords),
            _RAG_GROUP_PRIORITY.get(group, 0),
        )
        for group, keywords in _RAG_KEYWORD_GROUPS.items()
    ]
    group, count, _priority = max(matches, key=lambda item: (item[1], item[2]))
    if count <= 0:
        return None
    return f"rag keyword: {group}"


def _parse_decision(raw: str) -> ChatDecision | None:
    if not raw:
        return None

    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
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
