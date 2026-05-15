from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
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
    rag_category: str | None = None
    rag_detail: str | None = None


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


_PHONE_KEYWORDS = ("전화", "전화번호", "연락처", "문의처", "담당부서 전화", "학과사무실")
_MAP_KEYWORDS = ("위치", "찾아가", "가는 길", "어떻게 가", "길찾기", "캠퍼스맵", "강의실", "예술관", "학생회관", "공학관", "복지관", "박물관", "정문", "후문")
_WEATHER_KEYWORDS = ("날씨", "기온", "강수", "비 와", "비가", "비올", "더워", "추워", "예보")

_RAG_DOMAIN_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("scholarship", "scholarship", ("장학", "장학금", "국가장학금", "교내장학", "성적향상장학금", "학자금", "수혜", "감면")),
    ("tuition", "tuition", ("등록금 납부", "등록금 분납", "등록금 환불", "고지서", "납부", "분납", "환불")),
    ("course_registration", "course_registration", ("수강신청", "수강 정정", "수강정정", "수강취소", "수기수강", "수강 철회")),
    ("academic_calendar", "academic_calendar", ("학사일정", "개강", "종강", "중간고사", "기말고사", "시험 기간", "휴학", "복학", "성적공시", "강의평가")),
    ("graduation", "graduation", ("졸업", "졸업요건", "졸업 학점", "전공 학점", "교양 학점", "졸업인증", "학위")),
    ("document_materials", "document_materials", ("자료실", "첨부파일", "파일", "양식", "서식", "신청서", "제출서류", "pdf", "hwp", "docx")),
    ("student_life", "student_life", ("학생생활", "학생증", "동아리", "상담", "통학", "셔틀", "기숙사", "생활관", "복지", "식당", "편의시설")),
    ("career_support", "career_support", ("취업", "진로", "커리어", "현장실습", "인턴", "채용", "비교과", "취업지원")),
    ("department_notice", "department_notice", ("학과 공지", "전공 공지", "단과대", "학과별", "청소년학과", "컴퓨터공학과", "전공 안내")),
    ("general_notice", "general_notice", ("학교 공지", "전체 공지", "공지사항", "공지", "모집", "선발", "결과 발표", "일반 안내")),
    ("faq", "faq", ("faq", "자주 묻는 질문", "질문", "답변 모음")),
)
_RAG_DOMAIN_PRIORITY = {
    "scholarship": 8,
    "tuition": 8,
    "course_registration": 8,
    "graduation": 8,
    "document_materials": 7,
    "academic_calendar": 6,
    "career_support": 6,
    "student_life": 5,
    "department_notice": 4,
    "general_notice": 1,
    "faq": 1,
}
_RAG_DETAIL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("period", ("기간", "일정", "언제", "마감", "날짜", "기한", "시기")),
    ("eligibility", ("대상", "자격", "조건", "가능한가", "해당", "선발 기준")),
    ("procedure", ("신청 방법", "방법", "절차", "어떻게", "접수", "처리")),
    ("required_documents", ("제출서류", "제출 서류", "신청서", "양식", "증빙", "서식", "첨부파일")),
    ("benefit", ("금액", "혜택", "지원액", "감면", "지원 내용", "수혜")),
    ("announcement_lookup", ("공지", "안내", "모집", "확인", "찾아", "어디서", "발표")),
    ("summary", ("요약", "정리", "핵심", "간단히")),
)


def answer_chat(user_input: str, db: Session) -> ChatResult:
    decision = decide_chat_route(user_input)

    if decision.route == "relational_db":
        return _answer_from_relational_db(user_input, decision, db)
    if decision.route == "rag":
        return _answer_from_rag(user_input, decision)
    if decision.route == "weather":
        return _answer_from_weather(user_input)
    return ChatResult(reply=_safe_llm_reply(user_input), intent="일반", route="llm")


def decide_chat_route(user_input: str) -> ChatDecision:
    heuristic = _heuristic_decision(user_input)
    if heuristic.route != "llm":
        return heuristic

    prompt = f"""
You classify a user question for a university assistant.
Return only valid JSON:
{{"route":"llm|relational_db|rag|weather","db_intent":"map|phone|unknown","reason":"short reason","rag_category":"scholarship|tuition|course_registration|academic_calendar|graduation|document_materials|student_life|career_support|department_notice|general_notice|faq|unknown|null","rag_detail":"period|eligibility|procedure|required_documents|benefit|announcement_lookup|summary|unknown|null"}}

Routing rules:
- relational_db: exact campus place locations or phone numbers.
- weather: current or forecast weather.
- rag: crawled university documents, notices, schedules, policies, scholarship, tuition, forms, FAQ.
- llm: casual conversation or general knowledge that does not need local data.

User question:
{user_input}
"""
    try:
        raw = get_gemini_response(prompt)
    except Exception:
        logger.exception("Failed to classify chat route with LLM")
        return heuristic

    parsed = _parse_decision(raw)
    return parsed or heuristic


def infer_rag_category(user_input: str) -> str | None:
    match = _matched_rag_domain(_normalize_query(user_input))
    return match[0] if match else None


def infer_rag_detail(user_input: str) -> str | None:
    normalized = _normalize_query(user_input)
    for detail, keywords in _RAG_DETAIL_KEYWORDS:
        if _contains_any(normalized, keywords):
            return detail
    return "unknown"


def _answer_from_relational_db(user_input: str, decision: ChatDecision, db: Session) -> ChatResult:
    db_intent = decision.db_intent if decision.db_intent != "unknown" else _infer_db_intent(user_input)
    if db_intent == "phone":
        return ChatResult(
            reply=get_phone(user_input, db),
            intent="전화",
            route="relational_db",
            sources=[ChatSource(type="relational_db", title="kgu_contacts")],
        )
    if db_intent == "map":
        return ChatResult(
            reply=get_map_response(user_input, db),
            intent="지도",
            route="relational_db",
            sources=[ChatSource(type="relational_db", title="kgu_places")],
        )
    return ChatResult(
        reply="정확한 DB 정보를 찾기 위해서는 위치나 전화번호처럼 더 구체적으로 질문해 주세요.",
        intent="DB",
        route="relational_db",
        sources=[ChatSource(type="relational_db", title="relational_db")],
    )


def _answer_from_rag(user_input: str, decision: ChatDecision) -> ChatResult:
    category = decision.rag_category or infer_rag_category(user_input)
    detail = decision.rag_detail or infer_rag_detail(user_input)
    try:
        rag_result = answer_with_langchain_rag(user_input, top_k=5, category=category, detail=detail)
    except Exception:
        logger.exception("LangChain RAG chain failed; falling back to LLM")
        return ChatResult(reply=_safe_llm_reply(user_input), intent="일반", route="llm")

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
        sources=[ChatSource(type="weather_api", title=f"Open-Meteo forecast: {report.location_name}", source_url=report.source_url)],
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

    rag_match = _matched_rag_domain(normalized)
    if _contains_any(normalized, _MAP_KEYWORDS) and not rag_match:
        return ChatDecision(route="relational_db", db_intent="map", reason="map keyword")

    if rag_match:
        category, group = rag_match
        return ChatDecision(
            route="rag",
            reason=f"rag domain: {group}",
            rag_category=category,
            rag_detail=infer_rag_detail(user_input),
        )
    return ChatDecision(route="llm", reason="default")


def _infer_db_intent(user_input: str) -> DbIntent:
    normalized = _normalize_query(user_input)
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return "phone"
    if _contains_any(normalized, _MAP_KEYWORDS):
        return "map"
    return "unknown"


def _matched_rag_domain(normalized_text: str) -> tuple[str, str] | None:
    best: tuple[str, str, int, int] | None = None
    for category, group, keywords in _RAG_DOMAIN_KEYWORDS:
        count = sum(keyword.casefold() in normalized_text for keyword in keywords)
        if count <= 0:
            continue
        priority = _RAG_DOMAIN_PRIORITY.get(category, 0)
        if best is None or (priority, count) > (best[3], best[2]):
            best = (category, group, count, priority)
    if best is None:
        return None
    return best[0], best[1]


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
    rag_category = payload.get("rag_category")
    rag_detail = payload.get("rag_detail")
    if route not in {"llm", "relational_db", "rag", "weather"}:
        return None
    if db_intent not in {"map", "phone", "unknown"}:
        db_intent = "unknown"
    if rag_category in {"", "null", "unknown"}:
        rag_category = None
    if rag_detail in {"", "null"}:
        rag_detail = None
    if rag_category is not None and rag_category not in {category for category, _group, _keywords in _RAG_DOMAIN_KEYWORDS}:
        rag_category = None
    if rag_detail is not None and rag_detail not in {detail for detail, _keywords in _RAG_DETAIL_KEYWORDS} | {"unknown"}:
        rag_detail = None
    return ChatDecision(
        route=route,
        db_intent=db_intent,
        reason=str(payload.get("reason", "")),
        rag_category=rag_category,
        rag_detail=rag_detail,
    )


def _normalize_query(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input.strip().casefold())


def _contains_any(normalized_text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold() in normalized_text for keyword in keywords)
