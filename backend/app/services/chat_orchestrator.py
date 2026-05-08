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
_LOCATION_REQUEST_KEYWORDS = (
    "위치",
    "찾아가",
    "가는 길",
    "어떻게 가",
    "길찾기",
    "캠퍼스맵",
    "어디야",
    "어디 있",
    "호실",
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

# Sources in app/crawlers/sources.yaml are grouped into these user-question domains.
_RAG_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "academic_schedule": (
        "학사일정",
        "학사 일정",
        "수강신청",
        "수강 정정",
        "수강취소",
        "등록 기간",
        "휴학",
        "복학",
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
        "성적향상장학금",
        "학자금",
        "등록금",
        "수혜",
        "중복 수혜",
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
    "department_sources": (
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
}
_RAG_KEYWORDS = tuple(
    keyword
    for keywords in _RAG_KEYWORD_GROUPS.values()
    for keyword in keywords
)
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
    ("scholarship_support", ("장학", "장학금", "국가장학금", "성적향상장학금")),
    ("graduation_requirements", ("졸업", "졸업요건", "졸업학점")),
    ("materials", ("자료실", "첨부파일", "첨부 파일", "양식", "서식", "신청서")),
    ("academic_schedule", ("학사일정", "수강신청", "휴학", "복학", "개강", "종강")),
    ("career_support", ("취업", "진로", "현장실습", "인턴", "채용")),
    ("student_life", ("학생생활", "학생증", "동아리", "기숙사", "셔틀")),
)


def answer_chat(user_input: str, db: Session) -> ChatResult:
    plan = decide_chat_plan(user_input)
    if len(plan.actions) > 1:
        return _answer_from_multi(user_input, plan.actions, db)

    decision = plan.actions[0]
    return _answer_for_decision(user_input, decision, db)


def _answer_for_decision(user_input: str, decision: ChatDecision, db: Session) -> ChatResult:

    if decision.route == "relational_db":
        return _answer_from_relational_db(user_input, decision, db)

    if decision.route == "rag":
        return _answer_from_rag(user_input)

    if decision.route == "weather":
        return _answer_from_weather(user_input)

    return ChatResult(
        reply=get_gemini_response(user_input),
        intent="일반",
        route="llm",
    )


def decide_chat_route(user_input: str) -> ChatDecision:
    return decide_chat_plan(user_input).actions[0]


def decide_chat_plan(user_input: str) -> ChatPlan:
    compound_decisions = _compound_decisions(user_input)
    if len(compound_decisions) > 1:
        return ChatPlan(
            actions=tuple(compound_decisions),
            reason="compound keyword match",
        )

    bert_decision = _klue_bert_decision(user_input)
    if bert_decision is not None:
        return ChatPlan(actions=(bert_decision,), reason=bert_decision.reason)

    heuristic = _heuristic_decision(user_input)
    if not settings.intent_classifier_model_name and heuristic.route != "llm":
        return ChatPlan(actions=(heuristic,), reason=heuristic.reason)

    prompt = f"""
You classify a user question for a university assistant.
Return only valid JSON with this schema:
{{"actions":[{{"route":"llm|relational_db|rag|weather","db_intent":"map|phone|unknown"}}],"reason":"short reason"}}

Routing rules:
- llm: basic general knowledge or casual conversation that does not need local data.
- relational_db: exact campus data stored in relational DB, such as place locations or phone numbers.
- rag: information that must be grounded in crawled documents, notices, policies, schedules, or other text sources.
- weather: current or forecast weather questions that need live weather API data.
- If the user asks for multiple independent things, return multiple actions in the order they should be answered.
- Use relational_db with db_intent map for campus location/path requests.
- Use relational_db with db_intent phone for phone number/contact requests.

User question:
{user_input}
"""
    raw = get_gemini_response(prompt)
    parsed = _parse_decision_plan(raw)
    if parsed is None:
        return ChatPlan(actions=(heuristic,), reason=heuristic.reason)
    return parsed


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


def _answer_from_rag(user_input: str) -> ChatResult:
    try:
        results = search_documents(query=user_input, top_k=5)
    except NotImplementedError:
        return ChatResult(
            reply=(
                "이 질문은 문서 기반 검색(RAG)으로 답해야 하지만, 현재 검색 파이프라인이 "
                "연결되어 있지 않습니다."
            ),
            intent="RAG",
            route="rag",
        )

    if not results:
        return ChatResult(
            reply="관련 문서를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요.",
            intent="RAG",
            route="rag",
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
    return ChatResult(reply=reply, intent="RAG", route="rag", sources=sources)


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

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        return ChatDecision(route="weather", reason="weather keyword")
    if _contains_any(normalized, _PHONE_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="phone", reason="phone keyword")
    if _contains_any(normalized, _MAP_KEYWORDS):
        return ChatDecision(route="relational_db", db_intent="map", reason="map keyword")

    rag_reason = _matched_rag_group(normalized)
    if rag_reason:
        return ChatDecision(route="rag", reason=rag_reason)
    return ChatDecision(route="llm", reason="default")


def _compound_decisions(user_input: str) -> list[ChatDecision]:
    normalized = _normalize_query(user_input)
    decisions: list[ChatDecision] = []

    if _contains_any(normalized, _LOCATION_REQUEST_KEYWORDS):
        decisions.append(
            ChatDecision(route="relational_db", db_intent="map", reason="compound map keyword")
        )
    if _contains_any(normalized, _PHONE_KEYWORDS):
        decisions.append(
            ChatDecision(route="relational_db", db_intent="phone", reason="compound phone keyword")
        )

    rag_reason = _matched_rag_group(normalized)
    if rag_reason:
        decisions.append(ChatDecision(route="rag", reason=rag_reason))

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        decisions.append(ChatDecision(route="weather", reason="weather keyword"))

    return _dedupe_decisions(decisions)


def _dedupe_decisions(decisions: list[ChatDecision]) -> list[ChatDecision]:
    deduped: list[ChatDecision] = []
    seen: set[tuple[str, str]] = set()
    for decision in decisions:
        key = (decision.route, decision.db_intent)
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


def _contains_any(normalized_text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in normalized_text for keyword in keywords)


def _matched_rag_group(normalized_text: str) -> str | None:
    for group, keywords in _RAG_FORCE_GROUP_KEYWORDS:
        if _contains_any(normalized_text, keywords):
            return f"rag keyword: {group}"

    matches = [
        (
            group,
            sum(keyword.lower() in normalized_text for keyword in keywords),
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

    if route not in {"llm", "relational_db", "rag", "weather"}:
        return None
    if db_intent not in {"map", "phone", "unknown"}:
        db_intent = "unknown"
    if route != "relational_db":
        db_intent = "unknown"

    return ChatDecision(route=route, db_intent=db_intent, reason=reason)  # type: ignore[arg-type]


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
