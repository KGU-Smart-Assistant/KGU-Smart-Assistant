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
from app.services.rag_detail_classifier import classify_rag_details_with_klue_bert
from app.services.rag_domain_classifier import classify_rag_domains_with_klue_bert
from app.services.search_service import search_documents
from app.services.weather_service import get_weather_response

ChatRoute = Literal["llm", "relational_db", "rag", "weather"]
AtomicChatRoute = ChatRoute
DbIntent = Literal["map", "phone", "unknown"]
RagAmbiguity = Literal["clear", "multi_domain", "low_confidence", "missing_detail", "needs_clarification"]


@dataclass(frozen=True)
class ChatDecision:
    route: AtomicChatRoute
    db_intent: DbIntent = "unknown"
    reason: str = ""
    query: str | None = None
    rag_domain: str | None = None
    rag_domains: tuple[str, ...] = ()
    rag_detail: str | None = None
    rag_details: tuple[str, ...] = ()
    source_scope: str | None = None
    rag_confidence: float | None = None
    rag_ambiguity: RagAmbiguity | None = None
    rewritten_queries: tuple[str, ...] = ()
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
    rag_details: tuple[str, ...] = ()
    source_scope: str | None = None
    rag_confidence: float | None = None
    rag_ambiguity: RagAmbiguity | None = None
    rewritten_queries: tuple[str, ...] = ()
    matched_keywords: tuple[str, ...] = ()
    intent_scores: tuple["RagIntentScore", ...] = ()
    suggested_domains: tuple[str, ...] = ()
    suggested_details: tuple[str, ...] = ()
    answer_status: Literal["answered", "partial", "insufficient"] = "answered"
    unverified: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagIntentScore:
    domain: str
    score: float
    matched_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagDetailScore:
    detail: str
    score: float


@dataclass(frozen=True)
class RagClassification:
    domain: str
    domains: tuple[str, ...] = ()
    detail: str = "unknown"
    details: tuple[str, ...] = ()
    detail_score: float = 0.0
    source_scope: str = "unknown"
    confidence: float = 0.0
    ambiguity: RagAmbiguity = "clear"
    rewritten_queries: tuple[str, ...] = ()
    matched_keywords: tuple[str, ...] = ()
    intent_scores: tuple[RagIntentScore, ...] = ()


MIN_GROUNDED_RESULT_SCORE = 0.18



def answer_chat(user_input: str, db: Session) -> ChatResult:
    plan = decide_chat_plan(user_input)
    if len(plan.actions) > 1:
        return _answer_from_compound(user_input, plan.actions, db)

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
    bert_decision = _klue_bert_decision(user_input)
    if bert_decision is not None:
        return _attach_rag_classification_to_plan(
            user_input,
            ChatPlan(actions=(bert_decision,), reason=bert_decision.reason),
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
- Use relational_db for campus location/path/phone/contact requests.
- For relational_db, set db_intent to map for location/path requests and phone for phone/contact requests.

User question:
{user_input}
"""
    raw = get_gemini_response(prompt)
    parsed = _parse_decision_plan(raw)
    if parsed is None:
        return ChatPlan(actions=(ChatDecision(route="llm", reason="classifier fallback"),), reason="classifier fallback")
    return _attach_rag_classification_to_plan(user_input, parsed)


def _klue_bert_decision(user_input: str) -> ChatDecision | None:
    prediction = classify_with_klue_bert(user_input)
    if prediction is None:
        return None

    if prediction.confidence < settings.intent_classifier_confidence_threshold:
        return None

    return ChatDecision(
        route=prediction.route,
        db_intent=prediction.db_intent if prediction.route == "relational_db" else "unknown",
        reason=f"klue-bert:{prediction.label}:{prediction.confidence:.3f}",
    )


def _answer_from_relational_db(
    user_input: str,
    decision: ChatDecision,
    db: Session,
) -> ChatResult:
    db_intent = decision.db_intent

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
    if _should_request_rag_clarification(decision):
        return _rag_clarification_result(decision)

    try:
        search_parameters = inspect.signature(search_documents).parameters
        search_kwargs = {"query": user_input, "top_k": 5}
        if "rag_domain" in search_parameters:
            search_kwargs["rag_domain"] = decision.rag_domain
        if "rag_domains" in search_parameters:
            search_kwargs["rag_domains"] = list(decision.rag_domains)
        if "rag_detail" in search_parameters:
            search_kwargs["rag_detail"] = decision.rag_detail
        if "rag_details" in search_parameters:
            search_kwargs["rag_details"] = list(decision.rag_details)
        if "source_scope" in search_parameters:
            search_kwargs["source_scope"] = decision.source_scope
        if "rewritten_queries" in search_parameters:
            search_kwargs["rewritten_queries"] = list(decision.rewritten_queries)
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
            rag_details=decision.rag_details,
            rag_domains=decision.rag_domains,
            source_scope=decision.source_scope,
            rag_confidence=decision.rag_confidence,
            rag_ambiguity=decision.rag_ambiguity,
            rewritten_queries=decision.rewritten_queries,
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
            rag_details=decision.rag_details,
            source_scope=decision.source_scope,
            rag_confidence=decision.rag_confidence,
            rag_ambiguity=decision.rag_ambiguity,
            rewritten_queries=decision.rewritten_queries,
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
            rag_details=decision.rag_details,
            source_scope=decision.source_scope,
            rag_confidence=decision.rag_confidence,
            rag_ambiguity=decision.rag_ambiguity,
            rewritten_queries=decision.rewritten_queries,
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
        rag_details=decision.rag_details,
        source_scope=decision.source_scope,
        rag_confidence=decision.rag_confidence,
        rag_ambiguity=decision.rag_ambiguity,
        rewritten_queries=decision.rewritten_queries,
        matched_keywords=decision.matched_keywords,
        intent_scores=decision.intent_scores,
        answer_status=answer_status,
        unverified=() if answer_status == "answered" else (_unverified_reason(decision),),
    )


def _should_request_rag_clarification(decision: ChatDecision) -> bool:
    if decision.rag_ambiguity == "needs_clarification":
        return True
    if decision.rag_ambiguity == "low_confidence":
        return settings.rag_clarify_on_low_confidence
    if decision.rag_ambiguity == "multi_domain":
        return settings.rag_clarify_on_multi_domain
    if decision.rag_ambiguity == "missing_detail":
        return settings.rag_clarify_on_missing_detail
    return False


def _rag_clarification_result(decision: ChatDecision) -> ChatResult:
    return ChatResult(
        reply=_rag_clarification_reply(decision),
        intent="RAG",
        route="rag",
        rag_domain=decision.rag_domain,
        rag_domains=decision.rag_domains,
        rag_detail=decision.rag_detail,
        rag_details=decision.rag_details,
        source_scope=decision.source_scope,
        rag_confidence=decision.rag_confidence,
        rag_ambiguity=decision.rag_ambiguity,
        rewritten_queries=decision.rewritten_queries,
        matched_keywords=decision.matched_keywords,
        intent_scores=decision.intent_scores,
        suggested_domains=_suggested_rag_domains(decision),
        suggested_details=_suggested_rag_details(decision),
        answer_status="insufficient",
        unverified=(_unverified_reason(decision),),
    )


def _rag_clarification_reply(decision: ChatDecision) -> str:
    domains = _suggested_rag_domains(decision)
    if decision.rag_ambiguity == "multi_domain" and domains:
        return (
            "질문이 여러 업무 범위에 걸쳐 있어 바로 답변하기 어렵습니다. "
            f"{', '.join(domains)} 중 어느 내용인지 조금 더 구체적으로 알려주세요."
        )
    if decision.rag_ambiguity == "missing_detail":
        return (
            "질문에서 필요한 정보 종류가 분명하지 않습니다. "
            "기간, 자격, 제출서류, 신청방법, 금액처럼 원하는 항목을 함께 알려주세요."
        )
    return (
        "질문 의도를 충분히 확신하지 못했습니다. "
        "찾고 싶은 업무나 공지 범위를 조금 더 구체적으로 알려주세요."
    )


def _suggested_rag_domains(decision: ChatDecision) -> tuple[str, ...]:
    return tuple(domain for domain in decision.rag_domains[:3] if domain != "unknown")


def _suggested_rag_details(decision: ChatDecision) -> tuple[str, ...]:
    if decision.rag_details:
        return decision.rag_details[:3]
    if decision.rag_ambiguity == "missing_detail":
        return ("period", "eligibility", "required_documents", "procedure", "benefit")
    return ()


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


def _answer_from_compound(
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
        route=results[0].route if results else "llm",
        sources=sources,
    )



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


def _normalize_query(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input.strip().lower())


def _attach_rag_classification_to_plan(user_input: str, plan: ChatPlan) -> ChatPlan:
    return ChatPlan(
        actions=tuple(
            _attach_rag_classification(
                action.query or user_input,
                action,
            )
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
            rag_details=(),
            source_scope="unknown",
            rag_confidence=0.0,
            rag_ambiguity="needs_clarification",
            rewritten_queries=(_normalize_query(user_input),),
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
        rag_details=rag_classification.details,
        source_scope=rag_classification.source_scope,
        rag_confidence=rag_classification.confidence,
        rag_ambiguity=rag_classification.ambiguity,
        rewritten_queries=rag_classification.rewritten_queries,
        matched_keywords=rag_classification.matched_keywords,
        intent_scores=rag_classification.intent_scores,
    )


def _classify_rag_query(normalized_text: str) -> RagClassification | None:
    # RAG intent pipeline:
    # 1. routing is decided before this function
    # 2. rag_domain is predicted by the dedicated multi-label KLUE-BERT model
    # 3. rag_detail is classified independently
    # 4. confidence is calculated from model/detail signals
    # 5. ambiguity is derived from confidence and top-k domain shape
    # 6. rewritten_queries are prepared for downstream retrieval
    source_scope = "unknown"
    intent_scores = _model_rag_intent_scores(normalized_text)
    selected_domain = intent_scores[0].domain if intent_scores else "unknown"

    detail_predictions = _classify_rag_detail_predictions(normalized_text)
    details = tuple(prediction.detail for prediction in detail_predictions)
    detail = details[0] if details else "unknown"
    detail_score = detail_predictions[0].score if detail_predictions else 0.0
    confidence = _rag_confidence(
        intent_scores=intent_scores,
        detail=detail,
        detail_score=detail_score,
        source_scope=source_scope,
    )
    ambiguity = _rag_ambiguity(
        intent_scores=intent_scores,
        detail=detail,
        confidence=confidence,
    )
    rewritten_queries = _rewrite_rag_queries(
        normalized_text,
        domains=tuple(score.domain for score in intent_scores),
        detail=detail,
        source_scope=source_scope,
    )
    matched_keywords = ()

    return RagClassification(
        domain=selected_domain,
        domains=tuple(score.domain for score in intent_scores),
        detail=detail,
        details=details,
        detail_score=detail_score,
        source_scope=source_scope,
        confidence=confidence,
        ambiguity=ambiguity,
        rewritten_queries=rewritten_queries,
        matched_keywords=matched_keywords,
        intent_scores=tuple(intent_scores),
    )


def _model_rag_intent_scores(normalized_text: str) -> list[RagIntentScore]:
    return [
        RagIntentScore(
            domain=prediction.domain,
            score=prediction.score,
            matched_keywords=(),
        )
        for prediction in classify_rag_domains_with_klue_bert(normalized_text)
    ]


def _classify_rag_details(
    normalized_text: str,
    *,
    selected_domain: str,
) -> tuple[str, ...]:
    return tuple(
        prediction.detail
        for prediction in _classify_rag_detail_predictions(
            normalized_text,
        )
    )


def _classify_rag_detail_predictions(normalized_text: str) -> tuple[RagDetailScore, ...]:
    details: list[RagDetailScore] = []
    for prediction in classify_rag_details_with_klue_bert(normalized_text):
        if prediction.detail == "unknown" or any(detail.detail == prediction.detail for detail in details):
            continue
        details.append(RagDetailScore(detail=prediction.detail, score=prediction.score))
    return tuple(details)


def _rag_confidence(
    *,
    intent_scores: list[RagIntentScore],
    detail: str,
    detail_score: float,
    source_scope: str,
) -> float:
    if not intent_scores:
        confidence = 0.18
    else:
        top_score = intent_scores[0].score
        second_score = intent_scores[1].score if len(intent_scores) > 1 else 0.0
        domain_gap = max(top_score - second_score, 0.0)
        confidence = 0.25 + (top_score * 0.5) + min(domain_gap, 0.25)
    if detail != "unknown":
        confidence += 0.04 + min(max(detail_score, 0.0), 1.0) * 0.1
    if source_scope != "unknown":
        confidence += 0.04
    return round(min(max(confidence, 0.0), 0.95), 3)


def _rag_ambiguity(
    *,
    intent_scores: list[RagIntentScore],
    detail: str,
    confidence: float,
) -> RagAmbiguity:
    if not intent_scores:
        return "needs_clarification"
    if confidence < 0.45:
        return "low_confidence"
    if len(intent_scores) > 1 and intent_scores[0].score - intent_scores[1].score < 0.12:
        return "multi_domain"
    if detail == "unknown":
        return "missing_detail"
    return "clear"


def _rewrite_rag_queries(
    normalized_text: str,
    *,
    domains: tuple[str, ...],
    detail: str,
    source_scope: str,
) -> tuple[str, ...]:
    query_parts = [normalized_text]
    if domains:
        query_parts.append(" ".join(domains[:3]))
    if detail != "unknown":
        query_parts.append(detail)
    if source_scope != "unknown":
        query_parts.append(source_scope)

    expanded = tuple(
        dict.fromkeys(
            query
            for query in (
                normalized_text,
                " ".join(part for part in query_parts if part),
            )
            if query
        )
    )
    return expanded or (normalized_text,)


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
