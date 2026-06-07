from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas import SearchResponse, SearchResult
from app.services.domain_taxonomy import (
    DOMAIN_FILTERS,
    DETAIL_KEYWORDS,
    normalize_detail,
    normalize_domain,
)

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 50
MAX_KEYWORD_CANDIDATES = 30
LOW_CONFIDENCE_THRESHOLD = 0.35
PRIMARY_DETAIL_BOOST = 0.08
SECONDARY_DETAIL_BOOST = 0.04
CANONICAL_SOURCE_BOOST = 0.20
HARD_FILTER_CONFIDENCE_THRESHOLD = 0.75
PARENT_EXPANSION_WINDOW = 1
MAX_PARENT_EXPANDED_CHUNKS = 20
QUERY_ANCHOR_STOPWORDS = {
    "경기대학교",
    "경기대",
    "알려줘",
    "알려주세요",
    "궁금해",
    "확인",
    "보고",
    "싶어",
    "어떻게",
    "어디서",
    "신청",
    "절차",
    "방법",
    "기간",
    "일정",
    "공지",
    "안내",
    "기준",
    "자료",
    "찾아줘",
    "주세요",
}

CANONICAL_SOURCE_CONFIGS: dict[str, dict[str, tuple[str, ...]]] = {
    "academic_calendar": {
        "url_fragments": ("selecttnschafsschdullistus.do?key=5695",),
        "title_fragments": ("학사일정(학부)",),
    },
    "academic_status": {
        "url_fragments": (
            "contents.do?key=8412",
            "contents.do?key=8413",
            "contents.do?key=8489",
            "contents.do?key=8423",
            "contents.do?key=8706",
        ),
        "departments": ("academic_affairs",),
    },
    "course_registration": {
        "url_fragments": (
            "contents.do?key=8431",
            "contents.do?key=8430",
            "contents.do?key=8432",
            "contents.do?key=8433",
            "contents.do?key=8434",
            "contents.do?key=8435",
            "contents.do?key=8436",
            "contents.do?key=8427",
            "contents.do?key=8429",
        ),
        "departments": ("academic_affairs",),
    },
    "document_materials": {
        "url_fragments": ("contents.do?key=5729",),
        "title_fragments": ("증명서 발급",),
    },
    "graduation": {
        "url_fragments": ("contents.do?key=8418",),
        "title_fragments": ("1. 졸업안내",),
    },
    "major_change": {
        "url_fragments": (
            "contents.do?key=8414",
            "contents.do?key=8415",
            "contents.do?key=8416",
            "contents.do?key=7776",
        ),
        "departments": ("academic_affairs",),
    },
    "multi_major": {
        "url_fragments": (
            "contents.do?key=8420",
            "contents.do?key=8414",
            "contents.do?key=7776",
            "contents.do?key=9913",
            "contents.do?key=9914",
            "contents.do?key=9881",
        ),
        "departments": ("academic_affairs",),
    },
    "scholarship": {
        "url_fragments": (
            "contents.do?key=3066",
            "contents.do?key=3067",
            "contents.do?key=3068",
            "contents.do?key=3069",
            "contents.do?key=3071",
            "contents.do?key=3072",
            "contents.do?key=3073",
            "contents.do?key=3074",
            "contents.do?key=3075",
            "contents.do?key=3076",
            "contents.do?key=3084",
        ),
        "departments": ("scholarship_support",),
    },
    "teaching_certification": {
        "url_fragments": (
            "contents.do?key=8490",
            "contents.do?key=8491",
            "contents.do?key=8492",
            "contents.do?key=8493",
            "contents.do?key=8494",
        ),
        "departments": ("academic_affairs",),
    },
    "tuition": {
        "url_fragments": ("contents.do?key=3262",),
        "departments": ("finance_accounting",),
    },
}



@dataclass(frozen=True)
class RetrievalPolicy:
    """Normalized retrieval controls shared by chat, LangChain, and search trace."""

    domain: str = "unknown"
    domains: tuple[str, ...] = ()
    detail: str = "unknown"
    details: tuple[str, ...] = ()
    confidence: float | None = None
    source_scope: str = "unknown"
    rewritten_queries: tuple[str, ...] = ()
    trace_id: str | None = None
    trace_path: str | None = None

    @classmethod
    def from_inputs(
        cls,
        *,
        category: str | None = None,
        detail: str | None = None,
        rag_domain: str | None = None,
        rag_domains: list[str] | tuple[str, ...] | None = None,
        rag_detail: str | None = None,
        rag_details: list[str] | tuple[str, ...] | None = None,
        rag_confidence: float | None = None,
        source_scope: str | None = None,
        rewritten_queries: list[str] | tuple[str, ...] | None = None,
        trace_id: str | None = None,
        trace_path: str | None = None,
    ) -> "RetrievalPolicy":
        primary_domain = normalize_domain(category) or normalize_domain(rag_domain) or "unknown"
        normalized_domains = _normalize_policy_domains((primary_domain, *(rag_domains or ())))
        normalized_details = tuple(_normalize_details([detail, rag_detail, *(rag_details or ())]))
        primary_detail = normalized_details[0] if normalized_details else "unknown"
        normalized_scope = source_scope if source_scope in {"department", "university"} else "unknown"
        return cls(
            domain=primary_domain,
            domains=normalized_domains,
            detail=primary_detail,
            details=normalized_details,
            confidence=rag_confidence,
            source_scope=normalized_scope,
            rewritten_queries=tuple(dict.fromkeys(query for query in (rewritten_queries or ()) if query)),
            trace_id=trace_id,
            trace_path=trace_path,
        )

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_policy_domains(domains: tuple[str | None, ...]) -> tuple[str, ...]:
    normalized_domains: list[str] = []
    for domain in domains:
        normalized = normalize_domain(domain)
        if not normalized or normalized in normalized_domains:
            continue
        normalized_domains.append(normalized)
    return tuple(normalized_domains)


DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "scholarship": {"semantic": 0.35, "lexical": 0.25, "freshness": 0.25, "title": 0.10, "domain": 0.05},
    "tuition": {"semantic": 0.35, "lexical": 0.30, "freshness": 0.20, "title": 0.10, "domain": 0.05},
    "course_registration": {"semantic": 0.35, "lexical": 0.30, "freshness": 0.20, "title": 0.10, "domain": 0.05},
    "academic_calendar": {"semantic": 0.35, "lexical": 0.25, "freshness": 0.20, "title": 0.15, "domain": 0.05},
    "graduation": {"semantic": 0.45, "lexical": 0.30, "freshness": 0.00, "title": 0.20, "domain": 0.05},
    "document_materials": {"semantic": 0.35, "lexical": 0.30, "freshness": 0.05, "title": 0.25, "domain": 0.05},
    "student_life": {"semantic": 0.45, "lexical": 0.25, "freshness": 0.10, "title": 0.15, "domain": 0.05},
    "career_support": {"semantic": 0.40, "lexical": 0.25, "freshness": 0.20, "title": 0.10, "domain": 0.05},
    "department_notice": {"semantic": 0.40, "lexical": 0.25, "freshness": 0.20, "title": 0.10, "domain": 0.05},
    "general_notice": {"semantic": 0.40, "lexical": 0.25, "freshness": 0.20, "title": 0.10, "domain": 0.05},
    "faq": {"semantic": 0.50, "lexical": 0.30, "freshness": 0.00, "title": 0.15, "domain": 0.05},
    "default": {"semantic": 0.45, "lexical": 0.25, "freshness": 0.10, "title": 0.15, "domain": 0.05},
}


def search_documents(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    *,
    detail: Optional[str] = None,
    rag_domain: Optional[str] = None,
    rag_domains: Optional[List[str]] = None,
    rag_detail: Optional[str] = None,
    rag_details: Optional[List[str]] = None,
    rag_confidence: Optional[float] = None,
    source_scope: Optional[str] = None,
    rewritten_queries: Optional[List[str]] = None,
    enable_fallback: bool = True,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    enable_parent_expansion: bool = True,
    trace_id: str | None = None,
    trace_path: str | None = None,
    retrieval_policy: RetrievalPolicy | None = None,
) -> List[SearchResult]:
    """Search crawled chunks with domain filtering, detail boosting, and broad fallback."""
    policy = retrieval_policy or RetrievalPolicy.from_inputs(
        category=category,
        detail=detail,
        rag_domain=rag_domain,
        rag_domains=rag_domains,
        rag_detail=rag_detail,
        rag_details=rag_details,
        rag_confidence=rag_confidence,
        source_scope=source_scope,
        rewritten_queries=rewritten_queries,
        trace_id=trace_id,
        trace_path=trace_path,
    )
    effective_domain = "default" if policy.domain == "unknown" else policy.domain
    effective_details = list(policy.details)
    effective_detail = effective_details[0] if effective_details else None
    hard_filter = _should_hard_filter(policy.domain, policy.confidence)
    filter_categories = _filter_categories_for_domains(policy.domain, list(policy.domains)) if hard_filter else None
    trace: dict[str, Any] = {
        "query": query,
        "retrieval_policy": policy.to_trace_dict(),
        "effective_domain": effective_domain,
        "rag_confidence": policy.confidence,
        "hard_filter": hard_filter,
        "filter_categories": filter_categories,
        "rewritten_queries": list(policy.rewritten_queries),
        "low_confidence_threshold": low_confidence_threshold,
    }
    trace_id = trace_id or policy.trace_id
    trace_path = trace_path or policy.trace_path
    primary_rows = _run_search_attempts(
        queries=_search_queries(query, list(policy.rewritten_queries)),
        ranking_query=query,
        top_k=top_k,
        domain=effective_domain,
        detail=effective_detail,
        details=effective_details,
        categories=filter_categories,
        source_scope=policy.source_scope,
        attempt="primary",
    )
    trace["primary_rows"] = _trace_rows(primary_rows)

    if not enable_fallback or effective_domain == "default":
        trace["deduped_rows"] = _trace_rows(_dedupe_canonical_rows(primary_rows, effective_domain))
        final_rows = _finalize_rows(
            primary_rows,
            effective_domain,
            top_k,
            query=query,
            source_scope=policy.source_scope,
            low_confidence_threshold=low_confidence_threshold,
            enable_parent_expansion=enable_parent_expansion,
        )
        trace["final_rows"] = _trace_rows(final_rows)
        trace["parent_expanded"] = _trace_parent_expansion(final_rows, enabled=enable_parent_expansion)
        _write_search_trace(trace, trace_id=trace_id, trace_path=trace_path)
        return [_row_to_search_result(row) for row in final_rows]

    if primary_rows:
        trace["deduped_rows"] = _trace_rows(_dedupe_canonical_rows(primary_rows, effective_domain))
        final_rows = _finalize_rows(
            primary_rows,
            effective_domain,
            top_k,
            query=query,
            source_scope=policy.source_scope,
            low_confidence_threshold=low_confidence_threshold,
            enable_parent_expansion=enable_parent_expansion,
        )
        trace["final_rows"] = _trace_rows(final_rows)
        trace["parent_expanded"] = _trace_parent_expansion(final_rows, enabled=enable_parent_expansion)
        _write_search_trace(trace, trace_id=trace_id, trace_path=trace_path)
        return [_row_to_search_result(row) for row in final_rows]

    fallback_rows = _run_search_attempts(
        queries=_search_queries(query, list(policy.rewritten_queries)),
        ranking_query=query,
        top_k=top_k,
        domain="default",
        detail=effective_detail,
        details=effective_details,
        categories=None,
        source_scope=policy.source_scope,
        attempt="fallback_broad",
    )
    trace["fallback_rows"] = _trace_rows(fallback_rows)
    merged_rows = rerank_candidate_rows(
        rows=_merge_rows(primary_rows, _mark_fallback_rows(fallback_rows)),
        query=query,
        category=effective_domain,
        detail=effective_detail,
        details=effective_details,
        source_scope=policy.source_scope,
    )
    trace["deduped_rows"] = _trace_rows(_dedupe_canonical_rows(merged_rows, effective_domain))
    final_rows = _finalize_rows(
        merged_rows,
        effective_domain,
        top_k,
        query=query,
        source_scope=policy.source_scope,
        low_confidence_threshold=low_confidence_threshold,
        enable_parent_expansion=enable_parent_expansion,
    )
    trace["final_rows"] = _trace_rows(final_rows)
    _write_search_trace(trace, trace_id=trace_id, trace_path=trace_path)
    return [_row_to_search_result(row) for row in final_rows]


def _search_queries(query: str, rewritten_queries: list[str] | None) -> list[str]:
    return list(dict.fromkeys([query, *(rewritten_queries or [])]))


def _run_search_attempts(
    *,
    queries: list[str],
    ranking_query: str,
    top_k: int,
    domain: str | None,
    detail: str | None,
    details: list[str],
    categories: list[str] | None,
    source_scope: str | None,
    attempt: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, attempt_query in enumerate(queries):
        attempt_rows = _run_search_attempt(
            query=attempt_query,
            top_k=top_k,
            domain=domain,
            detail=detail,
            details=details,
            categories=categories,
            source_scope=source_scope,
            attempt=attempt if index == 0 else f"{attempt}_rewrite",
        )
        for row in attempt_rows:
            if index > 0:
                row["rewritten_query_used"] = attempt_query
        rows = _merge_rows(rows, attempt_rows)
    return rerank_candidate_rows(
        rows=rows,
        query=ranking_query,
        category=domain,
        detail=detail,
        details=details,
        source_scope=source_scope,
    )


def search(query: str, top_k: int = 5, category: Optional[str] = None, detail: Optional[str] = None) -> SearchResponse:
    kwargs = {"query": query, "top_k": top_k, "category": category}
    if detail is not None:
        kwargs["detail"] = detail
    return SearchResponse(query=query, results=search_documents(**kwargs))


def rerank_candidate_rows(
    *,
    rows: List[Dict[str, Any]],
    query: str,
    category: str | None,
    detail: str | None = None,
    details: list[str] | None = None,
    source_scope: str | None = None,
) -> List[Dict[str, Any]]:
    effective_domain = _normalize_domain(category) or "default"
    effective_details = _normalize_details([detail, *(details or [])])
    weights = DOMAIN_WEIGHTS.get(effective_domain, DOMAIN_WEIGHTS["default"])
    tokens = _tokenize(query)
    ranked_rows: List[Dict[str, Any]] = []

    for row in rows:
        semantic = _distance_to_score(row.get("distance"))
        lexical = max(
            float(row.get("lexical_score") or 0.0),
            _lexical_score(tokens=tokens, title=row.get("title") or "", text=row.get("text") or ""),
        )
        freshness = _freshness_score(row.get("published_at"))
        title = _title_match_score(tokens=tokens, title=row.get("title") or "")
        domain_match = _domain_match_score(effective_domain, row.get("domain") or row.get("category"))
        detail_boost = _detail_boost(effective_details, row)
        scope_boost = _scope_boost(source_scope, row)
        canonical_boost = _canonical_source_boost(effective_domain, source_scope, row)
        exact = _exact_phrase_score(query=query, title=row.get("title") or "", text=row.get("text") or "")
        source_penalty = _source_penalty(row)
        fallback_penalty = 0.35 if row.get("fallback_used") else 0.0

        base_score = (
            semantic * weights["semantic"]
            + lexical * weights["lexical"]
            + freshness * weights["freshness"]
            + title * weights["title"]
            + domain_match * weights["domain"]
        )
        score = max(
            min(
                base_score
                + detail_boost
                + scope_boost
                + canonical_boost
                + exact
                - source_penalty
                - fallback_penalty,
                1.0,
            ),
            0.0,
        )
        ranked_row = dict(row)
        ranked_row["score"] = round(score, 6)
        query_anchor_match = _row_matches_any_anchor(row, _query_anchor_terms(query))
        ranked_row["score_breakdown"] = {
            "semantic": round(semantic, 6),
            "lexical": round(lexical, 6),
            "freshness": round(freshness, 6),
            "title": round(title, 6),
            "domain": round(domain_match, 6),
            "category": round(domain_match, 6),
            "detail": round(detail_boost, 6),
            "scope": round(scope_boost, 6),
            "canonical": round(canonical_boost, 6),
            "source_scope_adjustment": round(scope_boost + canonical_boost, 6),
            "exact": round(exact, 6),
            "source_penalty": round(source_penalty, 6),
            "fallback_penalty": round(fallback_penalty, 6),
            "query_anchor_match": 1.0 if query_anchor_match else 0.0,
            "confidence": round(_confidence_score(score=score, lexical=lexical, title=title, source_penalty=source_penalty), 6),
            "fallback_used": 1.0 if row.get("fallback_used") else 0.0,
        }
        if row.get("search_attempt") == "fallback_broad":
            ranked_row["score_breakdown"]["fallback_attempt"] = 1.0
        ranked_rows.append(ranked_row)

    return sorted(
        ranked_rows,
        key=lambda row: (
            not bool(row.get("fallback_used")),
            row["score"],
            "keyword" in row.get("retrieval_sources", set()),
            "vector" in row.get("retrieval_sources", set()),
            _distance_to_score(row.get("distance")),
        ),
        reverse=True,
    )


def _finalize_rows(
    rows: list[dict[str, Any]],
    domain: str | None,
    top_k: int,
    *,
    query: str = "",
    source_scope: str | None = None,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    enable_parent_expansion: bool = True,
) -> list[dict[str, Any]]:
    deduped = _dedupe_canonical_rows(rows, domain)
    deduped = _prefer_canonical_rows(deduped, domain, source_scope)
    ranked = sorted(
        deduped,
        key=lambda row: (
            float(row.get("score") or 0.0),
            _canonical_priority(row, domain),
            "keyword" in row.get("retrieval_sources", set()),
            "vector" in row.get("retrieval_sources", set()),
        ),
        reverse=True,
    )
    marked = [_mark_low_confidence(row, low_confidence_threshold) for row in ranked]
    anchors = _query_anchor_terms(query)
    if anchors:
        aligned = [row for row in marked if _row_matches_any_anchor(row, anchors)]
        unaligned = [row for row in marked if row not in aligned]
        unaligned_have_domain = all(row.get("domain") or row.get("category") for row in unaligned)
        if aligned and (len(aligned) >= min(top_k, len(marked)) or unaligned_have_domain):
            marked = aligned
    confident = [row for row in marked if not row.get("low_confidence")]
    selected = (confident or marked)[:top_k]
    return _expand_parent_chunks(selected, top_k=top_k, enabled=enable_parent_expansion)


def _query_anchor_terms(query: str) -> list[str]:
    tokens = _tokenize(query)
    if not tokens:
        tokens = [
            token.casefold()
            for token in re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", query)
            if len(token) >= 2
        ]
    generic_stopwords = {
        "about",
        "how",
        "info",
        "please",
        "tell",
        "what",
        "when",
        "where",
    }
    return [
        token
        for token in dict.fromkeys(tokens)
        if token not in QUERY_ANCHOR_STOPWORDS and token not in generic_stopwords
    ][:6]


def _row_matches_any_anchor(row: dict[str, Any], anchors: list[str]) -> bool:
    if not anchors:
        return True
    haystack = _normalize_text(
        " ".join(
            str(row.get(key) or "")
            for key in ("title", "text", "source_url", "department", "domain", "category")
        )
    )
    for anchor in anchors:
        if anchor in haystack:
            return True
        if anchor == "교직이수" and "교직과정" in haystack and "이수" in haystack:
            return True
    return False


def _dedupe_canonical_rows(rows: list[dict[str, Any]], domain: str | None) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _canonical_dedupe_key(row, domain)
        existing = selected.get(key)
        if existing is None or _canonical_selection_key(row, domain) > _canonical_selection_key(existing, domain):
            selected[key] = row
    return list(selected.values())


def _prefer_canonical_rows(
    rows: list[dict[str, Any]],
    domain: str | None,
    source_scope: str | None,
) -> list[dict[str, Any]]:
    if source_scope == "department" or domain not in CANONICAL_SOURCE_CONFIGS:
        return rows
    canonical_rows = [row for row in rows if _is_canonical_source(domain, row)]
    return canonical_rows or rows


def _canonical_dedupe_key(row: dict[str, Any], domain: str | None) -> str:
    for field in ("content_hash", "text_hash"):
        value = str(row.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    normalized_text = _normalize_text(str(row.get("text") or ""))
    row_domain = _normalize_domain(str(row.get("domain") or row.get("category") or ""))
    if (domain == "academic_calendar" or row_domain == "academic_calendar") and normalized_text:
        return f"text:{normalized_text}"
    chunk_id = str(row.get("chunk_id") or "").strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    source_url = _normalize_source_url(str(row.get("source_url") or ""))
    title = _normalize_text(str(row.get("title") or ""))
    return f"url-title:{source_url}:{title}"


def _canonical_selection_key(row: dict[str, Any], domain: str | None) -> tuple[float, float, float]:
    return (
        _canonical_priority(row, domain),
        float(row.get("score") or 0.0),
        _distance_to_score(row.get("distance")),
    )


def _canonical_priority(row: dict[str, Any], domain: str | None) -> float:
    row_domain = _normalize_domain(str(row.get("domain") or row.get("category") or "")) or ""
    department = str(row.get("department") or "").strip().casefold()
    title = str(row.get("title") or "")
    priority = 0.0
    if row_domain == domain:
        priority += 1.0
    if domain == "academic_calendar":
        if row_domain == "academic_calendar":
            priority += 2.0
        elif row_domain == "general_notice":
            priority += 0.4
        elif row_domain == "department_notice":
            priority -= 0.2
        if department in {"university", "학교", "본교", ""}:
            priority += 0.6
        else:
            priority -= 0.3
        if "학사일정" in title:
            priority += 0.5
    if domain in CANONICAL_SOURCE_CONFIGS and _is_canonical_source(domain, row):
        priority += 1.2
    return priority


def _normalize_source_url(source_url: str) -> str:
    return re.sub(r"([?&])(utm_[^=&]+|fbclid|gclid)=[^&]+", "", source_url.strip().casefold()).rstrip("?&")


def _mark_low_confidence(row: dict[str, Any], threshold: float) -> dict[str, Any]:
    marked = dict(row)
    score = float(marked.get("score") or 0.0)
    is_low = score < threshold
    marked["low_confidence"] = is_low
    breakdown = dict(marked.get("score_breakdown") or {})
    breakdown["low_confidence"] = 1.0 if is_low else 0.0
    breakdown["low_confidence_threshold"] = threshold
    marked["score_breakdown"] = breakdown
    return marked


def _should_hard_filter(domain: str | None, rag_confidence: float | None) -> bool:
    if not domain or domain in {"default", "unknown"}:
        return False
    if domain == "department_notice":
        return False
    if rag_confidence is None:
        return True
    return rag_confidence >= HARD_FILTER_CONFIDENCE_THRESHOLD


def _expand_parent_chunks(rows: list[dict[str, Any]], *, top_k: int, enabled: bool) -> list[dict[str, Any]]:
    if not enabled or not rows:
        return rows
    anchors = [
        row
        for row in rows
        if row.get("doc_id") and row.get("chunk_index") is not None
    ]
    if not anchors:
        return rows
    expanded = _query_adjacent_chunk_rows(anchors, window=PARENT_EXPANSION_WINDOW)
    expanded = _bounded_parent_rows(rows=rows, expanded=expanded, window=PARENT_EXPANSION_WINDOW)
    if not expanded:
        return rows
    merged = _merge_rows(rows, expanded)
    expanded_chunk_ids = {expanded_row.get("chunk_id") for expanded_row in expanded}
    for row in merged:
        if row.get("chunk_id") in expanded_chunk_ids:
            row["parent_expanded"] = True
            row.setdefault("score", max(float(row.get("score") or 0.0), 0.01))
            breakdown = dict(row.get("score_breakdown") or {})
            breakdown["parent_expanded"] = 1.0
            row["score_breakdown"] = breakdown
    bounded_count = min(len(rows) + len(expanded), MAX_PARENT_EXPANDED_CHUNKS)
    return sorted(
        merged,
        key=lambda row: (
            row.get("parent_expanded") is not True,
            float(row.get("score") or 0.0),
            _distance_to_score(row.get("distance")),
        ),
        reverse=True,
    )[:bounded_count]


def _bounded_parent_rows(
    *,
    rows: list[dict[str, Any]],
    expanded: list[dict[str, Any]],
    window: int,
) -> list[dict[str, Any]]:
    if not expanded:
        return []
    anchor_positions: dict[str, set[int]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        try:
            chunk_index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        anchor_positions.setdefault(doc_id, set()).add(chunk_index)

    anchor_chunk_ids = {str(row.get("chunk_id")) for row in rows if row.get("chunk_id")}
    bounded: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    max_expanded = max(0, min(len(rows) * window * 2, MAX_PARENT_EXPANDED_CHUNKS - len(rows)))
    for row in expanded:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id or chunk_id in anchor_chunk_ids or chunk_id in seen_chunk_ids:
            continue
        doc_id = str(row.get("doc_id") or "")
        try:
            chunk_index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if not any(abs(chunk_index - anchor_index) <= window for anchor_index in anchor_positions.get(doc_id, set())):
            continue
        bounded.append(row)
        seen_chunk_ids.add(chunk_id)
        if len(bounded) >= max_expanded:
            break
    return bounded


def _query_adjacent_chunk_rows(anchors: list[dict[str, Any]], *, window: int) -> list[dict[str, Any]]:
    try:
        from sqlalchemy import and_, or_, select

        from app.db.session import SessionLocal
        from app.models import CrawlerDocument, CrawlerDocumentChunk
    except ImportError:
        logger.exception("Parent chunk expansion dependencies are unavailable")
        return []

    clauses = []
    anchor_chunk_ids = {str(row.get("chunk_id")) for row in anchors if row.get("chunk_id")}
    for row in anchors:
        doc_id = row.get("doc_id")
        try:
            chunk_index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        clauses.append(
            and_(
                CrawlerDocumentChunk.doc_id == doc_id,
                CrawlerDocumentChunk.chunk_index.between(chunk_index - window, chunk_index + window),
            )
        )
    if not clauses:
        return []

    try:
        with SessionLocal() as db:
            result_rows = db.execute(
                select(CrawlerDocumentChunk, CrawlerDocument.domain, CrawlerDocument.department, CrawlerDocument.published_at)
                .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
                .where(CrawlerDocumentChunk.status.in_(("active", "updated")))
                .where(CrawlerDocument.status.in_(("active", "updated")))
                .where(or_(*clauses))
                .order_by(CrawlerDocumentChunk.doc_id, CrawlerDocumentChunk.chunk_index)
            ).all()
    except Exception:
        logger.exception("Parent chunk expansion failed")
        return []

    rows: list[dict[str, Any]] = []
    for chunk, chunk_domain, department, published_at in result_rows:
        if chunk.chunk_id in anchor_chunk_ids:
            continue
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "title": chunk.title,
                "source_url": chunk.source_url,
                "source_type": chunk.source_type,
                "retrieval_sources": {"parent"},
                "domain": chunk_domain,
                "department": department,
                "source_name": chunk.source_name,
                "section_title": chunk.section_title,
                "section_kind": chunk.section_kind,
                "vector_point_id": chunk.vector_point_id,
                "published_at": published_at.isoformat() if published_at else None,
            }
        )
    return rows


def _trace_parent_expansion(rows: list[dict[str, Any]], *, enabled: bool) -> dict[str, Any]:
    expanded_rows = [row for row in rows if row.get("parent_expanded")]
    return {
        "enabled": enabled,
        "window": PARENT_EXPANSION_WINDOW,
        "max_context_count": MAX_PARENT_EXPANDED_CHUNKS,
        "expanded_count": len(expanded_rows),
        "chunk_ids": [row.get("chunk_id") for row in expanded_rows],
    }


def _write_search_trace(trace: dict[str, Any], *, trace_id: str | None, trace_path: str | None) -> None:
    path = trace_path or os.getenv("RAG_SEARCH_TRACE_PATH")
    if not path:
        return
    if os.getenv("RAG_SEARCH_TRACE_ENABLED", "true").casefold() in {"0", "false", "no"}:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        **trace,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "chunk_index": row.get("chunk_index"),
            "title": row.get("title"),
            "source_url": row.get("source_url"),
            "domain": row.get("domain") or row.get("category"),
            "score": row.get("score"),
            "score_breakdown": row.get("score_breakdown", {}),
            "low_confidence": row.get("low_confidence", False),
            "parent_expanded": row.get("parent_expanded", False),
            "retrieval_sources": sorted(row.get("retrieval_sources", set())),
        }
        for row in rows
    ]


def _run_search_attempt(
    *,
    query: str,
    top_k: int,
    domain: str | None,
    detail: str | None,
    details: list[str],
    categories: list[str] | None,
    source_scope: str | None,
    attempt: str,
) -> list[dict[str, Any]]:
    candidate_count = _candidate_count(top_k)
    vector_rows: list[dict[str, Any]] = []
    try:
        query_embedding = embed_text(query)
        vector_rows = _mark_vector_rows(
            _query_vector_candidates(
                query_embedding=query_embedding,
                top_k=candidate_count,
                categories=categories,
            )
        )
    except Exception as exc:
        logger.warning("Vector embedding failed; continuing with keyword search only: %s", exc)
    keyword_rows = _query_keyword_chunks(
        query=query,
        top_k=min(candidate_count, MAX_KEYWORD_CANDIDATES),
        categories=categories,
    )
    rows = _merge_rows(vector_rows, keyword_rows)
    for row in rows:
        row["search_attempt"] = attempt
    return rerank_candidate_rows(
        rows=rows,
        query=query,
        category=domain,
        detail=detail,
        details=details,
        source_scope=source_scope,
    )


def _query_vector_candidates(*, query_embedding: list[float], top_k: int, categories: list[str] | None) -> List[Dict[str, Any]]:
    if not categories:
        return query_embedded_chunks(query_embedding=query_embedding, top_k=top_k, domain=None)

    rows: List[Dict[str, Any]] = []
    per_category_top_k = max(top_k, 5)
    for category in categories:
        rows.extend(query_embedded_chunks(query_embedding=query_embedding, top_k=per_category_top_k, domain=category))
    return _dedupe_rows_by_chunk_id(rows)[:top_k]


def _query_keyword_chunks(*, query: str, top_k: int, categories: list[str] | None) -> List[Dict[str, Any]]:
    tokens = _tokenize(query)[:8]
    if not tokens or top_k <= 0:
        return []

    try:
        from sqlalchemy import or_, select

        from app.db.session import SessionLocal
        from app.models import CrawlerDocument, CrawlerDocumentChunk
    except ImportError:
        logger.exception("Keyword search dependencies are unavailable")
        return []

    try:
        with SessionLocal() as db:
            conditions = []
            for token in tokens:
                pattern = f"%{token}%"
                conditions.append(CrawlerDocumentChunk.title.ilike(pattern))
                conditions.append(CrawlerDocumentChunk.text.ilike(pattern))

            def build_stmt(category: str | None = None):
                stmt = (
                    select(CrawlerDocumentChunk, CrawlerDocument.domain, CrawlerDocument.department, CrawlerDocument.published_at)
                    .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
                    .where(CrawlerDocumentChunk.status.in_(("active", "updated")))
                    .where(CrawlerDocument.status.in_(("active", "updated")))
                    .where(or_(*conditions))
                    .order_by(CrawlerDocumentChunk.last_seen_at.desc(), CrawlerDocumentChunk.chunk_id)
                    .limit(top_k)
                )
                if category:
                    stmt = stmt.where(CrawlerDocument.domain == category)
                return stmt

            if categories:
                rows = []
                for category in categories:
                    rows.extend(db.execute(build_stmt(category)).all())
            else:
                rows = db.execute(build_stmt()).all()
    except Exception:
        logger.exception("Keyword document search failed")
        return []

    results: List[Dict[str, Any]] = []
    for chunk, chunk_domain, department, published_at in rows:
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "title": chunk.title,
                "source_url": chunk.source_url,
                "source_type": chunk.source_type,
                "lexical_score": _lexical_score(tokens=tokens, title=chunk.title or "", text=chunk.text or ""),
                "retrieval_sources": {"keyword"},
                "domain": chunk_domain,
                "department": department,
                "published_at": published_at.isoformat() if published_at else None,
            }
        )
    return results


def _row_to_search_result(row: Dict[str, Any]) -> SearchResult:
    domain = row.get("domain") or row.get("category")
    return SearchResult(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        score=row["score"],
        text=row["text"],
        title=row["title"],
        source_url=row["source_url"],
        domain=domain,
        category=domain,
        department=row.get("department"),
        source_name=row.get("source_name"),
        section_title=row.get("section_title"),
        section_kind=row.get("section_kind"),
        vector_point_id=row.get("vector_point_id"),
        published_at=row.get("published_at"),
        score_breakdown=row.get("score_breakdown", {}),
    )


def _mark_vector_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    marked = []
    for row in rows:
        marked_row = dict(row)
        marked_row["retrieval_sources"] = set(marked_row.get("retrieval_sources", set())) | {"vector"}
        marked.append(marked_row)
    return marked


def _mark_fallback_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked = []
    for row in rows:
        copy = dict(row)
        copy["fallback_used"] = True
        copy["search_attempt"] = "fallback_broad"
        marked.append(copy)
    return marked


def _merge_rows(vector_rows: List[Dict[str, Any]], keyword_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: dict[str, Dict[str, Any]] = {}
    for row in vector_rows + keyword_rows:
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        existing = merged.get(chunk_id)
        if existing is None:
            merged[chunk_id] = dict(row)
            continue

        existing_sources = set(existing.get("retrieval_sources", set()))
        row_sources = set(row.get("retrieval_sources", set()))
        existing["retrieval_sources"] = existing_sources | row_sources
        existing["lexical_score"] = max(float(existing.get("lexical_score") or 0.0), float(row.get("lexical_score") or 0.0))
        existing["fallback_used"] = bool(existing.get("fallback_used") or row.get("fallback_used"))

        existing_distance = existing.get("distance")
        row_distance = row.get("distance")
        if row_distance is not None and (existing_distance is None or row_distance < existing_distance):
            existing.update(row)
            existing["retrieval_sources"] = existing_sources | row_sources
            existing["fallback_used"] = bool(existing.get("fallback_used") or row.get("fallback_used"))
    return list(merged.values())


def _dedupe_rows_by_chunk_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        existing = merged.get(chunk_id)
        if existing is None or (
            row.get("distance") is not None
            and (existing.get("distance") is None or row["distance"] < existing["distance"])
        ):
            merged[chunk_id] = row
    return sorted(merged.values(), key=lambda row: row.get("distance") if row.get("distance") is not None else 999)


def _filter_categories(domain: str | None) -> list[str] | None:
    if not domain or domain in {"default", "unknown"}:
        return None
    return DOMAIN_FILTERS.get(domain, [domain])


def _filter_categories_for_domains(domain: str | None, extra_domains: list[str] | None) -> list[str] | None:
    categories: list[str] = []
    for raw_domain in [domain, *(extra_domains or [])]:
        normalized = _normalize_domain(raw_domain)
        if not normalized or normalized == "default":
            continue
        for category in _filter_categories(normalized) or []:
            if category not in categories:
                categories.append(category)
    return categories or None


def _normalize_domain(category: str | None) -> str | None:
    return normalize_domain(category)


def _normalize_detail(detail: str | None) -> str | None:
    return normalize_detail(detail)


def _effective_details(
    *,
    query: str,
    detail: str | None,
    rag_detail: str | None,
    rag_details: list[str] | None,
) -> list[str]:
    return _normalize_details([detail, rag_detail, *(rag_details or [])])


def _normalize_details(details: list[str | None]) -> list[str]:
    normalized_details: list[str] = []
    for detail in details:
        normalized = _normalize_detail(detail)
        if not normalized or normalized == "unknown" or normalized in normalized_details:
            continue
        normalized_details.append(normalized)
    return normalized_details


def _detail_boost(details: list[str], row: dict[str, Any]) -> float:
    if not details:
        return 0.0
    title = str(row.get("title") or "").casefold()
    text = str(row.get("text") or "").casefold()
    boost = 0.0
    for index, detail in enumerate(details):
        keywords = DETAIL_KEYWORDS.get(detail, ())
        if not keywords:
            continue
        if any(keyword.casefold() in title for keyword in keywords):
            boost += PRIMARY_DETAIL_BOOST if index == 0 else SECONDARY_DETAIL_BOOST
        elif any(keyword.casefold() in text for keyword in keywords):
            boost += (PRIMARY_DETAIL_BOOST if index == 0 else SECONDARY_DETAIL_BOOST) * 0.75
    return min(boost, 0.12)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"[0-9A-Za-z가-힣]+", text) if len(token) >= 2]


def _lexical_score(*, tokens: list[str], title: str, text: str) -> float:
    if not tokens:
        return 0.0
    normalized_title = title.casefold()
    normalized_text = text.casefold()
    matched = 0.0
    for token in tokens:
        if token in normalized_title:
            matched += 1.5
        elif token in normalized_text:
            matched += 1.0
    return min(matched / len(tokens), 1.0)


def _distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(distance, 0.0))


def _candidate_count(top_k: int) -> int:
    return min(max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, top_k), MAX_CANDIDATES)


def _freshness_score(published_at: object) -> float:
    published = _parse_datetime(published_at)
    if published is None:
        return 0.0
    age_days = max((datetime.now() - published.replace(tzinfo=None)).days, 0)
    if age_days <= 365:
        return 1.0
    if age_days <= 365 * 3:
        return 0.7
    if age_days <= 365 * 5:
        return 0.4
    return 0.1


def _title_match_score(*, tokens: list[str], title: str) -> float:
    if not tokens:
        return 0.0
    title_text = title.casefold()
    return sum(1 for token in tokens if token in title_text) / len(tokens)


def _domain_match_score(expected_domain: str | None, row_domain: object) -> float:
    if not expected_domain or expected_domain in {"default", "unknown"} or not row_domain:
        return 0.0
    row = _normalize_domain(str(row_domain))
    if row == expected_domain:
        return 1.0
    return 0.35 if str(row_domain) in (_filter_categories(expected_domain) or []) else 0.0


def _scope_boost(source_scope: str | None, row: Dict[str, Any]) -> float:
    if source_scope == "department":
        department = str(row.get("department") or "").strip()
        return 0.08 if department and department != "university" else 0.0
    if source_scope == "university":
        return 0.08 if row.get("department") == "university" else 0.0
    return 0.0


def _canonical_source_boost(domain: str | None, source_scope: str | None, row: Dict[str, Any]) -> float:
    if domain in CANONICAL_SOURCE_CONFIGS and source_scope != "department" and _is_canonical_source(domain, row):
        return CANONICAL_SOURCE_BOOST
    return 0.0


def _is_canonical_source(domain: str | None, row: Dict[str, Any]) -> bool:
    if not domain:
        return False
    config = CANONICAL_SOURCE_CONFIGS.get(domain)
    if not config:
        return False
    source_url = str(row.get("source_url") or "").casefold()
    title = str(row.get("title") or "").strip()
    department = str(row.get("department") or "").strip().casefold()
    url_fragments = config.get("url_fragments", ())
    title_fragments = config.get("title_fragments", ())
    departments = config.get("departments", ())
    if url_fragments and any(fragment.casefold() in source_url for fragment in url_fragments):
        return True
    if title_fragments and any(fragment.casefold() in title.casefold() for fragment in title_fragments):
        return True
    return bool(departments and department in departments and url_fragments and "contents.do" in source_url)


def _exact_phrase_score(*, query: str, title: str, text: str) -> float:
    normalized_query = _normalize_text(query)
    if len(normalized_query) < 4:
        return 0.0
    if normalized_query in _normalize_text(title):
        return 0.05
    return 0.02 if normalized_query in _normalize_text(text) else 0.0


def _confidence_score(*, score: float, lexical: float, title: float, source_penalty: float) -> float:
    return max(min(score * 0.75 + lexical * 0.15 + title * 0.10 - source_penalty * 0.30, 1.0), 0.0)


def _source_penalty(row: Dict[str, Any]) -> float:
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    source_url = str(row.get("source_url") or "").casefold()
    penalty = 0.0
    if title in {"제목 없음", "untitled", ""}:
        penalty += 0.05
    if "page=list" in source_url or "book_idx=" in source_url:
        penalty += 0.4
    if "비밀번호 입력" in text or "대여불가" in text:
        penalty += 0.2
    if "원문 링크에서 파일을 직접 확인하세요" in text:
        penalty += 0.03
    return penalty


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
