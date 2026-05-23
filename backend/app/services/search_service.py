from __future__ import annotations

from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas import SearchResponse, SearchResult
from app.services.domain_taxonomy import (
    DOMAIN_FILTERS,
    normalize_detail,
    normalize_domain,
)

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 50
MAX_KEYWORD_CANDIDATES = 30
LOW_CONFIDENCE_THRESHOLD = 0.35

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
    source_scope: Optional[str] = None,
    rewritten_queries: Optional[List[str]] = None,
    enable_fallback: bool = True,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> List[SearchResult]:
    """Search crawled chunks with domain filtering, detail boosting, and broad fallback."""
    effective_domain = _normalize_domain(category) or _normalize_domain(rag_domain) or "default"
    effective_details = _effective_details(query=query, detail=detail, rag_detail=rag_detail, rag_details=rag_details)
    effective_detail = effective_details[0] if effective_details else None
    filter_categories = _filter_categories_for_domains(effective_domain, rag_domains)
    primary_rows = _run_search_attempts(
        queries=_search_queries(query, rewritten_queries),
        ranking_query=query,
        top_k=top_k,
        domain=effective_domain,
        detail=effective_detail,
        details=effective_details,
        categories=filter_categories,
        source_scope=source_scope,
        attempt="primary",
    )

    if not enable_fallback or effective_domain == "default":
        return [_row_to_search_result(row) for row in primary_rows[:top_k]]

    if primary_rows:
        return [_row_to_search_result(row) for row in primary_rows[:top_k]]

    fallback_rows = _run_search_attempts(
        queries=_search_queries(query, rewritten_queries),
        ranking_query=query,
        top_k=top_k,
        domain="default",
        detail=effective_detail,
        details=effective_details,
        categories=None,
        source_scope=source_scope,
        attempt="fallback_broad",
    )
    merged_rows = rerank_candidate_rows(
        rows=_merge_rows(primary_rows, _mark_fallback_rows(fallback_rows)),
        query=query,
        category=effective_domain,
        detail=effective_detail,
        details=effective_details,
        source_scope=source_scope,
    )
    return [_row_to_search_result(row) for row in merged_rows[:top_k]]


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
    effective_detail = effective_details[0] if effective_details else None
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
        detail_boost = 0.0
        scope_boost = _scope_boost(source_scope, row)
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
        score = max(min(base_score + detail_boost + scope_boost + exact - source_penalty - fallback_penalty, 1.0), 0.0)
        ranked_row = dict(row)
        ranked_row["score"] = round(score, 6)
        ranked_row["score_breakdown"] = {
            "semantic": round(semantic, 6),
            "lexical": round(lexical, 6),
            "freshness": round(freshness, 6),
            "title": round(title, 6),
            "domain": round(domain_match, 6),
            "category": round(domain_match, 6),
            "detail": round(detail_boost, 6),
            "scope": round(scope_boost, 6),
            "exact": round(exact, 6),
            "source_penalty": round(source_penalty, 6),
            "fallback_penalty": round(fallback_penalty, 6),
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
    query_embedding = embed_text(query)
    candidate_count = _candidate_count(top_k)
    vector_rows = _mark_vector_rows(
        _query_vector_candidates(
            query_embedding=query_embedding,
            top_k=candidate_count,
            categories=categories,
        )
    )
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
    if not domain or domain == "default":
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
    if not expected_domain or expected_domain == "default" or not row_domain:
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
