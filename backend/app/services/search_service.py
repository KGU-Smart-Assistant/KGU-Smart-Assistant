from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas import SearchResponse, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 50
MAX_KEYWORD_CANDIDATES = 30

CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "notice": {"semantic": 0.45, "lexical": 0.20, "freshness": 0.20, "title": 0.10, "category": 0.05},
    "scholarship": {"semantic": 0.40, "lexical": 0.20, "freshness": 0.25, "title": 0.10, "category": 0.05},
    "materials": {"semantic": 0.45, "lexical": 0.25, "freshness": 0.05, "title": 0.20, "category": 0.05},
    "faq": {"semantic": 0.55, "lexical": 0.25, "freshness": 0.00, "title": 0.15, "category": 0.05},
    "academic_schedule": {"semantic": 0.40, "lexical": 0.20, "freshness": 0.20, "title": 0.15, "category": 0.05},
    "graduation": {"semantic": 0.50, "lexical": 0.25, "freshness": 0.00, "title": 0.20, "category": 0.05},
    "career": {"semantic": 0.45, "lexical": 0.20, "freshness": 0.20, "title": 0.10, "category": 0.05},
    "student_life": {"semantic": 0.50, "lexical": 0.20, "freshness": 0.10, "title": 0.15, "category": 0.05},
    "default": {"semantic": 0.55, "lexical": 0.20, "freshness": 0.10, "title": 0.10, "category": 0.05},
}

CATEGORY_ALIASES = {
    "academic": "academic_schedule",
    "support": "scholarship",
    "graduation_requirements": "graduation",
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "academic_schedule": ("학사일정", "수강신청", "개강", "종강", "시험", "학기"),
    "graduation": ("졸업", "졸업요건", "졸업학점", "학위", "전공학점", "교양학점"),
    "scholarship": ("장학", "장학금", "등록금", "학자금", "지원금", "국가장학금"),
    "materials": ("자료", "자료실", "첨부", "파일", "서식", "양식", "pdf", "hwp"),
    "faq": ("faq", "질문", "답변", "자주 묻는"),
    "notice": ("공지", "공지사항", "모집", "신청", "안내"),
    "career": ("취업", "진로", "현장실습", "인턴", "채용", "비교과"),
    "student_life": ("학생생활", "학생증", "동아리", "기숙사", "식당", "복지"),
}


def search_documents(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[SearchResult]:
    """Return the most relevant document chunks for a query."""
    normalized_category = _normalize_category(category)
    effective_category = normalized_category or _infer_category(query)
    query_embedding = embed_text(query)
    candidate_count = _candidate_count(top_k)
    vector_rows = _mark_vector_rows(
        query_embedded_chunks(
            query_embedding=query_embedding,
            top_k=candidate_count,
            category=normalized_category,
        )
    )
    keyword_rows = _query_keyword_chunks(
        query=query,
        top_k=min(candidate_count, MAX_KEYWORD_CANDIDATES),
        category=normalized_category,
    )
    ranked_rows = rerank_candidate_rows(
        rows=_merge_rows(vector_rows, keyword_rows),
        query=query,
        category=effective_category,
    )[:top_k]
    return [_row_to_search_result(row) for row in ranked_rows]


def search(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> SearchResponse:
    """Wrap raw search results in the standard response schema."""
    results = search_documents(query=query, top_k=top_k, category=category)
    return SearchResponse(query=query, results=results)


def rerank_candidate_rows(
    *,
    rows: List[Dict[str, Any]],
    query: str,
    category: str | None,
) -> List[Dict[str, Any]]:
    effective_category = _normalize_category(category) or "default"
    weights = CATEGORY_WEIGHTS.get(effective_category, CATEGORY_WEIGHTS["default"])
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
        category_match = _category_match_score(effective_category, row.get("category"))
        exact = _exact_phrase_score(query=query, title=row.get("title") or "", text=row.get("text") or "")

        base_score = (
            semantic * weights["semantic"]
            + lexical * weights["lexical"]
            + freshness * weights["freshness"]
            + title * weights["title"]
            + category_match * weights["category"]
        )
        score = min(base_score + exact, 1.0)

        ranked_row = dict(row)
        ranked_row["score"] = round(score, 6)
        ranked_row["score_breakdown"] = {
            "semantic": round(semantic, 6),
            "lexical": round(lexical, 6),
            "freshness": round(freshness, 6),
            "title": round(title, 6),
            "category": round(category_match, 6),
            "exact": round(exact, 6),
        }
        ranked_rows.append(ranked_row)

    return sorted(
        ranked_rows,
        key=lambda row: (
            row["score"],
            "keyword" in row.get("retrieval_sources", set()),
            "vector" in row.get("retrieval_sources", set()),
            _distance_to_score(row.get("distance")),
        ),
        reverse=True,
    )


def _row_to_search_result(row: Dict[str, Any]) -> SearchResult:
    return SearchResult(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        score=row["score"],
        text=row["text"],
        title=row["title"],
        source_url=row["source_url"],
        category=row.get("category"),
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


def _distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(distance, 0.0))


def _query_keyword_chunks(
    *,
    query: str,
    top_k: int,
    category: str | None,
) -> List[Dict[str, Any]]:
    tokens = _tokenize(query)[:6]
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

            stmt = (
                select(
                    CrawlerDocumentChunk,
                    CrawlerDocument.category,
                    CrawlerDocument.department,
                    CrawlerDocument.published_at,
                )
                .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
                .where(CrawlerDocumentChunk.status == "active")
                .where(CrawlerDocument.status.in_(("active", "updated")))
                .where(or_(*conditions))
                .order_by(CrawlerDocumentChunk.last_seen_at.desc(), CrawlerDocumentChunk.chunk_id)
                .limit(top_k)
            )
            if category:
                stmt = stmt.where(CrawlerDocument.category == category)

            rows = db.execute(stmt).all()
    except Exception:
        logger.exception("Keyword document search failed")
        return []

    results: List[Dict[str, Any]] = []
    for chunk, chunk_category, department, published_at in rows:
        lexical_score = _lexical_score(
            tokens=tokens,
            title=chunk.title or "",
            text=chunk.text or "",
        )
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "title": chunk.title,
                "source_url": chunk.source_url,
                "source_type": chunk.source_type,
                "lexical_score": lexical_score,
                "retrieval_sources": {"keyword"},
                "category": chunk_category,
                "department": department,
                "published_at": published_at.isoformat() if published_at else None,
            }
        )
    return results


def _merge_rows(
    vector_rows: List[Dict[str, Any]],
    keyword_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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
        existing["lexical_score"] = max(
            float(existing.get("lexical_score") or 0.0),
            float(row.get("lexical_score") or 0.0),
        )

        existing_distance = existing.get("distance")
        row_distance = row.get("distance")
        if row_distance is not None and (
            existing_distance is None or row_distance < existing_distance
        ):
            existing.update(row)
            existing["retrieval_sources"] = existing_sources | row_sources
    return list(merged.values())


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
    matched = sum(1 for token in tokens if token in title_text)
    return matched / len(tokens)


def _category_match_score(expected_category: str | None, row_category: object) -> float:
    if not expected_category or expected_category == "default" or not row_category:
        return 0.0
    return 1.0 if _normalize_category(str(row_category)) == expected_category else 0.0


def _exact_phrase_score(*, query: str, title: str, text: str) -> float:
    normalized_query = _normalize_text(query)
    if len(normalized_query) < 4:
        return 0.0
    normalized_title = _normalize_text(title)
    if normalized_query and normalized_query in normalized_title:
        return 0.05
    normalized_text = _normalize_text(text)
    return 0.02 if normalized_query and normalized_query in normalized_text else 0.0


def _infer_category(query: str) -> str:
    normalized = query.casefold()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            return category
    return "default"


def _normalize_category(category: str | None) -> str | None:
    if not category:
        return None
    normalized = category.casefold()
    return CATEGORY_ALIASES.get(normalized, normalized)


def _tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣]+", text)
        if len(token) >= 2
    ]


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
