from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas import SearchResponse, SearchResult

DEFAULT_CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 50

CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "notice": {"similarity": 0.55, "freshness": 0.30, "title": 0.15},
    "scholarship": {"similarity": 0.50, "freshness": 0.35, "title": 0.15},
    "materials": {"similarity": 0.60, "freshness": 0.15, "title": 0.25},
    "faq": {"similarity": 0.80, "freshness": 0.00, "title": 0.20},
    "academic_schedule": {"similarity": 0.50, "freshness": 0.25, "title": 0.25},
    "graduation": {"similarity": 0.70, "freshness": 0.05, "title": 0.25},
    "default": {"similarity": 0.75, "freshness": 0.10, "title": 0.15},
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "academic_schedule": ("학사일정", "수강신청", "개강", "종강", "시험", "학기"),
    "graduation": ("졸업", "졸업요건", "졸업학점", "학위", "전공학점"),
    "scholarship": ("장학", "장학금", "등록금", "학자금"),
    "materials": ("자료", "자료실", "첨부", "파일", "서식", "양식", "pdf", "hwp"),
    "faq": ("faq", "질문", "답변", "자주 묻는"),
    "notice": ("공지", "공지사항", "모집", "신청", "안내"),
}


def search_documents(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[SearchResult]:
    """Return the most relevant document chunks for a query.

    This service interface is fixed first so the crawler, chunker,
    embedder, and vector store can all target the same contract.
    """
    query_embedding = embed_text(query)
    candidate_count = _candidate_count(top_k)
    rows = query_embedded_chunks(
        query_embedding=query_embedding,
        top_k=candidate_count,
        category=category,
    )
    ranked_rows = _rerank_rows(
        rows=rows,
        query=query,
        category=category or _infer_category(query),
    )[:top_k]
    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            score=row["score"],
            text=row["text"],
            title=row["title"],
            source_url=row["source_url"],
        )
        for row in ranked_rows
    ]


def search(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> SearchResponse:
    """Wrap raw search results in the standard response schema."""
    results = search_documents(query=query, top_k=top_k, category=category)
    return SearchResponse(query=query, results=results)


def _distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(distance, 0.0))


def _candidate_count(top_k: int) -> int:
    return min(max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, top_k), MAX_CANDIDATES)


def _rerank_rows(
    *,
    rows: List[Dict[str, Any]],
    query: str,
    category: str | None,
) -> List[Dict[str, Any]]:
    effective_category = category or "default"
    weights = CATEGORY_WEIGHTS.get(effective_category, CATEGORY_WEIGHTS["default"])
    ranked_rows: List[Dict[str, Any]] = []
    for row in rows:
        similarity = _distance_to_score(row.get("distance"))
        freshness = _freshness_score(row.get("published_at"))
        title = _title_match_score(query=query, title=row.get("title") or "")
        score = (
            similarity * weights["similarity"]
            + freshness * weights["freshness"]
            + title * weights["title"]
        )
        ranked_row = dict(row)
        ranked_row["score"] = round(score, 6)
        ranked_rows.append(ranked_row)

    return sorted(
        ranked_rows,
        key=lambda row: (
            row["score"],
            _distance_to_score(row.get("distance")),
        ),
        reverse=True,
    )


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


def _title_match_score(*, query: str, title: str) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    title_text = title.casefold()
    matched = sum(1 for token in tokens if token in title_text)
    return matched / len(tokens)


def _infer_category(query: str) -> str:
    normalized = query.casefold()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            return category
    return "default"


def _tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣]+", text)
        if len(token) >= 2
    ]


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
