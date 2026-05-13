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
    "support": {"similarity": 0.50, "freshness": 0.35, "title": 0.15},
    "materials": {"similarity": 0.60, "freshness": 0.15, "title": 0.25},
    "faq": {"similarity": 0.80, "freshness": 0.00, "title": 0.20},
    "academic_schedule": {"similarity": 0.50, "freshness": 0.25, "title": 0.25},
    "graduation": {"similarity": 0.70, "freshness": 0.05, "title": 0.25},
    "student_life": {"similarity": 0.70, "freshness": 0.10, "title": 0.20},
    "career": {"similarity": 0.65, "freshness": 0.15, "title": 0.20},
    "default": {"similarity": 0.75, "freshness": 0.10, "title": 0.15},
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "academic_schedule": ("학사일정", "수강신청", "개강", "종강", "시험", "학기"),
    "graduation": ("졸업", "졸업요건", "졸업학점", "학위", "전공학점"),
    "support": ("장학", "장학금", "등록금", "학자금"),
    "materials": ("자료", "자료실", "첨부", "파일", "서식", "양식", "pdf", "hwp"),
    "faq": ("faq", "질문", "답변", "자주 묻는"),
    "notice": ("공지", "공지사항", "모집", "신청", "안내"),
}


def search_documents(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    rag_domain: Optional[str] = None,
    rag_detail: Optional[str] = None,
) -> List[SearchResult]:
    """Return the most relevant document chunks for a query.

    This service interface is fixed first so the crawler, chunker,
    embedder, and vector store can all target the same contract.
    """
    query_embedding = embed_text(query)
    candidate_count = _candidate_count(top_k)
    mapped_domain_category = _category_from_rag_domain(rag_domain)
    retrieval_category = category or mapped_domain_category
    rerank_category = (
        category
        or mapped_domain_category
        or ("default" if rag_domain is not None else _infer_category(query))
    )
    rows = query_embedded_chunks(
        query_embedding=query_embedding,
        top_k=candidate_count,
        category=retrieval_category,
    )
    if retrieval_category is not None and not rows:
        rows = query_embedded_chunks(
            query_embedding=query_embedding,
            top_k=candidate_count,
            category=None,
        )
    ranked_rows = _rerank_rows(
        rows=rows,
        query=query,
        category=rerank_category,
        rag_detail=rag_detail,
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
    rag_detail: str | None = None,
) -> List[Dict[str, Any]]:
    effective_category = category or "default"
    weights = CATEGORY_WEIGHTS.get(effective_category, CATEGORY_WEIGHTS["default"])
    ranked_rows: List[Dict[str, Any]] = []
    for row in rows:
        similarity = _distance_to_score(row.get("distance"))
        freshness = _freshness_score(row.get("published_at"))
        title = _title_match_score(query=query, title=row.get("title") or "")
        detail = _detail_match_score(
            detail=rag_detail,
            title=row.get("title") or "",
            text=row.get("text") or "",
        )
        score = (
            similarity * weights["similarity"]
            + freshness * weights["freshness"]
            + title * weights["title"]
            + detail * 0.10
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


def _category_from_rag_domain(rag_domain: str | None) -> str | None:
    mapping = {
        "scholarship": "support",
        "course_registration": "academic_schedule",
        "academic_calendar": "academic_schedule",
        "graduation": "graduation",
        "document_materials": "materials",
        "student_life": "student_life",
        "career_support": "career",
        "general_notice": "notice",
        "department_notice": "notice",
    }
    if rag_domain is None:
        return None
    return mapping.get(rag_domain)


def _detail_match_score(*, detail: str | None, title: str, text: str) -> float:
    if not detail:
        return 0.0

    detail_keywords: dict[str, tuple[str, ...]] = {
        "period": ("기간", "일정", "마감", "언제"),
        "eligibility": ("대상", "자격", "조건", "가능"),
        "procedure": ("신청", "절차", "방법", "접수"),
        "required_documents": ("서류", "제출", "증빙", "첨부", "신청서", "양식", "서식"),
        "benefit": ("금액", "혜택", "지원액", "감면", "수혜"),
        "announcement_lookup": ("공지", "안내", "모집", "발표"),
        "summary": ("요약", "정리", "핵심"),
    }
    keywords = detail_keywords.get(detail)
    if not keywords:
        return 0.0

    haystack = f"{title} {text}".casefold()
    matched = sum(1 for keyword in keywords if keyword.casefold() in haystack)
    return matched / len(keywords)


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
