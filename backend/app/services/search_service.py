from typing import List, Optional

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas import SearchResponse, SearchResult


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
    rows = query_embedded_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        category=category,
    )
    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            score=_distance_to_score(row.get("distance")),
            text=row["text"],
            title=row["title"],
            source_url=row["source_url"],
        )
        for row in rows
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
