from app.services import search_service


def test_search_documents_embeds_query_and_maps_vector_results(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, category):
        captured["query_embedding"] = query_embedding
        captured["top_k"] = top_k
        captured["category"] = category
        return [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "distance": 0.25,
                "text": "Document chunk text",
                "title": "Document title",
                "source_url": "https://example.com/source",
            }
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(
        query="scholarship",
        top_k=3,
        category="notice",
    )

    assert captured == {
        "query_embedding": [0.1, 0.2, 0.3],
        "top_k": 12,
        "category": "notice",
    }
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].doc_id == "doc-1"
    assert results[0].score == 0.44
    assert results[0].text == "Document chunk text"
    assert results[0].title == "Document title"
    assert results[0].source_url == "https://example.com/source"
    assert results[0].score_breakdown == {
        "similarity": 0.8,
        "freshness": 0.0,
        "title": 0.0,
    }


def test_search_wraps_results_in_response(monkeypatch) -> None:
    monkeypatch.setattr(
        search_service,
        "search_documents",
        lambda query, top_k, category: [],
    )

    response = search_service.search(query="faq", top_k=2, category=None)

    assert response.query == "faq"
    assert response.results == []


def test_search_documents_reranks_by_category_weights(monkeypatch) -> None:
    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, category):
        return [
            {
                "chunk_id": "similar-old",
                "doc_id": "doc-1",
                "distance": 0.05,
                "text": "장학금 안내",
                "title": "장학금 안내",
                "source_url": "https://example.com/old",
                "published_at": "2018-01-01T00:00:00",
            },
            {
                "chunk_id": "recent-title",
                "doc_id": "doc-2",
                "distance": 0.30,
                "text": "신청 안내",
                "title": "장학금 신청 안내",
                "source_url": "https://example.com/recent",
                "published_at": "2026-01-01T00:00:00",
            },
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(
        query="장학금 신청",
        top_k=2,
        category="scholarship",
    )

    assert [result.chunk_id for result in results] == ["recent-title", "similar-old"]


def test_search_documents_merges_keyword_candidates(monkeypatch) -> None:
    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])

    monkeypatch.setattr(
        search_service,
        "query_embedded_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "vector-only",
                "doc_id": "doc-1",
                "distance": 0.2,
                "text": "공지 본문",
                "title": "공지",
                "source_url": "https://example.com/vector",
            }
        ],
    )
    monkeypatch.setattr(
        search_service,
        "_query_keyword_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "keyword-only",
                "doc_id": "doc-2",
                "distance": 0.0,
                "text": "졸업요건 본문",
                "title": "졸업요건 안내",
                "source_url": "https://example.com/keyword",
            }
        ],
    )

    results = search_service.search_documents(query="졸업요건", top_k=2)

    assert {result.chunk_id for result in results} == {"vector-only", "keyword-only"}
