from app.services import search_service


def test_search_documents_uses_explicit_category_filter(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        captured.setdefault("categories", []).append(domain)
        return [
            {
                "chunk_id": f"chunk-{domain}",
                "doc_id": "doc-1",
                "distance": 0.25,
                "text": "장학금 신청 안내",
                "title": "장학금 신청 안내",
                "source_url": "https://example.com/source",
                "domain": domain,
            }
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(query="장학금 신청", top_k=3, category="scholarship")

    assert captured["categories"] == ["scholarship", "general_notice", "department_notice"]
    assert len(results) == 3
    assert results[0].score_breakdown["category"] == 1.0
    assert "confidence" in results[0].score_breakdown


def test_search_documents_infers_category_and_applies_filter(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        captured.setdefault("categories", []).append(domain)
        return []

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    search_service.search_documents(query="성적향상장학금 신청 기간 알려줘", top_k=3)

    assert captured["categories"] == ["scholarship", "general_notice", "department_notice", None]


def test_search_wraps_results_in_response(monkeypatch) -> None:
    monkeypatch.setattr(search_service, "search_documents", lambda query, top_k, category: [])

    response = search_service.search(query="faq", top_k=2, category=None)

    assert response.query == "faq"
    assert response.results == []


def test_search_documents_reranks_by_category_weights(monkeypatch) -> None:
    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        return [
            {
                "chunk_id": "similar-old",
                "doc_id": "doc-1",
                "distance": 0.05,
                "text": "장학금 안내",
                "title": "장학금 안내",
                "source_url": "https://example.com/old",
                "domain": domain,
                "published_at": "2018-01-01T00:00:00",
            },
            {
                "chunk_id": "recent-title",
                "doc_id": "doc-2",
                "distance": 0.30,
                "text": "신청 안내",
                "title": "장학금 신청 안내",
                "source_url": "https://example.com/recent",
                "domain": domain,
                "published_at": "2026-01-01T00:00:00",
            },
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(query="장학금 신청", top_k=2, category="scholarship")

    assert results[0].chunk_id == "recent-title"


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
                "lexical_score": 1.0,
                "retrieval_sources": {"keyword"},
                "text": "졸업요건과 전공학점 기준 안내",
                "title": "졸업요건 안내",
                "source_url": "https://example.com/keyword",
                "domain": "graduation",
            }
        ],
    )

    results = search_service.search_documents(query="졸업요건", top_k=2)

    assert {result.chunk_id for result in results} == {"vector-only", "keyword-only"}


def test_keyword_signal_can_beat_weak_vector_match(monkeypatch) -> None:
    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        search_service,
        "query_embedded_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "weak-vector",
                "doc_id": "doc-1",
                "distance": 0.7,
                "text": "캠퍼스 일반 공지",
                "title": "일반 안내",
                "source_url": "https://example.com/vector",
                "domain": "general_notice",
            }
        ],
    )
    monkeypatch.setattr(
        search_service,
        "_query_keyword_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "exact-keyword",
                "doc_id": "doc-2",
                "lexical_score": 1.0,
                "retrieval_sources": {"keyword"},
                "text": "졸업요건과 전공학점 기준 안내",
                "title": "졸업요건 안내",
                "source_url": "https://example.com/graduation",
                "domain": "graduation",
            }
        ],
    )

    results = search_service.search_documents(query="졸업요건", top_k=2)

    assert results[0].chunk_id == "exact-keyword"
    assert results[0].score_breakdown["lexical"] == 1.0
    assert results[0].score_breakdown["category"] == 1.0


def test_low_confidence_category_search_falls_back_to_broad_search(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        captured.append(domain)
        if domain in {"scholarship", "general_notice", "department_notice"}:
            return []
        return [
            {
                "chunk_id": "fallback",
                "doc_id": "doc-1",
                "distance": 0.1,
                "text": "성적향상장학금 안내",
                "title": "성적향상장학금 신청 안내",
                "source_url": "https://example.com/fallback",
                "domain": "general_notice",
            }
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(query="성적향상장학금 신청 기간", top_k=1, category="scholarship")

    assert captured == ["scholarship", "general_notice", "department_notice", None]
    assert results[0].chunk_id == "fallback"
    assert results[0].score_breakdown["fallback_used"] == 1.0


def test_search_documents_accepts_rag_domain_filters(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        captured.append(domain)
        return []

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    search_service.search_documents(
        query="장학금 신청기간 알려줘",
        top_k=2,
        rag_domain="scholarship",
        rag_domains=["course_registration"],
        rag_detail="period",
    )

    assert captured == [
        "scholarship",
        "general_notice",
        "department_notice",
        "course_registration",
        "academic_calendar",
        None,
    ]


def test_search_documents_soft_boosts_department_scope(monkeypatch) -> None:
    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        return [
            {
                "chunk_id": "university",
                "doc_id": "doc-1",
                "distance": 0.1,
                "text": "취업 공지입니다.",
                "title": "취업 공지",
                "source_url": "https://example.com/university",
                "domain": domain,
                "department": "university",
            },
            {
                "chunk_id": "department",
                "doc_id": "doc-2",
                "distance": 0.1,
                "text": "컴퓨터공학과 취업 공지입니다.",
                "title": "컴퓨터공학과 취업 공지",
                "source_url": "https://example.com/department",
                "domain": domain,
                "department": "computer_science",
            },
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(
        query="컴퓨터공학과 취업 공지 알려줘",
        top_k=2,
        rag_domain="career_support",
        rag_detail="announcement_lookup",
        source_scope="department",
    )

    assert [result.chunk_id for result in results] == ["department", "university"]


def test_search_documents_boosts_secondary_rag_details(monkeypatch) -> None:
    secondary_keyword = search_service.DETAIL_KEYWORDS["required_documents"][0]

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        return [
            {
                "chunk_id": "generic",
                "doc_id": "doc-1",
                "distance": 0.2,
                "text": "일반 안내",
                "title": "일반 안내",
                "source_url": "https://example.com/generic",
                "domain": domain,
            },
            {
                "chunk_id": "secondary-detail",
                "doc_id": "doc-2",
                "distance": 0.2,
                "text": secondary_keyword,
                "title": secondary_keyword,
                "source_url": "https://example.com/detail",
                "domain": domain,
            },
        ]

    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(
        query="장학금 신청 알려줘",
        top_k=2,
        rag_domain="scholarship",
        rag_detail="period",
        rag_details=["required_documents"],
    )

    assert results[0].chunk_id == "secondary-detail"
    assert results[0].score_breakdown["detail"] > 0.0


def test_search_documents_uses_rewritten_queries(monkeypatch) -> None:
    captured_queries = []

    monkeypatch.setattr(search_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(search_service, "_query_keyword_chunks", lambda **kwargs: [])

    def _query_embedded_chunks(*, query_embedding, top_k, domain):
        query_index = len(captured_queries)
        captured_queries.append(query_embedding)
        return [
            {
                "chunk_id": f"chunk-{query_index}",
                "doc_id": "doc-1",
                "distance": 0.2,
                "text": "등록금 환불 신청서 안내",
                "title": "등록금 환불 신청서",
                "source_url": "https://example.com/refund",
                "domain": domain,
            }
        ]

    monkeypatch.setattr(search_service, "embed_text", lambda query: captured_queries.append(query) or [0.1])
    monkeypatch.setattr(search_service, "query_embedded_chunks", _query_embedded_chunks)

    results = search_service.search_documents(
        query="등록금 환불",
        top_k=2,
        rag_domain="tuition",
        rewritten_queries=["등록금 환불 신청서"],
    )

    assert "등록금 환불" in captured_queries
    assert "등록금 환불 신청서" in captured_queries
    assert results


def test_merge_preserves_vector_and_keyword_signals() -> None:
    rows = search_service._merge_rows(
        vector_rows=[
            {
                "chunk_id": "same",
                "doc_id": "doc-1",
                "distance": 0.2,
                "text": "본문",
                "title": "제목",
                "source_url": "https://example.com/vector",
                "retrieval_sources": {"vector"},
            }
        ],
        keyword_rows=[
            {
                "chunk_id": "same",
                "doc_id": "doc-1",
                "lexical_score": 1.0,
                "text": "본문",
                "title": "제목",
                "source_url": "https://example.com/keyword",
                "retrieval_sources": {"keyword"},
            }
        ],
    )

    assert rows[0]["distance"] == 0.2
    assert rows[0]["lexical_score"] == 1.0
    assert rows[0]["retrieval_sources"] == {"vector", "keyword"}


def test_tokenize_keeps_korean_words() -> None:
    assert search_service._tokenize("졸업요건과 전공 학점") == ["졸업요건과", "전공", "학점"]
