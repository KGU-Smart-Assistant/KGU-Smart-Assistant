import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.schemas.search import SearchResult
from app.services import langchain_rag_service
from app.services.langchain_rag_service import (
    ChromaVectorStoreRetriever,
    HybridSearchRetriever,
    answer_with_langchain_rag,
    build_rag_chain,
    compress_documents_for_query,
    ensure_source_urls,
    expand_search_queries,
    format_documents,
    search_result_to_document,
)


def _search_result(
    *,
    chunk_id: str = "chunk-1",
    title: str = "장학금 신청 안내",
    text: str = "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다. 제출 서류는 신청서입니다.",
    score: float = 0.91,
    confidence: float = 0.91,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="doc-1",
        score=score,
        text=text,
        title=title,
        source_url="https://example.com/scholarship",
        category="scholarship",
        published_at="2026-05-01T00:00:00",
        score_breakdown={"semantic": 0.8, "lexical": 1.0, "confidence": confidence},
    )


def test_search_result_to_document_preserves_metadata() -> None:
    document = search_result_to_document(_search_result())

    assert document.page_content.startswith("장학금 신청 기간")
    assert document.metadata["chunk_id"] == "chunk-1"
    assert document.metadata["title"] == "장학금 신청 안내"
    assert document.metadata["domain"] == "scholarship"
    assert document.metadata["confidence"] == 0.91


def test_expand_search_queries_uses_domain_dictionary() -> None:
    queries = expand_search_queries("성적장학금 신청 기간 알려줘")

    assert queries[0] == "성적장학금 신청 기간 알려줘"
    assert "성적향상장학금 신청 안내" in queries
    assert "장학금 신청 안내" in queries


def test_hybrid_search_retriever_runs_expanded_queries_and_deduplicates() -> None:
    captured = []

    def fake_search(*, query, top_k, category):
        captured.append((query, category))
        return [_search_result(chunk_id="same", title=f"{query} 결과", score=0.8)]

    retriever = HybridSearchRetriever(top_k=3, category="scholarship", search_fn=fake_search)
    documents = retriever.invoke("성적장학금 신청 기간 알려줘")

    assert len(captured) > 1
    assert all(category == "scholarship" for _query, category in captured)
    assert len(documents) == 1
    assert "성적장학금 신청 기간 알려줘" in documents[0].metadata["matched_queries"]


def test_contextual_compression_keeps_query_related_sentences() -> None:
    document = Document(
        page_content=(
            "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다. "
            "문의처는 학생지원처입니다. "
            "제출 서류는 신청서와 성적증명서입니다."
        ),
        metadata={"title": "장학금 신청 안내"},
    )

    compressed = compress_documents_for_query("장학금 신청 기간", [document], sentence_limit=1)

    assert compressed[0].metadata["compressed"] is True
    assert "장학금 신청 기간" in compressed[0].page_content
    assert "제출 서류" not in compressed[0].page_content


def test_langchain_chain_formats_context_and_calls_answer_fn() -> None:
    captured = {}

    class StaticRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [search_result_to_document(_search_result())]

    def fake_answer(prompt: str) -> str:
        captured["prompt"] = prompt
        return "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다."

    chain = build_rag_chain(retriever=StaticRetriever(), answer_fn=fake_answer)
    result = chain.invoke("장학금 신청 기간 알려줘")

    assert "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다." in result.reply
    assert "https://example.com/scholarship" not in result.reply
    assert result.documents[0].metadata["source_url"] == "https://example.com/scholarship"
    assert "검색 근거:" in captured["prompt"]
    assert "영어 답변을 쓰지 말고" in captured["prompt"]
    assert "근거 번호를 [1]처럼" in captured["prompt"]
    assert "[1] 장학금 신청 안내" in captured["prompt"]
    assert "출처:" not in result.reply
    assert result.documents[0].metadata["source_number"] == 1


def test_langchain_chain_rewrites_english_answers_to_korean() -> None:
    calls = []

    class StaticRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [search_result_to_document(_search_result())]

    def fake_answer(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "The scholarship application period is May 1 to May 10."
        assert "다음 답변을" in prompt
        return "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다."

    chain = build_rag_chain(retriever=StaticRetriever(), answer_fn=fake_answer)
    result = chain.invoke("장학금 신청 기간 알려줘")

    assert len(calls) == 2
    assert "장학금 신청 기간은 5월 1일부터 5월 10일까지입니다." in result.reply
    assert "The scholarship" not in result.reply
    assert "https://example.com/scholarship" not in result.reply


def test_langchain_chain_uses_source_only_reply_when_korean_rewrite_fails() -> None:
    class StaticRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [search_result_to_document(_search_result())]

    chain = build_rag_chain(
        retriever=StaticRetriever(),
        answer_fn=lambda prompt: "The answer is unavailable.",
    )
    result = chain.invoke("장학금 신청 기간 알려줘")

    assert "답변을 생성하지 못했습니다" in result.reply
    assert "출처:" not in result.reply
    assert "https://example.com/scholarship" not in result.reply


def test_langchain_chain_uses_snippet_fallback_when_model_quota_fails() -> None:
    class StaticRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [
                search_result_to_document(
                    _search_result(
                        title="졸업요건 안내",
                        text="졸업요건은 졸업학점, 졸업논문, 인권과 성평등교육, 졸업인증제를 이수해야 합니다.",
                        score=0.92,
                        confidence=0.92,
                    )
                )
            ]

    chain = build_rag_chain(
        retriever=StaticRetriever(),
        answer_fn=lambda prompt: "현재 Gemini API 사용량 제한에 도달했습니다. 잠시 후 다시 시도해 주세요.",
    )
    result = chain.invoke("졸업요건 알려줘")

    assert "현재 답변을 생성하지 못했습니다" in result.reply
    assert "졸업요건은 졸업학점" not in result.reply
    assert "현재 Gemini API 사용량 제한" not in result.reply
    assert "https://example.com/scholarship" not in result.reply
    assert result.low_confidence is False


def test_answer_with_langchain_rag_returns_documents_context_and_trace() -> None:
    def fake_search(*, query, top_k, category):
        return [_search_result()]

    trace_path = Path(".tmp/test_rag_trace.jsonl")
    trace_path.unlink(missing_ok=True)
    result = answer_with_langchain_rag(
        "장학금 신청 기간 알려줘",
        top_k=2,
        category="scholarship",
        search_fn=fake_search,
        answer_fn=lambda prompt: "문서 근거 답변",
        trace_id="trace-1",
        trace_path=str(trace_path),
    )

    assert "문서 근거 답변" in result.reply
    assert "https://example.com/scholarship" not in result.reply
    assert result.documents[0].metadata["title"] == "장학금 신청 안내"
    assert "장학금 신청 기간" in result.context
    assert result.expanded_queries[0] == "장학금 신청 기간 알려줘"
    assert result.confidence == 0.91
    assert "trace-1" in trace_path.read_text(encoding="utf-8")
    trace_path.unlink(missing_ok=True)


def test_low_confidence_result_limits_answer_generation() -> None:
    class StaticRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [search_result_to_document(_search_result(confidence=0.1))]

    chain = build_rag_chain(
        retriever=StaticRetriever(),
        answer_fn=lambda prompt: "이 답변은 호출되면 안 됩니다.",
        confidence_threshold=0.35,
    )
    result = chain.invoke("장학금 신청 기간 알려줘")

    assert result.low_confidence is True
    assert "관련 자료를 충분히 찾지 못했습니다" in result.reply
    assert "출처:" not in result.reply
    assert "https://example.com/scholarship" not in result.reply


def test_chroma_vectorstore_retriever_adapts_rows_to_documents(monkeypatch) -> None:
    monkeypatch.setattr(langchain_rag_service, "embed_text", lambda query: [0.1, 0.2])
    monkeypatch.setattr(
        langchain_rag_service,
        "query_embedded_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "distance": 0.2,
                "text": "졸업요건 본문",
                "title": "졸업요건 안내",
                "source_url": "https://example.com/graduation",
                "category": "graduation",
            }
        ],
    )

    retriever = ChromaVectorStoreRetriever(top_k=1, category="graduation")
    documents = retriever.invoke("졸업요건")

    assert documents[0].metadata["title"] == "졸업요건 안내"
    assert documents[0].metadata["score"] > 0


def test_format_documents_includes_source_score_and_matched_queries() -> None:
    document = search_result_to_document(_search_result())
    document.metadata["matched_queries"] = ["장학금 신청 기간"]
    context = format_documents([document])

    assert "[1] 장학금 신청 안내" in context
    assert "source_url: https://example.com/scholarship" in context
    assert "score: 0.91" in context
    assert "confidence: 0.91" in context
    assert "matched_queries: 장학금 신청 기간" in context




def test_parent_expanded_metadata_flows_to_context_and_trace() -> None:
    document = search_result_to_document(
        _search_result(
            chunk_id="parent-chunk",
            text="인접한 4월 학사일정입니다.",
            score=0.01,
            confidence=0.91,
        ).model_copy(update={"score_breakdown": {"parent_expanded": 1.0, "confidence": 0.91}})
    )
    trace_path = Path(".tmp/test_parent_metadata_trace.jsonl")
    trace_path.unlink(missing_ok=True)
    result = langchain_rag_service.LangChainRagResult(
        reply="학사일정 답변",
        documents=[document],
        context=format_documents([document]),
        expanded_queries=["학사일정"],
        trace_id="parent-meta",
        confidence=0.91,
    )

    assert document.metadata["parent_expanded"] is True
    assert "parent_expanded: true" in result.context

    langchain_rag_service.trace_rag_result("학사일정", result, trace_path=str(trace_path))

    payload = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert payload["documents"][0]["parent_expanded"] is True
    trace_path.unlink(missing_ok=True)


def test_ensure_source_urls_strips_inline_source_section() -> None:
    document = search_result_to_document(_search_result())

    reply = ensure_source_urls(
        "답변입니다.\n\n출처:\n- [1] 장학금 신청 안내: https://example.com/scholarship",
        [document],
    )

    assert reply == "답변입니다."
    assert "출처:" not in reply
    assert "https://example.com/scholarship" not in reply
