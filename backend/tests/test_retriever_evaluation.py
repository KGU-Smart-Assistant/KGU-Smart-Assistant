from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.schemas.search import SearchResult
from app.services.retriever_evaluation import (
    SearchEvalCase,
    evaluate_search_retrievers,
    summarize_eval_results,
)


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.documents


def test_evaluate_search_retrievers_compares_baseline_and_langchain_hits() -> None:
    cases = [
        SearchEvalCase(
            query="성적장학금 신청 기간 알려줘",
            expected_terms=("성적우수장학금", "장학금"),
            category="scholarship",
        )
    ]

    def baseline_search(*, query, top_k, category):
        return [
            SearchResult(
                chunk_id="chunk-1",
                doc_id="doc-1",
                score=0.8,
                text="본문",
                title="일반 공지",
                source_url="https://example.com/notice",
            )
        ]

    retriever = StaticRetriever(
        documents=[
            Document(
                page_content="본문",
                metadata={"title": "성적우수장학금 신청 안내"},
            )
        ]
    )

    results = evaluate_search_retrievers(
        cases,
        retriever=retriever,
        baseline_search=baseline_search,
    )
    summary = summarize_eval_results(results)

    assert results[0].baseline_hit is False
    assert results[0].langchain_hit is True
    assert summary["baseline_recall_at_k"] == 0.0
    assert summary["langchain_recall_at_k"] == 1.0
