import os
from pathlib import Path

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.schemas.search import SearchResult
from app.services import retriever_evaluation
from app.services.langchain_rag_service import LangChainRagResult
from app.services.retriever_evaluation import (
    RagEvalQuestion,
    SearchEvalCase,
    evaluate_answer_quality,
    evaluate_rag_eval_smoke,
    evaluate_search_retrievers,
    load_eval_cases,
    load_rag_eval_questions,
    summarize_answer_eval_results,
    summarize_rag_eval_smoke,
    summarize_eval_results,
    write_answer_eval_results_csv,
    write_eval_results_csv,
)

class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.documents


def test_evaluate_search_retrievers_compares_baseline_and_langchain_hits() -> None:
    cases = [
        SearchEvalCase(
            query="성적장학금 신청 기간 알려줘",
            expected_terms=("성적향상장학금", "장학금"),
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
                metadata={"title": "성적향상장학금 신청 안내", "source_url": "https://example.com/scholarship"},
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
    assert results[0].langchain_urls == ["https://example.com/scholarship"]
    assert summary["baseline_recall_at_k"] == 0.0
    assert summary["langchain_recall_at_k"] == 1.0


def test_evaluate_answer_quality_checks_source_and_expected_terms(monkeypatch) -> None:
    document = Document(
        page_content="본문",
        metadata={"title": "장학금 신청 안내", "source_url": "https://example.com/scholarship", "score": 0.91},
    )

    monkeypatch.setattr(
        retriever_evaluation,
        "answer_with_langchain_rag",
        lambda query, **kwargs: LangChainRagResult(
            reply="장학금 신청 안내입니다.\n\n출처:\n- https://example.com/scholarship",
            documents=[document],
            context="context",
            expanded_queries=[query],
            confidence=0.91,
            low_confidence=False,
        ),
    )

    results = evaluate_answer_quality(
        [SearchEvalCase(query="장학금 신청 알려줘", expected_terms=("장학금",), category="scholarship")]
    )
    summary = summarize_answer_eval_results(results)

    assert results[0].has_source_url is True
    assert results[0].expected_term_hit is True
    assert summary["source_url_rate"] == 1.0
    assert summary["expected_term_rate"] == 1.0


def test_load_eval_cases_supports_json_and_csv() -> None:
    json_cases = load_eval_cases("evaluation/search_eval_cases.json")
    csv_cases = load_eval_cases("evaluation/search_eval_cases.csv")

    assert len(json_cases) >= 70
    assert json_cases[0].query == "성적향상장학금 신청 기간과 기준은 어디서 확인해?"
    assert json_cases[0].expected_terms == ("성적향상장학금", "장학금")
    assert csv_cases[0].category == "scholarship"


def test_write_eval_results_csv(tmp_path: Path) -> None:
    retriever = StaticRetriever(
        documents=[Document(page_content="본문", metadata={"title": "장학금 안내", "source_url": "https://example.com/a"})]
    )

    results = evaluate_search_retrievers(
        [SearchEvalCase(query="장학금", expected_terms=("장학금",), category="scholarship")],
        retriever=retriever,
        baseline_search=lambda **kwargs: [],
    )
    output = tmp_path / "results.csv"
    write_eval_results_csv(results, output)

    text = output.read_text(encoding="utf-8-sig")
    assert "장학금" in text
    assert "https://example.com/a" in text


def test_write_answer_eval_results_csv(tmp_path: Path) -> None:
    results = evaluate_answer_quality.__annotations__
    assert "return" in results

    output = tmp_path / "answer-results.csv"
    write_answer_eval_results_csv(
        [
            retriever_evaluation.AnswerEvalResult(
                query="장학금",
                category="scholarship",
                detail="summary",
                expected_terms=("장학금",),
                has_source_url=True,
                low_confidence=False,
                expected_term_hit=True,
                document_count=1,
                confidence=0.9,
                reply_preview="장학금 답변",
            )
        ],
        output,
    )

    text = output.read_text(encoding="utf-8-sig")
    assert "has_source_url" in text
    assert "장학금 답변" in text


def test_rag_eval_smoke_fixture_has_required_fields() -> None:
    questions = load_rag_eval_questions("tests/fixtures/rag_eval_questions.jsonl")

    assert len(questions) == 20
    assert questions[0].id == "rag-smoke-001"
    assert questions[0].expected_domains == ("scholarship",)
    assert questions[0].expected_detail == "period"
    assert questions[0].expected_source_number == 1
    assert questions[0].required_trace_fields == ("source_url", "title", "source_number", "retrieval_sources")
    assert questions[-1].should_answer is False
    assert questions[-1].ambiguity == "multi_domain"


def test_rag_eval_smoke_runs_offline_with_injected_fakes() -> None:
    questions = load_rag_eval_questions("tests/fixtures/rag_eval_questions.jsonl")[:3]

    def fake_route(question: str):
        match = next(item for item in questions if item.question == question)
        return {
            "route": match.expected_route,
            "domains": match.expected_domains,
            "detail": match.expected_detail,
            "should_answer": match.should_answer,
        }

    def fake_retrieve(question: RagEvalQuestion):
        return [
            {
                "source_url": question.gold_source_url,
                "title": "gold",
                "source_number": question.expected_source_number,
                "retrieval_sources": ["fixture"],
            },
            {"source_url": "https://example.com/other", "title": "other"},
        ]

    results = evaluate_rag_eval_smoke(questions, route_fn=fake_route, retrieve_fn=fake_retrieve)
    summary = summarize_rag_eval_smoke(results)

    assert all(result.route_hit for result in results)
    assert all(result.domain_top1_hit for result in results)
    assert all(result.detail_hit for result in results)
    assert all(result.source_number_hit for result in results)
    assert all(result.trace_fields_hit for result in results)
    assert summary["source_number_accuracy"] == 1.0
    assert summary["trace_fields_accuracy"] == 1.0
    assert summary["retrieval_hit_at_1"] == 1.0
    assert summary["retrieval_mrr"] == 1.0


def test_rag_eval_smoke_reports_retrieval_miss_without_live_search() -> None:
    questions = load_rag_eval_questions("tests/fixtures/rag_eval_questions.jsonl")[:1]

    results = evaluate_rag_eval_smoke(
        questions,
        route_fn=lambda _question: {"route": "rag", "domains": ["scholarship"], "detail": "period"},
        retrieve_fn=lambda _question: [{"source_url": "https://example.com/not-gold"}],
    )
    summary = summarize_rag_eval_smoke(results)

    assert results[0].retrieval_rank is None
    assert results[0].source_number_hit is False
    assert results[0].trace_fields_hit is False
    assert summary["retrieval_hit_at_5"] == 0.0
    assert summary["source_number_accuracy"] == 0.0
    assert summary["trace_fields_accuracy"] == 0.0
