from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.services.langchain_rag_service import HybridSearchRetriever, answer_with_langchain_rag
from app.services.search_service import search_documents


@dataclass(frozen=True)
class SearchEvalCase:
    query: str
    expected_terms: tuple[str, ...]
    category: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SearchEvalResult:
    query: str
    expected_terms: tuple[str, ...]
    baseline_hit: bool
    langchain_hit: bool
    baseline_titles: list[str]
    langchain_titles: list[str]
    baseline_urls: list[str]
    langchain_urls: list[str]


@dataclass(frozen=True)
class AnswerEvalResult:
    query: str
    category: str | None
    detail: str | None
    expected_terms: tuple[str, ...]
    has_source_url: bool
    low_confidence: bool
    expected_term_hit: bool
    document_count: int
    confidence: float
    reply_preview: str


def evaluate_search_retrievers(
    cases: Sequence[SearchEvalCase],
    *,
    retriever: BaseRetriever | None = None,
    top_k: int = 5,
    baseline_search: Callable[..., list] = search_documents,
) -> list[SearchEvalResult]:
    results = []
    for case in cases:
        baseline_kwargs = {"query": case.query, "top_k": top_k, "category": case.category}
        if case.detail is not None:
            baseline_kwargs["detail"] = case.detail
        baseline_results = baseline_search(**baseline_kwargs)
        effective_retriever = retriever or HybridSearchRetriever(top_k=top_k, category=case.category, detail=case.detail)
        langchain_documents = effective_retriever.invoke(case.query)
        baseline_titles = [getattr(result, "title", "") for result in baseline_results]
        langchain_titles = [str(document.metadata.get("title", "")) for document in langchain_documents]
        baseline_urls = [getattr(result, "source_url", "") for result in baseline_results]
        langchain_urls = [str(document.metadata.get("source_url", "")) for document in langchain_documents]
        results.append(
            SearchEvalResult(
                query=case.query,
                expected_terms=case.expected_terms,
                baseline_hit=_contains_expected_terms(baseline_titles, case.expected_terms),
                langchain_hit=_contains_expected_terms(langchain_titles, case.expected_terms),
                baseline_titles=baseline_titles,
                langchain_titles=langchain_titles,
                baseline_urls=baseline_urls,
                langchain_urls=langchain_urls,
            )
        )
    return results


def summarize_eval_results(results: Sequence[SearchEvalResult]) -> dict[str, float | int]:
    total = len(results)
    baseline_hits = sum(1 for result in results if result.baseline_hit)
    langchain_hits = sum(1 for result in results if result.langchain_hit)
    return {
        "total": total,
        "baseline_hits": baseline_hits,
        "langchain_hits": langchain_hits,
        "baseline_recall_at_k": baseline_hits / total if total else 0.0,
        "langchain_recall_at_k": langchain_hits / total if total else 0.0,
    }


def evaluate_answer_quality(
    cases: Sequence[SearchEvalCase],
    *,
    top_k: int = 5,
    answer_fn=None,
    confidence_threshold: float = 0.35,
) -> list[AnswerEvalResult]:
    results: list[AnswerEvalResult] = []
    for case in cases:
        kwargs = {
            "top_k": top_k,
            "category": case.category,
            "detail": case.detail,
            "confidence_threshold": confidence_threshold,
        }
        if answer_fn is not None:
            kwargs["answer_fn"] = answer_fn
        rag_result = answer_with_langchain_rag(case.query, **kwargs)
        source_urls = [str(document.metadata.get("source_url") or "") for document in rag_result.documents]
        has_source_url = any(url and url in rag_result.reply for url in source_urls)
        expected_term_hit = _contains_expected_terms([rag_result.reply], case.expected_terms)
        results.append(
            AnswerEvalResult(
                query=case.query,
                category=case.category,
                detail=case.detail,
                expected_terms=case.expected_terms,
                has_source_url=has_source_url,
                low_confidence=rag_result.low_confidence,
                expected_term_hit=expected_term_hit,
                document_count=len(rag_result.documents),
                confidence=rag_result.confidence,
                reply_preview=rag_result.reply[:300],
            )
        )
    return results


def summarize_answer_eval_results(results: Sequence[AnswerEvalResult]) -> dict[str, float | int]:
    total = len(results)
    source_url_hits = sum(1 for result in results if result.has_source_url)
    expected_term_hits = sum(1 for result in results if result.expected_term_hit)
    low_confidence_count = sum(1 for result in results if result.low_confidence)
    no_document_count = sum(1 for result in results if result.document_count == 0)
    return {
        "total": total,
        "source_url_hits": source_url_hits,
        "expected_term_hits": expected_term_hits,
        "low_confidence_count": low_confidence_count,
        "no_document_count": no_document_count,
        "source_url_rate": source_url_hits / total if total else 0.0,
        "expected_term_rate": expected_term_hits / total if total else 0.0,
    }


def load_eval_cases(path: str | Path) -> list[SearchEvalCase]:
    source = Path(path)
    if not source.exists() and not source.is_absolute():
        backend_relative_source = Path(__file__).resolve().parents[2] / source
        if backend_relative_source.exists():
            source = backend_relative_source
    if source.suffix.casefold() == ".csv":
        return _load_csv_cases(source)
    return _load_json_cases(source)


def write_eval_results_csv(results: Sequence[SearchEvalResult], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query",
                "expected_terms",
                "baseline_hit",
                "langchain_hit",
                "baseline_titles",
                "langchain_titles",
                "baseline_urls",
                "langchain_urls",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "query": result.query,
                    "expected_terms": "|".join(result.expected_terms),
                    "baseline_hit": result.baseline_hit,
                    "langchain_hit": result.langchain_hit,
                    "baseline_titles": " | ".join(result.baseline_titles),
                    "langchain_titles": " | ".join(result.langchain_titles),
                    "baseline_urls": " | ".join(result.baseline_urls),
                    "langchain_urls": " | ".join(result.langchain_urls),
                }
            )


def write_answer_eval_results_csv(results: Sequence[AnswerEvalResult], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query",
                "category",
                "detail",
                "expected_terms",
                "has_source_url",
                "low_confidence",
                "expected_term_hit",
                "document_count",
                "confidence",
                "reply_preview",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "query": result.query,
                    "category": result.category or "",
                    "detail": result.detail or "",
                    "expected_terms": "|".join(result.expected_terms),
                    "has_source_url": result.has_source_url,
                    "low_confidence": result.low_confidence,
                    "expected_term_hit": result.expected_term_hit,
                    "document_count": result.document_count,
                    "confidence": result.confidence,
                    "reply_preview": result.reply_preview.replace("\n", " "),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate KGU search and RAG answer quality.")
    parser.add_argument("--cases", default="evaluation/search_eval_cases.json", help="JSON or CSV evaluation cases.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("search", "answer", "all"), default="search")
    parser.add_argument("--output", default=".tmp/retriever_eval_results.csv")
    parser.add_argument("--answer-output", default=".tmp/answer_eval_results.csv")
    args = parser.parse_args(argv)

    cases = load_eval_cases(args.cases)
    payload: dict[str, object] = {}
    if args.mode in {"search", "all"}:
        results = evaluate_search_retrievers(cases, top_k=args.top_k)
        summary = summarize_eval_results(results)
        write_eval_results_csv(results, args.output)
        payload["search"] = {"summary": summary, "output": args.output}
    if args.mode in {"answer", "all"}:
        answer_results = evaluate_answer_quality(cases, top_k=args.top_k)
        answer_summary = summarize_answer_eval_results(answer_results)
        write_answer_eval_results_csv(answer_results, args.answer_output)
        payload["answer"] = {"summary": answer_summary, "output": args.answer_output}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _load_json_cases(path: Path) -> list[SearchEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload["cases"] if isinstance(payload, dict) else payload
    return [_case_from_mapping(row) for row in rows]


def _load_csv_cases(path: Path) -> list[SearchEvalCase]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [_case_from_mapping(row) for row in csv.DictReader(file)]


def _case_from_mapping(row: dict) -> SearchEvalCase:
    expected = row.get("expected_terms") or row.get("expected") or []
    if isinstance(expected, str):
        expected_terms = tuple(term.strip() for term in re_split_terms(expected) if term.strip())
    else:
        expected_terms = tuple(str(term).strip() for term in expected if str(term).strip())
    return SearchEvalCase(
        query=str(row["query"]).strip(),
        expected_terms=expected_terms,
        category=str(row.get("category") or "").strip() or None,
        detail=str(row.get("detail") or "").strip() or None,
    )


def re_split_terms(value: str) -> list[str]:
    return [term for term in value.replace(",", "|").split("|")]


def _contains_expected_terms(titles: Iterable[str], expected_terms: tuple[str, ...]) -> bool:
    joined = " ".join(titles).casefold()
    return any(term.casefold() in joined for term in expected_terms)


if __name__ == "__main__":
    raise SystemExit(main())
