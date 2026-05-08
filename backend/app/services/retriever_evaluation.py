from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.services.search_service import search_documents


@dataclass(frozen=True)
class SearchEvalCase:
    query: str
    expected_terms: tuple[str, ...]
    category: str | None = None


@dataclass(frozen=True)
class SearchEvalResult:
    query: str
    expected_terms: tuple[str, ...]
    baseline_hit: bool
    langchain_hit: bool
    baseline_titles: list[str]
    langchain_titles: list[str]


def evaluate_search_retrievers(
    cases: Sequence[SearchEvalCase],
    *,
    retriever: BaseRetriever,
    top_k: int = 5,
    baseline_search: Callable[..., list] = search_documents,
) -> list[SearchEvalResult]:
    results = []
    for case in cases:
        baseline_results = baseline_search(query=case.query, top_k=top_k, category=case.category)
        langchain_documents = retriever.invoke(case.query)
        baseline_titles = [getattr(result, "title", "") for result in baseline_results]
        langchain_titles = [str(document.metadata.get("title", "")) for document in langchain_documents]
        results.append(
            SearchEvalResult(
                query=case.query,
                expected_terms=case.expected_terms,
                baseline_hit=_contains_expected_terms(baseline_titles, case.expected_terms),
                langchain_hit=_contains_expected_terms(langchain_titles, case.expected_terms),
                baseline_titles=baseline_titles,
                langchain_titles=langchain_titles,
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


def _contains_expected_terms(titles: Iterable[str], expected_terms: tuple[str, ...]) -> bool:
    joined = " ".join(titles).casefold()
    return any(term.casefold() in joined for term in expected_terms)
