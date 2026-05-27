from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever



@dataclass(frozen=True)
class RagEvalQuestion:
    id: str
    question: str
    expected_route: str
    expected_domains: tuple[str, ...]
    expected_detail: str | None
    gold_source_url: str | None
    should_answer: bool
    ambiguity: str | None = None
    expected_source_number: int | None = None
    required_trace_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagEvalSmokeResult:
    id: str
    question: str
    route_hit: bool
    domain_top1_hit: bool
    domain_top3_hit: bool
    detail_hit: bool
    retrieval_rank: int | None
    should_answer_hit: bool
    source_number_hit: bool = True
    trace_fields_hit: bool = True


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


def load_rag_eval_questions(path: str | Path) -> list[RagEvalQuestion]:
    source = Path(path)
    if not source.exists() and not source.is_absolute():
        backend_relative_source = Path(__file__).resolve().parents[2] / source
        if backend_relative_source.exists():
            source = backend_relative_source

    questions: list[RagEvalQuestion] = []
    with source.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            questions.append(_rag_question_from_mapping(row, line_number=line_number))
    return questions


def evaluate_rag_eval_smoke(
    questions: Sequence[RagEvalQuestion],
    *,
    route_fn: Callable[[str], Mapping[str, Any]],
    retrieve_fn: Callable[[RagEvalQuestion], Sequence[Any]],
) -> list[RagEvalSmokeResult]:
    """Evaluate RAG routing/retrieval smoke cases with fully injected offline functions."""
    results: list[RagEvalSmokeResult] = []
    for question in questions:
        route = route_fn(question.question)
        predicted_route = str(route.get("route") or route.get("expected_route") or "").strip()
        predicted_domains = _normalise_prediction_list(route.get("domains") or route.get("domain") or route.get("rag_domains"))
        predicted_detail = str(route.get("detail") or route.get("rag_detail") or "").strip() or None
        should_answer = bool(route.get("should_answer", True))
        retrieved = list(retrieve_fn(question))
        retrieval_rank = _source_rank(retrieved, question.gold_source_url)
        source_number_hit = _source_number_hit(
            retrieved,
            gold_source_url=question.gold_source_url,
            expected_source_number=question.expected_source_number,
        )
        trace_fields_hit = _trace_fields_hit(
            retrieved,
            gold_source_url=question.gold_source_url,
            required_fields=question.required_trace_fields,
        )
        results.append(
            RagEvalSmokeResult(
                id=question.id,
                question=question.question,
                route_hit=predicted_route == question.expected_route,
                domain_top1_hit=bool(question.expected_domains) and predicted_domains[:1] == list(question.expected_domains[:1]),
                domain_top3_hit=any(domain in predicted_domains[:3] for domain in question.expected_domains),
                detail_hit=(predicted_detail or "unknown") == (question.expected_detail or "unknown"),
                retrieval_rank=retrieval_rank,
                should_answer_hit=should_answer is question.should_answer,
                source_number_hit=source_number_hit,
                trace_fields_hit=trace_fields_hit,
            )
        )
    return results


def summarize_rag_eval_smoke(results: Sequence[RagEvalSmokeResult]) -> dict[str, float | int]:
    total = len(results)
    answerable = [result for result in results if result.retrieval_rank is not None]
    return {
        "total": total,
        "route_accuracy": _rate(result.route_hit for result in results),
        "domain_top1_accuracy": _rate(result.domain_top1_hit for result in results),
        "domain_top3_accuracy": _rate(result.domain_top3_hit for result in results),
        "detail_accuracy": _rate(result.detail_hit for result in results),
        "should_answer_accuracy": _rate(result.should_answer_hit for result in results),
        "source_number_accuracy": _rate(result.source_number_hit for result in results),
        "trace_fields_accuracy": _rate(result.trace_fields_hit for result in results),
        "retrieval_hit_at_1": _rate(result.retrieval_rank == 1 for result in answerable),
        "retrieval_hit_at_3": _rate(result.retrieval_rank is not None and result.retrieval_rank <= 3 for result in answerable),
        "retrieval_hit_at_5": _rate(result.retrieval_rank is not None and result.retrieval_rank <= 5 for result in answerable),
        "retrieval_mrr": sum(1 / result.retrieval_rank for result in answerable if result.retrieval_rank) / len(answerable) if answerable else 0.0,
    }

def search_documents(*args, **kwargs):
    from app.services.search_service import search_documents as real_search_documents

    return real_search_documents(*args, **kwargs)


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
        if retriever is None:
            from app.services.langchain_rag_service import HybridSearchRetriever

            effective_retriever = HybridSearchRetriever(top_k=top_k, category=case.category, detail=case.detail)
        else:
            effective_retriever = retriever
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


def answer_with_langchain_rag(*args, **kwargs):
    from app.services.langchain_rag_service import answer_with_langchain_rag as real_answer_with_langchain_rag

    return real_answer_with_langchain_rag(*args, **kwargs)


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
    parser.add_argument("--mode", choices=("search", "answer", "rag-smoke", "all"), default="search")
    parser.add_argument("--output", default=".tmp/retriever_eval_results.csv")
    parser.add_argument("--answer-output", default=".tmp/answer_eval_results.csv")
    parser.add_argument(
        "--rag-smoke-questions",
        default="tests/fixtures/rag_eval_questions.jsonl",
        help="Offline RAG smoke JSONL fixture used by --mode rag-smoke.",
    )
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
    if args.mode == "rag-smoke":
        smoke_questions = load_rag_eval_questions(args.rag_smoke_questions)
        smoke_results = evaluate_rag_eval_smoke(
            smoke_questions,
            route_fn=lambda question: _fixture_route_prediction(question, smoke_questions),
            retrieve_fn=_fixture_retrieval_rows,
        )
        payload["rag_smoke"] = {"summary": summarize_rag_eval_smoke(smoke_results)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _rag_question_from_mapping(row: Mapping[str, Any], *, line_number: int) -> RagEvalQuestion:
    expected_domains = tuple(str(domain).strip() for domain in row.get("expected_domains", []) if str(domain).strip())
    if not expected_domains:
        raise ValueError(f"RAG eval question line {line_number} must include expected_domains.")
    return RagEvalQuestion(
        id=str(row.get("id") or f"line-{line_number}").strip(),
        question=str(row["question"]).strip(),
        expected_route=str(row["expected_route"]).strip(),
        expected_domains=expected_domains,
        expected_detail=str(row.get("expected_detail") or "unknown").strip() or None,
        gold_source_url=str(row.get("gold_source_url") or "").strip() or None,
        should_answer=bool(row.get("should_answer", True)),
        ambiguity=str(row.get("ambiguity") or "none").strip() or None,
        expected_source_number=_optional_int(row.get("expected_source_number")),
        required_trace_fields=tuple(
            str(field).strip() for field in row.get("required_trace_fields", []) if str(field).strip()
        ),
    )


def _fixture_route_prediction(question: str, questions: Sequence[RagEvalQuestion]) -> dict[str, Any]:
    match = next(item for item in questions if item.question == question)
    return {
        "route": match.expected_route,
        "domains": match.expected_domains,
        "detail": match.expected_detail,
        "should_answer": match.should_answer,
    }


def _fixture_retrieval_rows(question: RagEvalQuestion) -> list[dict[str, Any]]:
    if not question.gold_source_url:
        return []
    return [
        {
            "source_url": question.gold_source_url,
            "title": question.id,
            "source_number": question.expected_source_number or 1,
            "retrieval_sources": ["fixture"],
        }
    ]


def _normalise_prediction_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _source_rank(rows: Sequence[Any], gold_source_url: str | None) -> int | None:
    if not gold_source_url:
        return None
    for index, row in enumerate(rows, start=1):
        source_url = _row_source_url(row)
        if source_url == gold_source_url:
            return index
    return None


def _row_source_url(row: Any) -> str:
    if isinstance(row, Document):
        return str(row.metadata.get("source_url") or "")
    if isinstance(row, Mapping):
        return str(row.get("source_url") or row.get("gold_source_url") or "")
    return str(getattr(row, "source_url", "") or "")


def _source_number_hit(rows: Sequence[Any], *, gold_source_url: str | None, expected_source_number: int | None) -> bool:
    if expected_source_number is None:
        return True
    row = _matching_source_row(rows, gold_source_url)
    return _row_value(row, "source_number") == expected_source_number if row is not None else False


def _trace_fields_hit(rows: Sequence[Any], *, gold_source_url: str | None, required_fields: Sequence[str]) -> bool:
    if not required_fields:
        return True
    row = _matching_source_row(rows, gold_source_url)
    if row is None:
        return False
    return all(_row_value(row, field) not in (None, "", []) for field in required_fields)


def _matching_source_row(rows: Sequence[Any], gold_source_url: str | None) -> Any | None:
    if not rows:
        return None
    if not gold_source_url:
        return rows[0]
    for row in rows:
        if _row_source_url(row) == gold_source_url:
            return row
    return None


def _row_value(row: Any, field: str) -> Any:
    if isinstance(row, Document):
        return row.metadata.get(field)
    if isinstance(row, Mapping):
        if field in row:
            return row[field]
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(field)
        return None
    return getattr(row, field, None)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _rate(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for value in items if value) / len(items) if items else 0.0

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
