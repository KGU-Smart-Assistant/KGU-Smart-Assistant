from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.langchain_rag_service import HybridSearchRetriever
from app.services.retriever_evaluation import (
    SearchEvalCase,
    evaluate_search_retrievers,
    summarize_eval_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline search vs LangChain retriever.")
    parser.add_argument(
        "--eval-set",
        default="data/search_eval_set.jsonl",
        help="JSONL file with query, expected_terms, and optional category.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = _load_eval_cases(Path(args.eval_set))
    retriever = HybridSearchRetriever(top_k=args.top_k)
    results = evaluate_search_retrievers(cases, retriever=retriever, top_k=args.top_k)
    summary = summarize_eval_results(results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for result in results:
        print(
            json.dumps(
                {
                    "query": result.query,
                    "expected_terms": result.expected_terms,
                    "baseline_hit": result.baseline_hit,
                    "langchain_hit": result.langchain_hit,
                    "baseline_titles": result.baseline_titles,
                    "langchain_titles": result.langchain_titles,
                },
                ensure_ascii=False,
            )
        )


def _load_eval_cases(path: Path) -> list[SearchEvalCase]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                SearchEvalCase(
                    query=payload["query"],
                    expected_terms=tuple(payload["expected_terms"]),
                    category=payload.get("category"),
                )
            )
    return cases


if __name__ == "__main__":
    main()
