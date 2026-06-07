from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import get_vector_collection, query_embedded_chunks
from app.main import app
from app.services.langchain_rag_service import answer_with_langchain_rag
from app.services.search_service import search_documents


REPORT_DIR = Path("data/embedding_audit")
REQUIRED_VECTOR_METADATA = (
    "source_name",
    "domain",
    "department",
    "title",
    "source_url",
    "chunk_id",
    "vector_point_id",
)
REQUIRED_RESULT_METADATA = ("source_name", "section_title", "section_kind", "vector_point_id")


@dataclass(frozen=True)
class QualityCase:
    query: str
    domain: str
    expected: str
    note: str = ""


QUALITY_CASES: tuple[QualityCase, ...] = (
    QualityCase("졸업요건 알려줘", "graduation", "contents.do?key=8418"),
    QualityCase("졸업학점 기준 알려줘", "graduation", "contents.do?key=8418"),
    QualityCase("졸업논문 제출 자격 알려줘", "graduation", "contents.do?key=8418"),
    QualityCase("졸업인증제 외국어인증 기준 알려줘", "graduation", "contents.do?key=8418"),
    QualityCase("휴학 신청 어떻게 해?", "academic_status", "contents.do?key=8412"),
    QualityCase("일반휴학 신청기간 알려줘", "academic_status", "contents.do?key=8412"),
    QualityCase("복학 신청 절차 알려줘", "academic_status", "contents.do?key=8413"),
    QualityCase("자퇴 신청 방법 알려줘", "academic_status", "contents.do?key=8489"),
    QualityCase("전공배정은 어디서 봐?", "academic_status", "contents.do?key=8423"),
    QualityCase("편입생 학점인정 알려줘", "academic_status", "contents.do?key=8706"),
    QualityCase("수강신청 안내 알려줘", "course_registration", "contents.do?key=8431"),
    QualityCase("강의시간 안내 알려줘", "course_registration", "contents.do?key=8430"),
    QualityCase("학점이월제 안내 알려줘", "course_registration", "contents.do?key=8432"),
    QualityCase("교내 이러닝 수업 안내 알려줘", "course_registration", "contents.do?key=8433"),
    QualityCase("교외 가상대학 이러닝 안내 알려줘", "course_registration", "contents.do?key=8434"),
    QualityCase("캠퍼스 교차수강 안내 알려줘", "course_registration", "contents.do?key=8435"),
    QualityCase("학점포기 재수강 안내 알려줘", "course_registration", "contents.do?key=8436"),
    QualityCase("계절학기 안내 알려줘", "course_registration", "contents.do?key=8427"),
    QualityCase("학점교류 안내 알려줘", "course_registration", "contents.do?key=8429"),
    QualityCase("복수전공 신청 알려줘", "major_change", "contents.do?key=8414"),
    QualityCase("전과 신청 알려줘", "major_change", "contents.do?key=8415"),
    QualityCase("전공선택유연화 알려줘", "major_change", "contents.do?key=8416"),
    QualityCase("융합전공 안내 알려줘", "major_change", "contents.do?key=9881"),
    QualityCase("교직이수 신청 알려줘", "teaching_certification", "contents.do?key=8491"),
    QualityCase("교직 전공 이수 기준 알려줘", "teaching_certification", "contents.do?key=8492"),
    QualityCase("교육실습 교육봉사 알려줘", "teaching_certification", "contents.do?key=8493"),
    QualityCase("교원자격 무시험검정 알려줘", "teaching_certification", "contents.do?key=8494"),
    QualityCase("장학금 신청 대상 알려줘", "scholarship", "contents.do?key=3067"),
    QualityCase("국가장학금 안내 알려줘", "scholarship", "contents.do?key=3068"),
    QualityCase("교내장학금 종류 알려줘", "scholarship", "contents.do?key=3067"),
    QualityCase("학자금대출 안내 알려줘", "scholarship", "contents.do?key=3071"),
    QualityCase("장학 FAQ 어디서 봐?", "faq", "bbsNo=904"),
    QualityCase("통학버스 정보 알려줘", "student_life", "contents.do?key=5146"),
    QualityCase("스쿨버스 노선 알려줘", "student_life", "contents.do?key=9853"),
    QualityCase("학생식당 메뉴 알려줘", "student_life", "selectTnRstrntMenuListU.do"),
    QualityCase("증명서 발급 안내 알려줘", "document_materials", "contents.do?key=5729"),
    QualityCase("등록금 납부 안내 알려줘", "tuition", "contents.do?key=3262"),
    QualityCase("취업지원 안내 알려줘", "career_support", "contents.do?key=5782"),
    QualityCase("학사일정 알려줘", "academic_calendar", "selectTnSchafsSchdulListUS.do?key=5695"),
    QualityCase("학교 주요전화번호 어디서 봐?", "student_life", "contents.do?key=5155"),
)


def _metadata_complete(mapping: dict[str, Any], required: tuple[str, ...]) -> bool:
    return all(mapping.get(key) for key in required)


def _result_payload(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "score": result.score,
        "title": result.title,
        "source_url": result.source_url,
        "domain": result.domain,
        "department": result.department,
        "source_name": result.source_name,
        "section_title": result.section_title,
        "section_kind": result.section_kind,
        "vector_point_id": result.vector_point_id,
        "metadata_complete": _metadata_complete(result.model_dump(), REQUIRED_RESULT_METADATA),
    }


def _document_payload(document: Any) -> dict[str, Any]:
    metadata = document.metadata
    return {
        "chunk_id": metadata.get("chunk_id"),
        "title": metadata.get("title"),
        "source_url": metadata.get("source_url"),
        "domain": metadata.get("domain"),
        "department": metadata.get("department"),
        "source_name": metadata.get("source_name"),
        "section_title": metadata.get("section_title"),
        "section_kind": metadata.get("section_kind"),
        "vector_point_id": metadata.get("vector_point_id"),
        "metadata_complete": _metadata_complete(metadata, REQUIRED_RESULT_METADATA),
    }


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "source_url": row.get("source_url"),
        "domain": row.get("domain"),
        "department": row.get("department"),
        "source_name": row.get("source_name"),
        "section_title": row.get("section_title"),
        "section_kind": row.get("section_kind"),
        "vector_point_id": row.get("vector_point_id"),
        "metadata_complete": _metadata_complete(row, REQUIRED_RESULT_METADATA),
    }


def _expected_hit(expected: str, rows: list[dict[str, Any]]) -> bool:
    expected_normalized = expected.casefold()
    for row in rows:
        source_url = str(row.get("source_url") or "").casefold()
        title = str(row.get("title") or "").casefold()
        if expected_normalized in source_url or expected_normalized in title:
            return True
    return False


def _inspect_chroma() -> dict[str, Any]:
    collection = get_vector_collection()
    metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    missing_counts: Counter[str] = Counter()
    page_types: Counter[str] = Counter()
    source_names: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    for metadata in metadatas:
        metadata = metadata or {}
        page_types[metadata.get("page_type") or "UNKNOWN"] += 1
        source_names[metadata.get("source_name") or "UNKNOWN"] += 1
        domains[metadata.get("domain") or "UNKNOWN"] += 1
        for key in REQUIRED_VECTOR_METADATA:
            if not metadata.get(key):
                missing_counts[key] += 1
    return {
        "chroma_count": collection.count(),
        "chroma_page_types": dict(page_types),
        "chroma_domains": dict(domains),
        "chroma_source_names_top": dict(source_names.most_common(25)),
        "chroma_metadata_missing_counts": dict(missing_counts),
    }


def _expected_fragment_counts(expected_fragments: list[str]) -> dict[str, int]:
    collection = get_vector_collection()
    metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    counts: dict[str, int] = {}
    for fragment in expected_fragments:
        normalized_fragment = fragment.casefold()
        counts[fragment] = sum(
            1
            for metadata in metadatas
            if normalized_fragment in str((metadata or {}).get("source_url") or "").casefold()
            or normalized_fragment in str((metadata or {}).get("title") or "").casefold()
        )
    return counts


def run_quality_audit(*, top_k: int, include_api: bool, include_chat: bool, report_path: Path) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(app) if include_api or include_chat else None
    expected_counts = _expected_fragment_counts([case.expected for case in QUALITY_CASES])
    report: dict[str, Any] = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "include_api": include_api,
        "include_chat": include_chat,
        **_inspect_chroma(),
        "cases": [],
        "chat_api": [],
    }

    for case in QUALITY_CASES:
        trace_stem = case.domain.replace("/", "_")
        query_embedding = embed_text(case.query)
        direct_rows = query_embedded_chunks(query_embedding=query_embedding, top_k=min(top_k, 5))
        search_results = search_documents(
            query=case.query,
            top_k=top_k,
            rag_domain=case.domain,
            rag_confidence=0.95,
            enable_parent_expansion=True,
            trace_path=str(REPORT_DIR / f"search_trace_quality_{trace_stem}.jsonl"),
        )
        rag_result = answer_with_langchain_rag(
            case.query,
            top_k=top_k,
            rag_domain=case.domain,
            rag_confidence=0.95,
            trace_path=str(REPORT_DIR / f"rag_trace_quality_{trace_stem}.jsonl"),
            answer_fn=lambda prompt: "검색 품질 검증용 RAG 응답입니다. [1]",
        )
        search_rows = [result.model_dump() for result in search_results]
        rag_rows = [document.metadata for document in rag_result.documents]
        api_payload: dict[str, Any] | None = None
        api_status: int | None = None
        if client is not None and include_api:
            response = client.post("/api/v1/search", json={"query": case.query, "top_k": top_k, "domain": case.domain})
            api_status = response.status_code
            api_payload = response.json() if response.content else None

        report["cases"].append(
            {
                **asdict(case),
                "expected_indexed_count": expected_counts.get(case.expected, 0),
                "direct_vector_top": [_row_payload(row) for row in direct_rows[:3]],
                "search_result_count": len(search_results),
                "search_top": _result_payload(search_results[0] if search_results else None),
                "expected_hit_any_search_result": _expected_hit(case.expected, search_rows),
                "search_top_metadata_complete": bool(search_results)
                and _metadata_complete(search_results[0].model_dump(), REQUIRED_RESULT_METADATA),
                "rag_document_count": len(rag_result.documents),
                "rag_confidence": rag_result.confidence,
                "rag_low_confidence": rag_result.low_confidence,
                "rag_top_documents": [_document_payload(document) for document in rag_result.documents[:3]],
                "expected_hit_any_rag_document": _expected_hit(case.expected, rag_rows),
                "rag_top_metadata_complete": bool(rag_result.documents)
                and _metadata_complete(rag_result.documents[0].metadata, REQUIRED_RESULT_METADATA),
                "api_search_status": api_status,
                "api_result_count": len(api_payload.get("results", [])) if isinstance(api_payload, dict) else None,
                "api_top": api_payload.get("results", [None])[0]
                if isinstance(api_payload, dict) and api_payload.get("results")
                else None,
            }
        )

    if include_chat and client is not None:
        for case in QUALITY_CASES[:10]:
            response = client.post("/api/v1/chat/chat", json={"message": case.query})
            payload = response.json() if response.content else {}
            sources = payload.get("sources", []) if isinstance(payload, dict) else []
            report["chat_api"].append(
                {
                    "query": case.query,
                    "status": response.status_code,
                    "route": payload.get("route") if isinstance(payload, dict) else None,
                    "answer_status": payload.get("answer_status") if isinstance(payload, dict) else None,
                    "rag_domain": payload.get("rag_domain") if isinstance(payload, dict) else None,
                    "source_count": len(sources),
                    "document_source_metadata_complete": [
                        _metadata_complete(source, ("chunk_id", *REQUIRED_RESULT_METADATA))
                        for source in sources
                        if source.get("type") == "document"
                    ],
                    "top_source": sources[0] if sources else None,
                    "reply_preview": str(payload.get("reply", ""))[:220] if isinstance(payload, dict) else str(payload)[:220],
                }
            )

    summary = summarize_report(report)
    report["summary"] = summary
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    return {
        "case_count": len(cases),
        "search_expected_hits": sum(1 for row in cases if row["expected_hit_any_search_result"]),
        "rag_expected_hits": sum(1 for row in cases if row["expected_hit_any_rag_document"]),
        "search_nonempty": sum(1 for row in cases if row["search_result_count"] > 0),
        "rag_nonempty": sum(1 for row in cases if row["rag_document_count"] > 0),
        "api_ok": sum(1 for row in cases if row["api_search_status"] in {None, 200}),
        "search_top_metadata_complete": sum(1 for row in cases if row["search_top_metadata_complete"]),
        "rag_top_metadata_complete": sum(1 for row in cases if row["rag_top_metadata_complete"]),
        "metadata_missing_counts": report["chroma_metadata_missing_counts"],
        "failed_search_expected_cases": [
            {
                "query": row["query"],
                "domain": row["domain"],
                "expected": row["expected"],
                "expected_indexed_count": row["expected_indexed_count"],
                "top": row["search_top"],
            }
            for row in cases
            if not row["expected_hit_any_search_result"]
        ],
        "failed_rag_expected_cases": [
            {
                "query": row["query"],
                "domain": row["domain"],
                "expected": row["expected"],
                "expected_indexed_count": row["expected_indexed_count"],
                "top": row["rag_top_documents"][:1],
            }
            for row in cases
            if not row["expected_hit_any_rag_document"]
        ],
        "expected_not_indexed_cases": [
            {"query": row["query"], "domain": row["domain"], "expected": row["expected"]}
            for row in cases
            if row["expected_indexed_count"] == 0
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate embedded search and RAG retrieval quality.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report-path", default=str(REPORT_DIR / "search_rag_quality_latest.json"))
    parser.add_argument("--skip-api", action="store_true", help="Skip /api/v1/search smoke calls.")
    parser.add_argument("--include-chat", action="store_true", help="Also smoke the chat API for the first 10 cases.")
    parser.add_argument("--fail-on-miss", action="store_true", help="Exit non-zero if any expected retrieval hit is missed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_quality_audit(
        top_k=args.top_k,
        include_api=not args.skip_api,
        include_chat=args.include_chat,
        report_path=Path(args.report_path),
    )
    summary = report["summary"]
    print(json.dumps({"report_path": args.report_path, **summary}, ensure_ascii=False, indent=2))
    if args.fail_on_miss and (
        summary["search_expected_hits"] != summary["case_count"]
        or summary["rag_expected_hits"] != summary["case_count"]
        or summary["chroma_count"] <= 0
        or summary["metadata_missing_counts"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
