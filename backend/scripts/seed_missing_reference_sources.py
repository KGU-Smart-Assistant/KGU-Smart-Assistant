from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.crawler_store import store_ingest_source_result
from app.db.session import SessionLocal
from app.schemas import Document, DocumentChunk


ACADEMIC_SCHEDULE_DIR = Path("data/crawled_markdown/academic-schedule-20260607/final/academic_schedule")
SCHOLARSHIP_FAQ_DIR = Path("data/prepared_markdown/rebuild-20260604-keep-v2/final/scholarship_faq")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    _empty, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter) or {}, body.strip()


def _clean_body(body: str) -> str:
    lines = body.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _chunk_id(source_name: str, source_url: str, index: int, text: str) -> str:
    return _sha256_text(f"{source_name}\n{source_url}\n{index}\n{text}")


def _document_from_metadata(metadata: dict[str, Any], body: str, *, collected_at: datetime) -> Document:
    return Document(
        doc_id=str(metadata["doc_id"]),
        source_type=str(metadata.get("source_type") or "html"),
        source_url=str(metadata["source_url"]),
        title=str(metadata["title"]),
        content=body,
        domain=str(metadata.get("domain") or ""),
        department=str(metadata.get("department") or ""),
        collected_at=collected_at,
        attachment_urls=[str(url) for url in metadata.get("attachment_urls") or []],
    )


def _base_chunk(
    *,
    metadata: dict[str, Any],
    content: str,
    index: int,
    section_title: str,
    section_kind: str,
) -> DocumentChunk:
    source_name = str(metadata["source_name"])
    source_url = str(metadata["source_url"])
    title = str(metadata["title"])
    embedding_text = f"문서제목: {title}\n구간: {section_title}\n\n{content.strip()}"
    body_hash = _sha256_text(content.strip())
    return DocumentChunk(
        chunk_id=_chunk_id(source_name, source_url, index, embedding_text),
        doc_id=str(metadata["doc_id"]),
        chunk_index=index,
        text=embedding_text,
        title=title,
        source_url=source_url,
        source_type=str(metadata.get("source_type") or "html"),
        domain=str(metadata.get("domain") or ""),
        department=str(metadata.get("department") or ""),
        content=content.strip(),
        embedding_text=embedding_text,
        chunk_text_hash=_sha256_text(embedding_text),
        section_title=section_title,
        section_path=[title, section_title],
        section_kind=section_kind,
        source_name=source_name,
        embedding_eligibility="ELIGIBLE",
        chunk_quality_status="PASSED",
        document_quality_status="PASSED",
        content_token_count=len(content.split()),
        embedding_token_count=len(embedding_text.split()),
        prepared_body_hash=body_hash,
        prepared_artifact_hash=_sha256_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + content),
        metadata_json={
            "page_type": metadata.get("page_type"),
            "document_type": metadata.get("document_type"),
            "processing_effective_date": "2026-06-07",
            "quality_gate_version": metadata.get("quality_gate_version") or "manual-reference-seed-v1",
            "review_status": metadata.get("review_status") or "SEEDED_REFERENCE",
            "initial_filter_status": metadata.get("initial_filter_status") or "KEEP",
            "content_char_count": len(content),
            "embedding_char_count": len(embedding_text),
        },
    )


def _academic_schedule_rows(collected_at: datetime) -> tuple[list[Document], list[DocumentChunk]]:
    paths = sorted(ACADEMIC_SCHEDULE_DIR.glob("*.md"))
    if not paths:
        return [], []
    metadata, raw_body = _split_frontmatter(paths[0].read_text(encoding="utf-8-sig"))
    metadata["title"] = "학사일정(학부)"
    metadata["document_type"] = "ACADEMIC_CALENDAR"
    body = _clean_body(raw_body)
    documents = [_document_from_metadata(metadata, body, collected_at=collected_at)]
    month_blocks = re.split(r"(?=^202[0-9]년\s+\d{2}월\s*$)", body, flags=re.MULTILINE)
    chunks: list[DocumentChunk] = []
    for block in month_blocks:
        block = block.strip()
        if not block or "상세일정" not in block:
            continue
        first_line = block.splitlines()[0].strip()
        chunks.append(
            _base_chunk(
                metadata=metadata,
                content=block,
                index=len(chunks),
                section_title=first_line,
                section_kind="TIME_SENSITIVE_ACTION",
            )
        )
    return documents, chunks


def _faq_entries(body: str) -> list[str]:
    body = _clean_body(body)
    footer_pattern = r"처음 페이지이전 10 페이지이전 페이지.*$"
    body = re.sub(footer_pattern, "", body, flags=re.DOTALL).strip()
    marker = "담당부서 : 장학지원팀홈페이지 주소 : https://www.kyonggi.ac.kr/scholarship/index.do"
    entries: list[str] = []
    start = 0
    while True:
        end = body.find(marker, start)
        if end < 0:
            break
        entries.append((body[start : end + len(marker)]).strip())
        start = end + len(marker)
    return [entry for entry in entries if len(entry) >= 20]


def _scholarship_faq_rows(collected_at: datetime) -> tuple[list[Document], list[DocumentChunk]]:
    documents: list[Document] = []
    chunks: list[DocumentChunk] = []
    for path in sorted(SCHOLARSHIP_FAQ_DIR.glob("*.md")):
        metadata, raw_body = _split_frontmatter(path.read_text(encoding="utf-8-sig"))
        metadata["source_type"] = "html"
        body = _clean_body(raw_body)
        documents.append(_document_from_metadata(metadata, body, collected_at=collected_at))
        for entry in _faq_entries(body):
            question = entry.splitlines()[0].strip()[:120] or "FAQ 항목"
            chunks.append(
                _base_chunk(
                    metadata=metadata,
                    content=entry,
                    index=len(chunks),
                    section_title=question,
                    section_kind="FAQ",
                )
            )
    return documents, chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed missing reference source chunks from local markdown.")
    parser.add_argument("--source", action="append", choices=["academic_schedule", "scholarship_faq"], required=True)
    parser.add_argument("--run-id", default="seed-missing-reference-sources-20260607")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc)
    source_builders = {
        "academic_schedule": _academic_schedule_rows,
        "scholarship_faq": _scholarship_faq_rows,
    }
    report: dict[str, Any] = {"run_id": args.run_id, "sources": {}}
    with SessionLocal() as db:
        for source_name in args.source:
            documents, chunks = source_builders[source_name](now)
            if source_name == "academic_schedule":
                source = {
                    "name": source_name,
                    "seed_urls": ["https://www.kyonggi.ac.kr/www/selectTnSchafsSchdulListUS.do?key=5695"],
                    "domain": "academic_calendar",
                    "department": "university",
                }
            else:
                source = {
                    "name": source_name,
                    "seed_urls": ["https://www.kyonggi.ac.kr/scholarship/selectBbsNttList.do?bbsNo=904&key=3081"],
                    "domain": "faq",
                    "department": "scholarship_support",
                }
            result = store_ingest_source_result(
                db,
                run_id=args.run_id,
                source=source,
                documents=documents,
                chunks=chunks,
                source_report={
                    "status": "ok",
                    "status_reason": "Seeded from local markdown reference source.",
                    "raw_documents": len(documents),
                    "documents": len(documents),
                    "exact_duplicates_removed": 0,
                    "version_duplicates_removed": 0,
                    "chunks": len(chunks),
                    "embedded_chunks": 0,
                    "stored_chunks": 0,
                },
                started_at=now,
                completed_at=now,
                run_type="manual_reference_seed",
            )
            report["sources"][source_name] = result
        db.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
