from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
import hashlib
import json
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    CrawlerAttachment,
    CrawlerDocument,
    CrawlerDocumentChunk,
    CrawlerIngestRun,
    CrawlerSource,
)
from app.schemas import Document, DocumentChunk


def store_ingest_source_result(
    db: Session,
    *,
    run_id: str,
    source: dict[str, Any],
    documents: list[Document],
    chunks: list[DocumentChunk],
    source_report: dict[str, Any],
    started_at: datetime,
    completed_at: datetime,
) -> dict[str, int]:
    """Persist one source ingest result into PostgreSQL."""
    source_name = source["name"]
    _upsert_source(
        db,
        source=source,
        status=source_report["status"],
        seen_at=completed_at,
    )
    # Ensure FK parents are visible before dependent rows are flushed.
    db.flush()
    document_count = _upsert_documents(
        db,
        source_name=source_name,
        documents=documents,
        seen_at=completed_at,
    )
    chunk_count = _upsert_chunks(db, chunks=chunks, seen_at=completed_at)
    _insert_ingest_run(
        db,
        run_id=run_id,
        source_name=source_name,
        source_report=source_report,
        started_at=started_at,
        completed_at=completed_at,
    )
    db.commit()
    return {"documents": document_count, "chunks": chunk_count}


def _upsert_source(
    db: Session,
    *,
    source: dict[str, Any],
    status: str,
    seen_at: datetime,
) -> None:
    row = db.get(CrawlerSource, source["name"])
    seed_urls_json = json.dumps(source.get("seed_urls", []), ensure_ascii=False)
    if row is None:
        db.add(
            CrawlerSource(
                name=source["name"],
                category=source.get("category"),
                department=source.get("department"),
                seed_urls_json=seed_urls_json,
                status=status,
                last_seen_at=seen_at,
                last_crawled_at=seen_at,
            )
        )
        return

    row.category = source.get("category")
    row.department = source.get("department")
    row.seed_urls_json = seed_urls_json
    row.status = status
    row.last_seen_at = seen_at
    row.last_crawled_at = seen_at


def _upsert_documents(
    db: Session,
    *,
    source_name: str,
    documents: list[Document],
    seen_at: datetime,
) -> int:
    for document in documents:
        content_hash = _hash_text(document.content)
        row = db.get(CrawlerDocument, document.doc_id)
        status = "active"
        if row is not None and row.content_hash != content_hash:
            status = "updated"

        values = {
            "source_name": source_name,
            "source_url": document.source_url,
            "title": document.title,
            "content": document.content,
            "content_hash": content_hash,
            "source_type": document.source_type,
            "doc_type": _classify_doc_type(document),
            "category": document.category,
            "department": document.department,
            "author_department": document.author_department,
            "published_at": document.published_at,
            "collected_at": document.collected_at,
            "last_seen_at": seen_at,
            "status": status,
        }
        if row is None:
            db.add(CrawlerDocument(doc_id=document.doc_id, **values))
        else:
            for key, value in values.items():
                setattr(row, key, value)

        # Ensure the parent row exists before attachment/chunk FK checks run.
        db.flush()
        _upsert_attachments(db, document=document, seen_at=seen_at)
    return len(documents)


def _upsert_attachments(
    db: Session,
    *,
    document: Document,
    seen_at: datetime,
) -> None:
    for attachment_url in document.attachment_urls:
        row = db.execute(
            select(CrawlerAttachment).where(
                CrawlerAttachment.doc_id == document.doc_id,
                CrawlerAttachment.attachment_url == attachment_url,
            )
        ).scalar_one_or_none()
        values = {
            "filename": _filename_from_url(attachment_url),
            "file_type": _file_type_from_url(attachment_url),
            "extraction_status": "discovered",
            "error_reason": None,
            "last_seen_at": seen_at,
        }
        if row is None:
            db.add(
                CrawlerAttachment(
                    doc_id=document.doc_id,
                    attachment_url=attachment_url,
                    **values,
                )
            )
        else:
            for key, value in values.items():
                setattr(row, key, value)


def _upsert_chunks(
    db: Session,
    *,
    chunks: list[DocumentChunk],
    seen_at: datetime,
) -> int:
    for chunk in chunks:
        content_hash = _hash_text(chunk.text)
        row = db.get(CrawlerDocumentChunk, chunk.chunk_id)
        status = "active"
        if row is not None and row.content_hash != content_hash:
            status = "updated"

        values = {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "content_hash": content_hash,
            "title": chunk.title,
            "source_url": chunk.source_url,
            "source_type": chunk.source_type,
            "status": status,
            "last_seen_at": seen_at,
        }
        if row is None:
            db.add(CrawlerDocumentChunk(chunk_id=chunk.chunk_id, **values))
        else:
            for key, value in values.items():
                setattr(row, key, value)
    return len(chunks)


def _insert_ingest_run(
    db: Session,
    *,
    run_id: str,
    source_name: str,
    source_report: dict[str, Any],
    started_at: datetime,
    completed_at: datetime,
) -> None:
    db.add(
        CrawlerIngestRun(
            run_id=run_id,
            source_name=source_name,
            status=source_report["status"],
            status_reason=source_report["status_reason"],
            raw_documents=source_report["raw_documents"],
            documents=source_report["documents"],
            exact_duplicates_removed=source_report["exact_duplicates_removed"],
            version_duplicates_removed=source_report["version_duplicates_removed"],
            chunks=source_report["chunks"],
            embedded_chunks=source_report["embedded_chunks"],
            stored_chunks=source_report["stored_chunks"],
            started_at=started_at,
            completed_at=completed_at,
        )
    )


def _classify_doc_type(document: Document) -> str:
    if document.category == "academic_schedule":
        return "calendar"
    if document.category == "faq":
        return "faq"
    if document.source_type in {"pdf", "docx", "hwp", "hwpx", "zip", "file"}:
        return "file"
    if document.category:
        return document.category
    return document.source_type


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _filename_from_url(url: str) -> str | None:
    path = urlparse(url).path
    name = PurePosixPath(path).name
    return name or None


def _file_type_from_url(url: str) -> str | None:
    filename = _filename_from_url(url)
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()
