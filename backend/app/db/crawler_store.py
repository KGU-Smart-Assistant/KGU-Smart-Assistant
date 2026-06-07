from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
import json
import re
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.crawler_identity import (
    attachment_hash,
    canonical_doc_key,
    content_hash,
    index_fingerprint,
    metadata_hash,
    normalize_url,
    sha256_text,
    source_system,
    vector_point_id,
)
from app.models import (
    CrawlerAttachment,
    CrawlerDocument,
    CrawlerDocumentChunk,
    CrawlerDocumentSource,
    CrawlerIngestRun,
    CrawlerSource,
)
from app.schemas import Document, DocumentChunk
from app.services.domain_taxonomy import classify_domain, normalize_domain


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
    run_type: str = "incremental",
) -> dict[str, int | list[str]]:
    """Persist one source ingest result into PostgreSQL using canonical documents."""
    source_name = source["name"]
    _upsert_source(db, source=source, status=source_report["status"], seen_at=completed_at)
    db.flush()

    document_id_map = _upsert_documents(
        db,
        run_id=run_id,
        source=source,
        source_name=source_name,
        source_domain=_source_domain(source),
        documents=documents,
        chunks=chunks,
        source_report=source_report,
        seen_at=completed_at,
    )
    chunk_count = _upsert_chunks(
        db,
        documents=documents,
        chunks=chunks,
        document_id_map=document_id_map,
        source_report=source_report,
        seen_at=completed_at,
    )
    stale_chunk_count = _mark_missing_document_chunks_stale(
        db,
        active_doc_ids=set(document_id_map.values()),
        active_chunk_ids={chunk.chunk_id for chunk in chunks},
        seen_at=completed_at,
    )
    stale_doc_ids = _mark_missing_source_rows_stale(
        db,
        source_name=source_name,
        active_doc_ids=set(document_id_map.values()),
        seen_at=completed_at,
    )
    _insert_ingest_run(
        db,
        run_id=run_id,
        source_name=source_name,
        source_report=source_report,
        started_at=started_at,
        completed_at=completed_at,
        run_type=run_type,
    )
    db.commit()
    return {
        "documents": len(documents),
        "chunks": chunk_count,
        "stale_documents": len(stale_doc_ids),
        "stale_chunks": stale_chunk_count,
        "stale_doc_ids": stale_doc_ids,
    }


def _upsert_source(db: Session, *, source: dict[str, Any], status: str, seen_at: datetime) -> None:
    row = db.get(CrawlerSource, source["name"])
    seed_urls_json = json.dumps(source.get("seed_urls", []), ensure_ascii=False)
    if row is None:
        db.add(
            CrawlerSource(
                name=source["name"],
                domain=_source_domain(source),
                department=source.get("department"),
                seed_urls_json=seed_urls_json,
                status=status,
                last_seen_at=seen_at,
                last_crawled_at=seen_at,
            )
        )
        return
    row.domain = _source_domain(source)
    row.department = source.get("department")
    row.seed_urls_json = seed_urls_json
    row.status = status
    row.last_seen_at = seen_at
    row.last_crawled_at = seen_at


def _upsert_documents(
    db: Session,
    *,
    run_id: str,
    source: dict[str, Any],
    source_name: str,
    source_domain: str | None,
    documents: list[Document],
    chunks: list[DocumentChunk],
    source_report: dict[str, Any],
    seen_at: datetime,
) -> dict[str, str]:
    chunks_by_doc_id: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        chunks_by_doc_id.setdefault(chunk.doc_id, []).append(chunk)

    system = source_system(source)
    rebuild_id = _rebuild_id(run_id)
    document_id_map: dict[str, str] = {}
    for document in documents:
        domain = _document_domain(document=document, source_name=source_name, source_domain=source_domain)
        department = document.department or source.get("department")
        normalized_source_url = normalize_url(document.source_url)
        doc_content_hash = content_hash(document.title, document.content)
        canonical_key = canonical_doc_key(
            document,
            system=system,
            domain=domain,
            department=department,
        )
        target_fingerprint = index_fingerprint(content_hash_value=doc_content_hash)
        row = db.execute(
            select(CrawlerDocument).where(CrawlerDocument.canonical_doc_key == canonical_key)
        ).scalar_one_or_none()
        doc_id = row.doc_id if row is not None else document.doc_id
        document_id_map[document.doc_id] = doc_id
        doc_chunks = chunks_by_doc_id.get(document.doc_id, [])
        index_status = _document_index_status(chunks=doc_chunks, source_report=source_report)
        vector_metadata_status = "not_applicable" if index_status == "skipped" else "pending"
        current_index_fingerprint = row.current_index_fingerprint if row is not None else None
        if source_report.get("stored_chunks", 0) > 0 and doc_chunks:
            index_status = "indexed"
            vector_metadata_status = "synced"
            current_index_fingerprint = target_fingerprint
        elif row is not None and row.content_hash != doc_content_hash and doc_chunks:
            index_status = "chunked"
        document_status = "active"
        if row is not None and row.content_hash != doc_content_hash:
            document_status = "updated"

        values = {
            "source_name": source_name,
            "source_url": normalized_source_url,
            "title": document.title,
            "content": document.content,
            "content_hash": doc_content_hash,
            "source_type": document.source_type,
            "doc_type": _classify_doc_type(document),
            "domain": domain,
            "department": department,
            "author_department": document.author_department,
            "published_at": document.published_at,
            "collected_at": document.collected_at,
            "last_seen_at": seen_at,
            "status": document_status,
            "canonical_doc_key": canonical_key,
            "canonical_source_url": _choose_canonical_source_url(
                existing=row.canonical_source_url if row else None,
                incoming=normalized_source_url,
            ),
            "validity_status": row.validity_status if row is not None else "unknown",
            "index_status": index_status,
            "vector_metadata_status": vector_metadata_status,
            "skip_reason": None,
            "target_index_fingerprint": target_fingerprint,
            "current_index_fingerprint": current_index_fingerprint,
            "rebuild_id": rebuild_id,
        }
        if row is None:
            db.add(CrawlerDocument(doc_id=doc_id, **values))
        else:
            for key, value in values.items():
                setattr(row, key, value)
        db.flush()

        _upsert_document_source(
            db,
            doc_id=doc_id,
            source_name=source_name,
            source_url=normalized_source_url,
            domain=domain,
            department=department,
            run_id=run_id,
            rebuild_id=rebuild_id,
            seen_at=seen_at,
        )
        _upsert_attachments(db, doc_id=doc_id, document=document, seen_at=seen_at)
        db.flush()
        _refresh_document_hashes(db, doc_id=doc_id)
    return document_id_map


def _upsert_document_source(
    db: Session,
    *,
    doc_id: str,
    source_name: str,
    source_url: str,
    domain: str | None,
    department: str | None,
    run_id: str,
    rebuild_id: str,
    seen_at: datetime,
) -> None:
    row = db.execute(
        select(CrawlerDocumentSource).where(
            CrawlerDocumentSource.doc_id == doc_id,
            CrawlerDocumentSource.source_name == source_name,
            CrawlerDocumentSource.source_url == source_url,
        )
    ).scalar_one_or_none()
    values = {
        "domain": domain,
        "department": department,
        "last_seen_at": seen_at,
        "last_seen_run_id": run_id,
        "status": "active",
        "rebuild_id": rebuild_id,
    }
    if row is None:
        db.add(
            CrawlerDocumentSource(
                doc_id=doc_id,
                source_name=source_name,
                source_url=source_url,
                first_seen_at=seen_at,
                **values,
            )
        )
        return
    for key, value in values.items():
        setattr(row, key, value)


def _upsert_attachments(db: Session, *, doc_id: str, document: Document, seen_at: datetime) -> None:
    metadata_by_url = getattr(document, "attachment_metadata", {}) or {}
    for attachment_url in document.attachment_urls:
        normalized_attachment_url = normalize_url(attachment_url)
        attachment_metadata = metadata_by_url.get(attachment_url, {}) or metadata_by_url.get(normalized_attachment_url, {}) or {}
        filename = _clean_attachment_filename(attachment_metadata.get("filename"))
        file_type = _clean_file_type(attachment_metadata.get("file_type"))
        row = db.execute(
            select(CrawlerAttachment).where(
                CrawlerAttachment.doc_id == doc_id,
                CrawlerAttachment.attachment_url == normalized_attachment_url,
            )
        ).scalar_one_or_none()
        values = {
            "filename": filename or _filename_from_url(normalized_attachment_url),
            "file_type": file_type or _file_type_from_url(normalized_attachment_url),
            "extraction_status": "discovered",
            "error_reason": None,
            "last_seen_at": seen_at,
        }
        if row is None:
            db.add(
                CrawlerAttachment(
                    doc_id=doc_id,
                    attachment_url=normalized_attachment_url,
                    **values,
                )
            )
            continue
        for key, value in values.items():
            setattr(row, key, value)


def _upsert_chunks(
    db: Session,
    *,
    documents: list[Document],
    chunks: list[DocumentChunk],
    document_id_map: dict[str, str],
    source_report: dict[str, Any],
    seen_at: datetime,
) -> int:
    documents_by_id = {document_id_map.get(document.doc_id, document.doc_id): document for document in documents}
    for chunk in chunks:
        stored_doc_id = document_id_map.get(chunk.doc_id, chunk.doc_id)
        document = documents_by_id.get(stored_doc_id)
        document_content_hash = (
            content_hash(document.title, document.content) if document is not None else _hash_text(chunk.text)
        )
        embedding_text = chunk.embedding_text or chunk.text
        chunk_text_hash = chunk.chunk_text_hash or sha256_text(embedding_text)
        fingerprint = index_fingerprint(content_hash_value=chunk_text_hash)
        vector_id = vector_point_id(
            doc_id=stored_doc_id,
            index_fingerprint_value=fingerprint,
            chunk_index=chunk.chunk_index,
        )
        row = db.get(CrawlerDocumentChunk, chunk.chunk_id)
        chunk_status = "active"
        if row is not None and (
            row.content_hash != document_content_hash
            or row.chunk_text_hash != chunk_text_hash
            or row.embedding_text != embedding_text
            or row.content != chunk.content
        ):
            chunk_status = "updated"
        values = {
            "doc_id": stored_doc_id,
            "chunk_index": chunk.chunk_index,
            "text": embedding_text,
            "content_hash": document_content_hash,
            "content": chunk.content,
            "embedding_text": embedding_text,
            "chunk_text_hash": chunk_text_hash,
            "title": chunk.title,
            "source_url": normalize_url(chunk.source_url),
            "source_type": chunk.source_type,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "section_kind": chunk.section_kind,
            "embedding_eligibility": chunk.embedding_eligibility,
            "chunk_quality_status": chunk.chunk_quality_status,
            "document_quality_status": chunk.document_quality_status,
            "content_token_count": chunk.content_token_count,
            "embedding_token_count": chunk.embedding_token_count,
            "valid_until": chunk.valid_until,
            "chunk_valid_until": chunk.chunk_valid_until,
            "domain": chunk.domain,
            "department": chunk.department,
            "source_name": chunk.source_name,
            "prepared_body_hash": chunk.prepared_body_hash,
            "prepared_artifact_hash": chunk.prepared_artifact_hash,
            "status": chunk_status,
            "last_seen_at": seen_at,
            "error_reason": None,
            "embedded_at": seen_at if source_report.get("stored_chunks", 0) > 0 else None,
            "index_fingerprint": fingerprint,
            "vector_point_id": vector_id,
            "metadata_json": chunk.metadata_json,
        }
        if row is None:
            db.add(CrawlerDocumentChunk(chunk_id=chunk.chunk_id, **values))
            continue
        for key, value in values.items():
            setattr(row, key, value)
    return len(chunks)


def _mark_missing_source_rows_stale(
    db: Session,
    *,
    source_name: str,
    active_doc_ids: set[str],
    seen_at: datetime,
) -> list[str]:
    stale_doc_ids: list[str] = []
    source_rows = db.execute(
        select(CrawlerDocumentSource).where(
            CrawlerDocumentSource.source_name == source_name,
            CrawlerDocumentSource.status == "active",
        )
    ).scalars()
    for source_row in source_rows:
        if source_row.doc_id in active_doc_ids:
            continue
        source_row.status = "stale"
        source_row.last_seen_at = seen_at
        if not _document_has_active_source(db, source_row.doc_id):
            document = db.get(CrawlerDocument, source_row.doc_id)
            if document is not None:
                document.status = "stale"
                document.last_seen_at = seen_at
                stale_doc_ids.append(document.doc_id)
                for chunk in db.execute(
                    select(CrawlerDocumentChunk).where(CrawlerDocumentChunk.doc_id == document.doc_id)
                ).scalars():
                    chunk.status = "stale"
                    chunk.last_seen_at = seen_at
    return stale_doc_ids


def _mark_missing_document_chunks_stale(
    db: Session,
    *,
    active_doc_ids: set[str],
    active_chunk_ids: set[str],
    seen_at: datetime,
) -> int:
    if not active_doc_ids:
        return 0
    stale_count = 0
    rows = db.execute(
        select(CrawlerDocumentChunk).where(
            CrawlerDocumentChunk.doc_id.in_(active_doc_ids),
            CrawlerDocumentChunk.status.in_(("active", "updated")),
        )
    ).scalars()
    for chunk in rows:
        if chunk.chunk_id in active_chunk_ids:
            continue
        chunk.status = "stale"
        chunk.last_seen_at = seen_at
        stale_count += 1
    return stale_count


def _insert_ingest_run(
    db: Session,
    *,
    run_id: str,
    source_name: str,
    source_report: dict[str, Any],
    started_at: datetime,
    completed_at: datetime,
    run_type: str,
) -> None:
    db.add(
        CrawlerIngestRun(
            run_id=run_id,
            run_type=run_type,
            rebuild_id=_rebuild_id(run_id) if run_type == "bootstrap" else None,
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


def _refresh_document_hashes(db: Session, *, doc_id: str) -> None:
    document = db.get(CrawlerDocument, doc_id)
    if document is None:
        return
    sources = list(
        db.execute(
            select(CrawlerDocumentSource).where(CrawlerDocumentSource.doc_id == doc_id)
        ).scalars()
    )
    attachments = list(
        db.execute(
            select(CrawlerAttachment).where(CrawlerAttachment.doc_id == doc_id)
        ).scalars()
    )
    new_metadata_hash = metadata_hash(
        domains=[source.domain for source in sources],
        departments=[source.department for source in sources],
        source_urls=[source.source_url for source in sources],
    )
    if document.metadata_hash and document.metadata_hash != new_metadata_hash and document.current_index_fingerprint:
        document.vector_metadata_status = "needs_sync"
    document.metadata_hash = new_metadata_hash
    document.attachment_hash = attachment_hash(
        {
            "attachment_url": attachment.attachment_url,
            "filename": attachment.filename,
            "file_type": attachment.file_type,
        }
        for attachment in attachments
    )


def _document_has_active_source(db: Session, doc_id: str) -> bool:
    return (
        db.execute(
            select(CrawlerDocumentSource.id).where(
                CrawlerDocumentSource.doc_id == doc_id,
                CrawlerDocumentSource.status == "active",
            )
        ).first()
        is not None
    )


def _document_index_status(*, chunks: list[DocumentChunk], source_report: dict[str, Any]) -> str:
    if source_report.get("documents", 0) <= 0:
        return "skipped"
    if not chunks:
        return "failed"
    if source_report.get("stored_chunks", 0) > 0:
        return "indexed"
    return "chunked"


def _choose_canonical_source_url(*, existing: str | None, incoming: str) -> str:
    if existing and _looks_like_detail_url(existing):
        return existing
    if _looks_like_detail_url(incoming):
        return incoming
    return existing or incoming


def _looks_like_detail_url(url: str) -> bool:
    lowered = url.casefold()
    return "selectbbsnttview.do" in lowered or "contents.do" in lowered or "/view/" in lowered


def _rebuild_id(run_id: str) -> str:
    return f"rebuild-{run_id}"


def _classify_doc_type(document: Document) -> str:
    domain = _document_domain(document=document, source_name="")
    if domain in {"academic_calendar", "course_registration"}:
        return "calendar"
    if domain == "faq":
        return "faq"
    if document.source_type in {"pdf", "docx", "hwp", "hwpx", "zip", "file"}:
        return "file"
    if domain:
        return domain
    return document.source_type


def _source_domain(source: dict[str, Any]) -> str | None:
    return normalize_domain(source.get("domain")) or normalize_domain(source.get("category"))


def _document_domain(*, document: Document, source_name: str, source_domain: str | None = None) -> str:
    return classify_domain(
        title=document.title,
        content=document.content,
        source_url=document.source_url,
        source_name=source_name,
        source_domain=getattr(document, "domain", None) or source_domain,
        legacy_category=getattr(document, "category", None),
    ).domain


def _hash_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _filename_from_url(url: str) -> str | None:
    path = urlparse(url).path
    return _clean_attachment_filename(unquote(PurePosixPath(path).name))


def _file_type_from_url(url: str) -> str | None:
    filename = _filename_from_url(url)
    if not filename or "." not in filename:
        return None
    return _clean_file_type(filename.rsplit(".", 1)[-1])


def _clean_attachment_filename(value: str | None) -> str | None:
    if not value:
        return None
    filename = re.sub(r"\s+", " ", value).strip().strip("\"'")
    if not filename:
        return None
    return filename[:300]


def _clean_file_type(value: str | None) -> str | None:
    if not value:
        return None
    file_type = re.sub(r"[^0-9a-zA-Z]+", "", value).lower()
    return file_type[:32] or None
