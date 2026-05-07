from datetime import datetime

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db.crawler_store import store_ingest_source_result
from app.models import (
    Base,
    CrawlerAttachment,
    CrawlerDocument,
    CrawlerDocumentChunk,
    CrawlerIngestRun,
    CrawlerSource,
)
from app.schemas import Document, DocumentChunk


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _document(*, content: str = "document text") -> Document:
    return Document(
        doc_id="doc-1",
        source_type="html",
        source_url="https://example.com/notices/1",
        title="Notice title",
        content=content,
        category="notice",
        department="academic_affairs",
        author_department="department office",
        published_at=datetime(2026, 4, 1, 9, 0, 0),
        collected_at=datetime(2026, 4, 2, 9, 0, 0),
        attachment_urls=["https://example.com/files/form.pdf"],
    )


def _chunk(*, text: str = "chunk text") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="doc-1-chunk-0",
        doc_id="doc-1",
        chunk_index=0,
        text=text,
        title="Notice title",
        source_url="https://example.com/notices/1",
    )


def _source_report() -> dict:
    return {
        "status": "ok",
        "status_reason": "Source produced usable documents.",
        "raw_documents": 1,
        "documents": 1,
        "exact_duplicates_removed": 0,
        "version_duplicates_removed": 0,
        "chunks": 1,
        "embedded_chunks": 1,
        "stored_chunks": 1,
    }


def test_store_ingest_source_result_persists_crawler_rows() -> None:
    db = _session()
    seen_at = datetime(2026, 5, 5, 12, 0, 0)

    result = store_ingest_source_result(
        db,
        run_id="run-1",
        source={
            "name": "alpha_notice",
            "seed_urls": ["https://example.com/notices"],
            "category": "notice",
            "department": "academic_affairs",
        },
        documents=[_document()],
        chunks=[_chunk()],
        source_report=_source_report(),
        started_at=seen_at,
        completed_at=seen_at,
    )

    assert result == {"documents": 1, "chunks": 1}

    source = db.get(CrawlerSource, "alpha_notice")
    assert source.category == "notice"
    assert source.status == "ok"

    document = db.get(CrawlerDocument, "doc-1")
    assert document.doc_type == "notice"
    assert document.status == "active"
    assert document.content_hash
    assert document.last_seen_at == seen_at

    chunk = db.get(CrawlerDocumentChunk, "doc-1-chunk-0")
    assert chunk.chunk_index == 0
    assert chunk.text == "chunk text"
    assert chunk.source_type == "html"
    assert chunk.content_hash

    attachment = db.query(CrawlerAttachment).one()
    assert attachment.attachment_url == "https://example.com/files/form.pdf"
    assert attachment.filename == "form.pdf"
    assert attachment.file_type == "pdf"
    assert attachment.extraction_status == "discovered"

    run = db.query(CrawlerIngestRun).one()
    assert run.run_id == "run-1"
    assert run.source_name == "alpha_notice"
    assert run.status == "ok"


def test_store_ingest_source_result_marks_changed_rows_updated() -> None:
    db = _session()
    first_seen_at = datetime(2026, 5, 5, 12, 0, 0)
    second_seen_at = datetime(2026, 5, 5, 13, 0, 0)
    source = {
        "name": "alpha_notice",
        "seed_urls": ["https://example.com/notices"],
        "category": "notice",
        "department": "academic_affairs",
    }

    store_ingest_source_result(
        db,
        run_id="run-1",
        source=source,
        documents=[_document(content="old content")],
        chunks=[_chunk(text="old chunk")],
        source_report=_source_report(),
        started_at=first_seen_at,
        completed_at=first_seen_at,
    )
    store_ingest_source_result(
        db,
        run_id="run-2",
        source=source,
        documents=[_document(content="new content")],
        chunks=[_chunk(text="new chunk")],
        source_report=_source_report(),
        started_at=second_seen_at,
        completed_at=second_seen_at,
    )

    document = db.get(CrawlerDocument, "doc-1")
    assert document.content == "new content"
    assert document.status == "updated"
    assert document.last_seen_at == second_seen_at

    chunk = db.get(CrawlerDocumentChunk, "doc-1-chunk-0")
    assert chunk.text == "new chunk"
    assert chunk.status == "updated"
    assert chunk.last_seen_at == second_seen_at
