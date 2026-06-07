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
        domain="general_notice",
        department="academic_affairs",
        author_department="department office",
        published_at=datetime(2026, 4, 1, 9, 0, 0),
        collected_at=datetime(2026, 4, 2, 9, 0, 0),
        attachment_urls=["https://example.com/files/form.pdf"],
    )


def _document_with_attachment_metadata() -> Document:
    return Document(
        doc_id="doc-1",
        source_type="html",
        source_url="https://example.com/notices/1",
        title="Notice title",
        content="document text",
        domain="document_materials",
        department="academic_affairs",
        published_at=datetime(2026, 4, 1, 9, 0, 0),
        collected_at=datetime(2026, 4, 2, 9, 0, 0),
        attachment_urls=["https://www.kyonggi.ac.kr/www/downloadBbsFile.do?atchmnflNo=981574"],
        attachment_metadata={
            "https://www.kyonggi.ac.kr/www/downloadBbsFile.do?atchmnflNo=981574": {
                "filename": "2026학년도 편입생 오리엔테이션 자료.pdf",
                "file_type": "pdf",
            }
        },
    )


def _chunk(
    *,
    chunk_id: str = "doc-1-chunk-0",
    chunk_index: int = 0,
    text: str = "chunk text",
    chunk_text_hash: str = "a" * 64,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=chunk_index,
        text=text,
        title="Notice title",
        source_url="https://example.com/notices/1",
        content="clean chunk content",
        embedding_text=text,
        chunk_text_hash=chunk_text_hash,
        section_title="본문",
        section_path=["Notice title", "본문"],
        section_kind="STABLE_REFERENCE",
        source_name="alpha_notice",
        embedding_eligibility="ELIGIBLE",
        chunk_quality_status="PASSED",
        document_quality_status="PASSED",
        content_token_count=3,
        embedding_token_count=5,
        domain="general_notice",
        department="academic_affairs",
        prepared_body_hash="b" * 64,
        prepared_artifact_hash="c" * 64,
        metadata_json={"page_type": "LIST_PAGE", "warning_codes": []},
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
            "domain": "general_notice",
            "department": "academic_affairs",
        },
        documents=[_document()],
        chunks=[_chunk()],
        source_report=_source_report(),
        started_at=seen_at,
        completed_at=seen_at,
    )

    assert result["documents"] == 1
    assert result["chunks"] == 1
    assert result["stale_documents"] == 0
    assert result["stale_chunks"] == 0
    assert result["stale_doc_ids"] == []

    source = db.get(CrawlerSource, "alpha_notice")
    assert source.domain == "general_notice"
    assert source.status == "ok"

    document = db.get(CrawlerDocument, "doc-1")
    assert document.doc_type == "general_notice"
    assert document.status == "active"
    assert document.content_hash
    assert document.last_seen_at == seen_at

    chunk = db.get(CrawlerDocumentChunk, "doc-1-chunk-0")
    assert chunk.chunk_index == 0
    assert chunk.text == "chunk text"
    assert chunk.embedding_text == "chunk text"
    assert chunk.content == "clean chunk content"
    assert chunk.chunk_text_hash == "a" * 64
    assert chunk.vector_point_id
    assert chunk.vector_point_id != chunk.chunk_id
    assert chunk.section_title == "본문"
    assert chunk.section_path == ["Notice title", "본문"]
    assert chunk.section_kind == "STABLE_REFERENCE"
    assert chunk.embedding_eligibility == "ELIGIBLE"
    assert chunk.chunk_quality_status == "PASSED"
    assert chunk.document_quality_status == "PASSED"
    assert chunk.content_token_count == 3
    assert chunk.embedding_token_count == 5
    assert chunk.domain == "general_notice"
    assert chunk.department == "academic_affairs"
    assert chunk.source_name == "alpha_notice"
    assert chunk.prepared_body_hash == "b" * 64
    assert chunk.prepared_artifact_hash == "c" * 64
    assert chunk.metadata_json["page_type"] == "LIST_PAGE"
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


def test_store_ingest_source_result_prefers_attachment_metadata_filename() -> None:
    db = _session()
    seen_at = datetime(2026, 5, 5, 12, 0, 0)

    store_ingest_source_result(
        db,
        run_id="run-1",
        source={
            "name": "academic_affairs_materials",
            "seed_urls": ["https://www.kyonggi.ac.kr/www/selectBbsNttList.do"],
            "domain": "document_materials",
            "department": "academic_affairs",
        },
        documents=[_document_with_attachment_metadata()],
        chunks=[_chunk()],
        source_report=_source_report(),
        started_at=seen_at,
        completed_at=seen_at,
    )

    attachment = db.query(CrawlerAttachment).one()
    assert attachment.attachment_url.endswith("atchmnflNo=981574")
    assert attachment.filename == "2026학년도 편입생 오리엔테이션 자료.pdf"
    assert attachment.file_type == "pdf"


def test_store_ingest_source_result_remaps_chunks_after_canonical_document_merge() -> None:
    db = _session()
    seen_at = datetime(2026, 5, 5, 12, 0, 0)
    first_document = Document(
        doc_id="doc-academic",
        source_type="html",
        source_url="https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=5258&nttNo=624075",
        title="Duplicated notice",
        content="same canonical document",
        domain="general_notice",
        department="academic_affairs",
        collected_at=seen_at,
    )
    second_document = Document(
        doc_id="doc-university",
        source_type="html",
        source_url="https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=7520&nttNo=624075",
        title="Duplicated notice",
        content="same canonical document",
        domain="general_notice",
        department="university",
        collected_at=seen_at,
    )
    first_chunk = _chunk(text="first chunk").model_copy(
        update={
            "chunk_id": "chunk-academic",
            "doc_id": "doc-academic",
            "source_url": first_document.source_url,
            "source_name": "academic_affairs_notices",
        }
    )
    second_chunk = _chunk(text="second chunk").model_copy(
        update={
            "chunk_id": "chunk-university",
            "doc_id": "doc-university",
            "source_url": second_document.source_url,
            "source_name": "university_notices",
        }
    )

    store_ingest_source_result(
        db,
        run_id="run-1",
        source={
            "name": "academic_affairs_notices",
            "seed_urls": [first_document.source_url],
            "domain": "general_notice",
            "department": "academic_affairs",
        },
        documents=[first_document],
        chunks=[first_chunk],
        source_report=_source_report(),
        started_at=seen_at,
        completed_at=seen_at,
    )
    store_ingest_source_result(
        db,
        run_id="run-2",
        source={
            "name": "university_notices",
            "seed_urls": [second_document.source_url],
            "domain": "general_notice",
            "department": "university",
        },
        documents=[second_document],
        chunks=[second_chunk],
        source_report=_source_report(),
        started_at=seen_at,
        completed_at=seen_at,
    )

    assert db.query(CrawlerDocument).count() == 1
    assert db.get(CrawlerDocumentChunk, "chunk-university").doc_id == "doc-academic"


def test_store_ingest_source_result_marks_changed_rows_updated() -> None:
    db = _session()
    first_seen_at = datetime(2026, 5, 5, 12, 0, 0)
    second_seen_at = datetime(2026, 5, 5, 13, 0, 0)
    source = {
        "name": "alpha_notice",
        "seed_urls": ["https://example.com/notices"],
        "domain": "general_notice",
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


def test_store_ingest_source_result_marks_missing_rows_stale() -> None:
    db = _session()
    first_seen_at = datetime(2026, 5, 5, 12, 0, 0)
    second_seen_at = datetime(2026, 5, 5, 13, 0, 0)
    source = {
        "name": "alpha_notice",
        "seed_urls": ["https://example.com/notices"],
        "domain": "general_notice",
        "department": "academic_affairs",
    }

    store_ingest_source_result(
        db,
        run_id="run-1",
        source=source,
        documents=[_document()],
        chunks=[_chunk()],
        source_report=_source_report(),
        started_at=first_seen_at,
        completed_at=first_seen_at,
    )
    result = store_ingest_source_result(
        db,
        run_id="run-2",
        source=source,
        documents=[],
        chunks=[],
        source_report={**_source_report(), "documents": 0, "chunks": 0, "embedded_chunks": 0, "stored_chunks": 0},
        started_at=second_seen_at,
        completed_at=second_seen_at,
    )

    assert result["stale_documents"] == 1
    assert result["stale_chunks"] == 1
    assert result["stale_doc_ids"] == ["doc-1"]
    assert db.get(CrawlerDocument, "doc-1").status == "stale"
    assert db.get(CrawlerDocumentChunk, "doc-1-chunk-0").status == "stale"


def test_store_ingest_source_result_marks_missing_document_chunks_stale() -> None:
    db = _session()
    first_seen_at = datetime(2026, 5, 5, 12, 0, 0)
    second_seen_at = datetime(2026, 5, 5, 13, 0, 0)
    source = {
        "name": "alpha_notice",
        "seed_urls": ["https://example.com/notices"],
        "domain": "general_notice",
        "department": "academic_affairs",
    }

    store_ingest_source_result(
        db,
        run_id="run-1",
        source=source,
        documents=[_document()],
        chunks=[
            _chunk(chunk_id="doc-1-old-0", chunk_index=0),
            _chunk(chunk_id="doc-1-old-1", chunk_index=1, text="old second", chunk_text_hash="b" * 64),
        ],
        source_report={**_source_report(), "chunks": 2, "embedded_chunks": 2, "stored_chunks": 2},
        started_at=first_seen_at,
        completed_at=first_seen_at,
    )
    result = store_ingest_source_result(
        db,
        run_id="run-2",
        source=source,
        documents=[_document()],
        chunks=[_chunk(chunk_id="doc-1-new-0", chunk_index=0, text="new only", chunk_text_hash="c" * 64)],
        source_report=_source_report(),
        started_at=second_seen_at,
        completed_at=second_seen_at,
    )

    assert result["chunks"] == 1
    assert result["stale_documents"] == 0
    assert result["stale_chunks"] == 2
    assert db.get(CrawlerDocument, "doc-1").status == "active"
    assert db.get(CrawlerDocumentChunk, "doc-1-new-0").status == "active"
    assert db.get(CrawlerDocumentChunk, "doc-1-old-0").status == "stale"
    assert db.get(CrawlerDocumentChunk, "doc-1-old-1").status == "stale"
