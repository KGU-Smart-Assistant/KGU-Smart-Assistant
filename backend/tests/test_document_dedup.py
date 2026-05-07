from datetime import datetime, timedelta

from app.crawlers.document_dedup import select_latest_documents
from app.schemas import Document


def _build_document(
    *,
    doc_id: str,
    source_url: str,
    title: str,
    content: str,
    source_type: str = "html",
    collected_at: datetime,
    published_at: datetime | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        source_type=source_type,
        source_url=source_url,
        title=title,
        content=content,
        category="notice",
        department="academic",
        published_at=published_at,
        collected_at=collected_at,
    )


def test_select_latest_documents_removes_exact_duplicates() -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    documents = [
        _build_document(
            doc_id="older",
            source_url="https://example.com/notice/1",
            title="Academic Notice",
            content="same content " * 30,
            collected_at=now,
        ),
        _build_document(
            doc_id="newer",
            source_url="https://example.com/notice/2",
            title="Academic Notice",
            content="same content " * 30,
            collected_at=now + timedelta(minutes=1),
        ),
    ]

    result = select_latest_documents(documents)

    assert [document.doc_id for document in result.documents] == ["newer"]
    assert result.exact_duplicates_removed == 1
    assert result.version_duplicates_removed == 0


def test_select_latest_documents_keeps_latest_version_of_same_notice() -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    base_content = "important scholarship information " * 25
    documents = [
        _build_document(
            doc_id="html",
            source_url="https://example.com/notice/view",
            title="Scholarship Notice",
            content=base_content,
            source_type="html",
            collected_at=now,
        ),
        _build_document(
            doc_id="pdf",
            source_url="https://example.com/files/notice.pdf",
            title="Scholarship Notice",
            content=base_content + "updated appendix",
            source_type="pdf",
            collected_at=now + timedelta(minutes=1),
        ),
    ]

    result = select_latest_documents(documents)

    assert [document.doc_id for document in result.documents] == ["html"]
    assert result.exact_duplicates_removed == 0
    assert result.version_duplicates_removed == 1


def test_select_latest_documents_keeps_distinct_notices_with_same_title() -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    documents = [
        _build_document(
            doc_id="first",
            source_url="https://example.com/notice/1",
            title="General Notice",
            content="first notice body " * 20,
            collected_at=now,
        ),
        _build_document(
            doc_id="second",
            source_url="https://example.com/notice/2",
            title="General Notice",
            content="second notice body " * 20,
            collected_at=now + timedelta(minutes=1),
        ),
    ]

    result = select_latest_documents(documents)

    assert {document.doc_id for document in result.documents} == {"first", "second"}
    assert result.exact_duplicates_removed == 0
    assert result.version_duplicates_removed == 0
