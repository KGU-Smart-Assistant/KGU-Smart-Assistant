from datetime import datetime

import pytest

from app.crawlers.chunking_pipeline import chunk_document, chunk_documents
from app.schemas import Document


def _build_document(content: str, doc_id: str = "doc-1") -> Document:
    return Document(
        doc_id=doc_id,
        source_type="html",
        source_url=f"https://example.com/{doc_id}",
        title="Sample Notice",
        content=content,
        category="notice",
        department="academic",
        published_at=None,
        collected_at=datetime(2026, 4, 10, 12, 0, 0),
    )


def test_chunk_document_splits_with_overlap() -> None:
    document = _build_document("A" * 18)

    chunks = chunk_document(document, chunk_size=10, chunk_overlap=2)

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1-chunk-0",
        "doc-1-chunk-1",
    ]
    assert [chunk.text for chunk in chunks] == [
        "AAAAAAAAAA",
        "AAAAAAAAAA",
    ]


def test_chunk_documents_flattens_multiple_documents() -> None:
    documents = [
        _build_document("first document text", doc_id="doc-a"),
        _build_document("second document text", doc_id="doc-b"),
    ]

    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 2
    assert [chunk.doc_id for chunk in chunks] == ["doc-a", "doc-b"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (100, -1),
        (100, 100),
    ],
)
def test_chunk_document_rejects_invalid_options(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    document = _build_document("sample text")

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
