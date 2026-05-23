from datetime import datetime
from types import SimpleNamespace

from app.crawlers import reclassify_existing_documents as reclassifier


def test_classify_document_row_uses_content_specific_domain_over_source_domain() -> None:
    document = SimpleNamespace(
        title="2026-1 성적향상장학금 신청 안내",
        content="성적향상장학금 신청 기간과 제출 서류 안내",
        source_url="https://example.com/notice",
        source_name="university_notices",
        domain="general_notice",
    )

    assert reclassifier._classify_document_row(document=document, source_domain="general_notice") == "scholarship"


def test_metadata_for_chunk_uses_domain_key_only() -> None:
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_index=0,
        title="등록금 납부 안내",
        source_url="https://example.com/tuition",
        source_type="html",
    )

    metadata = reclassifier._metadata_for_chunk(
        chunk=chunk,
        domain="tuition",
        published_at=datetime(2026, 5, 1, 9, 0, 0),
        embedding_model="gemini-embedding-001",
    )

    assert metadata["domain"] == "tuition"
    assert "category" not in metadata
    assert metadata["published_at"] == "2026-05-01T09:00:00"


def test_index_chroma_get_handles_missing_embeddings() -> None:
    indexed = reclassifier._index_chroma_get(
        {
            "ids": ["chunk-1"],
            "documents": ["text"],
            "metadatas": [{"domain": "scholarship"}],
        }
    )

    assert indexed["chunk-1"]["document"] == "text"
    assert indexed["chunk-1"]["metadata"] == {"domain": "scholarship"}
    assert indexed["chunk-1"]["embedding"] is None
