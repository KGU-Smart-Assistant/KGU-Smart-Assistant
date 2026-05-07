from app.db import vector_store
from app.schemas import Document, EmbeddedChunk


def test_create_client_uses_http_mode(monkeypatch) -> None:
    calls = {}

    class FakeChroma:
        @staticmethod
        def HttpClient(*, host, port):
            calls["host"] = host
            calls["port"] = port
            return "http-client"

    monkeypatch.setattr(vector_store.settings, "vector_store_mode", "http")
    monkeypatch.setattr(vector_store.settings, "vector_store_host", "chroma")
    monkeypatch.setattr(vector_store.settings, "vector_store_port", 8000)
    monkeypatch.setitem(__import__("sys").modules, "chromadb", FakeChroma)

    assert vector_store._create_client() == "http-client"
    assert calls == {"host": "chroma", "port": 8000}


def test_create_client_rejects_unknown_mode(monkeypatch) -> None:
    class FakeChroma:
        pass

    monkeypatch.setattr(vector_store.settings, "vector_store_mode", "bad-mode")
    monkeypatch.setitem(__import__("sys").modules, "chromadb", FakeChroma)

    try:
        vector_store._create_client()
    except ValueError as exc:
        assert str(exc) == "VECTOR_STORE_MODE must be either 'embedded' or 'http'."
    else:
        raise AssertionError("Expected ValueError for unknown vector store mode.")


def test_build_chunk_metadata_includes_retrieval_fields() -> None:
    chunk = EmbeddedChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_index=3,
        text="chunk text",
        title="Notice title",
        source_url="https://example.com/notices/1",
        source_type="pdf",
        published_at="2026-04-01T09:00:00",
        embedding=[0.1, 0.2],
        embedding_model="gemini-embedding-001",
    )

    metadata = vector_store._build_chunk_metadata(
        chunk=chunk,
        category="notice",
        department="academic_affairs",
    )

    assert metadata["doc_id"] == "doc-1"
    assert metadata["chunk_index"] == 3
    assert metadata["source_type"] == "pdf"
    assert metadata["published_at"] == "2026-04-01T09:00:00"
    assert metadata["category"] == "notice"
    assert metadata["department"] == "academic_affairs"


def test_delete_embedded_chunks_for_documents_deletes_by_doc_id(monkeypatch) -> None:
    deleted_filters = []

    class FakeCollection:
        def delete(self, *, where):
            deleted_filters.append(where)

    monkeypatch.setattr(
        vector_store,
        "get_vector_collection",
        lambda **_: FakeCollection(),
    )

    count = vector_store.delete_embedded_chunks_for_documents(
        [
            Document(
                doc_id="doc-2",
                source_type="html",
                source_url="https://example.com/2",
                title="Title 2",
                content="content",
                collected_at="2026-05-06T12:00:00",
            ),
            Document(
                doc_id="doc-1",
                source_type="html",
                source_url="https://example.com/1",
                title="Title 1",
                content="content",
                collected_at="2026-05-06T12:00:00",
            ),
            Document(
                doc_id="doc-1",
                source_type="html",
                source_url="https://example.com/1",
                title="Title 1",
                content="content",
                collected_at="2026-05-06T12:00:00",
            ),
        ]
    )

    assert count == 2
    assert deleted_filters == [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
