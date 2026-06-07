from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.schemas import Document, EmbeddedChunk


def upsert_embedded_chunks(
    chunks: List[EmbeddedChunk],
    *,
    domain: str | None = None,
    department: str | None = None,
    collection_name: str | None = None,
    source_name: str | None = None,
) -> int:
    """Persist embedded chunks into the configured Chroma collection."""
    if not chunks:
        return 0

    collection = get_vector_collection(collection_name=collection_name)
    collection.upsert(
        ids=[_vector_storage_id(chunk) for chunk in chunks],
        embeddings=[chunk.embedding for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            _build_chunk_metadata(
                chunk=chunk,
                domain=domain,
                department=department,
                source_name=source_name,
            )
            for chunk in chunks
        ],
    )
    return len(chunks)


def _vector_storage_id(chunk: EmbeddedChunk) -> str:
    vector_point_id = chunk.metadata_json.get("vector_point_id") if chunk.metadata_json else None
    return str(vector_point_id or chunk.chunk_id)


def delete_embedded_chunks_for_documents(
    documents: List[Document],
    *,
    collection_name: str | None = None,
) -> int:
    """Delete existing vectors for documents before re-ingesting their chunks."""
    if not documents:
        return 0

    collection = get_vector_collection(collection_name=collection_name)
    deleted = 0
    for doc_id in sorted({document.doc_id for document in documents}):
        collection.delete(where={"doc_id": doc_id})
        deleted += 1
    return deleted


def delete_embedded_chunks_for_doc_ids(
    doc_ids: List[str],
    *,
    collection_name: str | None = None,
) -> int:
    """Delete existing vectors for stale document IDs."""
    if not doc_ids:
        return 0

    collection = get_vector_collection(collection_name=collection_name)
    deleted = 0
    for doc_id in sorted(set(doc_ids)):
        collection.delete(where={"doc_id": doc_id})
        deleted += 1
    return deleted


def query_embedded_chunks(
    query_embedding: List[float],
    *,
    top_k: int = 5,
    domain: str | None = None,
    collection_name: str | None = None,
) -> List[Dict[str, Any]]:
    if not query_embedding or top_k <= 0:
        return []

    collection = get_vector_collection(collection_name=collection_name)
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if domain:
        query_kwargs["where"] = {"domain": domain}

    result = collection.query(**query_kwargs)
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    rows: List[Dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        rows.append(
            {
                "chunk_id": metadata.get("chunk_id", chunk_id),
                "doc_id": metadata.get("doc_id", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "text": documents[index] if index < len(documents) else "",
                "title": metadata.get("title", ""),
                "source_url": metadata.get("source_url", ""),
                "source_type": metadata.get("source_type"),
                "distance": distances[index] if index < len(distances) else None,
                "domain": metadata.get("domain"),
                "department": metadata.get("department"),
                "source_name": metadata.get("source_name"),
                "section_title": metadata.get("section_title"),
                "section_kind": metadata.get("section_kind"),
                "vector_point_id": metadata.get("vector_point_id"),
                "published_at": metadata.get("published_at"),
            }
        )
    return rows


def count_embedded_chunks(*, collection_name: str | None = None) -> int:
    collection = get_vector_collection(collection_name=collection_name)
    return collection.count()


def get_vector_collection(*, collection_name: str | None = None):
    client = _create_client()
    return client.get_or_create_collection(
        name=collection_name or settings.vector_store_collection_name,
    )


def _create_client():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Install it before using vector storage."
        ) from exc

    mode = settings.vector_store_mode.lower()
    if mode == "http":
        return chromadb.HttpClient(
            host=settings.vector_store_host,
            port=settings.vector_store_port,
        )
    if mode != "embedded":
        raise ValueError(
            "VECTOR_STORE_MODE must be either 'embedded' or 'http'."
        )

    persist_dir = Path(settings.vector_store_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _build_chunk_metadata(
    *,
    chunk: EmbeddedChunk,
    domain: str | None,
    department: str | None,
    source_name: str | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "source_url": chunk.source_url,
        "source_type": chunk.source_type,
        "embedding_model": chunk.embedding_model,
    }
    if chunk.published_at:
        metadata["published_at"] = chunk.published_at.isoformat()
    effective_domain = chunk.domain or domain
    effective_department = chunk.department or department
    if effective_domain:
        metadata["domain"] = effective_domain
    if effective_department:
        metadata["department"] = effective_department
    if source_name:
        metadata["source_name"] = source_name
    elif chunk.source_name:
        metadata["source_name"] = chunk.source_name
    if chunk.section_title:
        metadata["section_title"] = chunk.section_title
    if chunk.section_kind:
        metadata["section_kind"] = chunk.section_kind
    if chunk.section_path:
        metadata["section_path"] = " > ".join(chunk.section_path)
    if chunk.chunk_text_hash:
        metadata["chunk_text_hash"] = chunk.chunk_text_hash
    if chunk.metadata_json:
        for key in (
            "page_type",
            "document_type",
            "chunk_quality_status",
            "document_quality_status",
            "embedding_eligibility",
        ):
            value = chunk.metadata_json.get(key)
            if value:
                metadata[key] = str(value)
    vector_point_id = chunk.metadata_json.get("vector_point_id") if chunk.metadata_json else None
    if vector_point_id:
        metadata["vector_point_id"] = str(vector_point_id)
    return metadata
