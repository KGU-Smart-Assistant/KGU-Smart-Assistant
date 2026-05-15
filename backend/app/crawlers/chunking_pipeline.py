from typing import List

from app.crawlers.document_quality import (
    is_searchable_chunk_text,
    normalize_document_text,
    sanitize_title,
)
from app.schemas import Document, DocumentChunk


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[DocumentChunk]:
    """Split a document into overlapping text chunks."""
    _validate_chunking_options(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    normalized_text = normalize_document_text(document)
    if not normalized_text:
        return []

    chunks: List[DocumentChunk] = []
    start = 0
    text_length = len(normalized_text)
    chunk_index = 0

    while start < text_length:
        end = _find_chunk_end(normalized_text, start=start, chunk_size=chunk_size)
        chunk_text = normalized_text[start:end].strip()

        if chunk_text and is_searchable_chunk_text(chunk_text):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_index}",
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    title=sanitize_title(document.title),
                    source_url=document.source_url,
                    source_type=document.source_type,
                    published_at=document.published_at,
                )
            )
            chunk_index += 1

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[DocumentChunk]:
    """Split multiple documents into a flat chunk list."""
    chunks: List[DocumentChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _find_chunk_end(text: str, *, start: int, chunk_size: int) -> int:
    hard_end = min(start + chunk_size, len(text))
    if hard_end >= len(text):
        return len(text)

    minimum_end = start + max(chunk_size // 4, 1)
    candidate_breaks = []
    for marker in (". ", "? ", "! ", "\n"):
        marker_index = text.rfind(marker, start, hard_end)
        if marker_index >= minimum_end:
            candidate_breaks.append(marker_index + 1)
    if candidate_breaks:
        return max(candidate_breaks)

    whitespace_break = text.rfind(" ", minimum_end, hard_end)
    if whitespace_break > start:
        return whitespace_break
    return hard_end


def _validate_chunking_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
