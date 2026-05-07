from typing import List

from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.schemas import Document, DocumentChunk


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[DocumentChunk]:
    """Split a document into overlapping text chunks."""
    _validate_chunking_options(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    normalized_text = _normalize_text(
        clean_crawled_markdown(document.content, source_url=document.source_url)
    )
    if not normalized_text:
        return []

    chunks: List[DocumentChunk] = []
    start = 0
    text_length = len(normalized_text)
    step = chunk_size - chunk_overlap
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = normalized_text[start:end].strip()

        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_index}",
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    title=document.title,
                    source_url=document.source_url,
                    source_type=document.source_type,
                    published_at=document.published_at,
                )
            )
            chunk_index += 1

        if end >= text_length:
            break

        start += step

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


def _validate_chunking_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
