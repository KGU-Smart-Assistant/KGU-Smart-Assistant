"""Crawler and document collection utilities."""

__all__ = [
    "chunk_document",
    "chunk_documents",
    "Crawl4AICollectorConfig",
    "collect_documents_with_crawl4ai",
    "DocumentDedupResult",
    "DoclingCollectorConfig",
    "embed_chunks",
    "embed_text",
    "embed_texts",
    "collect_documents_with_docling",
    "select_latest_documents",
]


def __getattr__(name: str):
    if name in {"chunk_document", "chunk_documents"}:
        from app.crawlers.chunking_pipeline import chunk_document, chunk_documents

        exports = {
            "chunk_document": chunk_document,
            "chunk_documents": chunk_documents,
        }
        return exports[name]

    if name in {"Crawl4AICollectorConfig", "collect_documents_with_crawl4ai"}:
        from app.crawlers.crawl4ai_collector import (
            Crawl4AICollectorConfig,
            collect_documents_with_crawl4ai,
        )

        exports = {
            "Crawl4AICollectorConfig": Crawl4AICollectorConfig,
            "collect_documents_with_crawl4ai": collect_documents_with_crawl4ai,
        }
        return exports[name]

    if name in {"DoclingCollectorConfig", "collect_documents_with_docling"}:
        from app.crawlers.docling_collector import (
            DoclingCollectorConfig,
            collect_documents_with_docling,
        )

        exports = {
            "DoclingCollectorConfig": DoclingCollectorConfig,
            "collect_documents_with_docling": collect_documents_with_docling,
        }
        return exports[name]

    if name in {"DocumentDedupResult", "select_latest_documents"}:
        from app.crawlers.document_dedup import (
            DocumentDedupResult,
            select_latest_documents,
        )

        exports = {
            "DocumentDedupResult": DocumentDedupResult,
            "select_latest_documents": select_latest_documents,
        }
        return exports[name]

    if name in {"embed_text", "embed_texts", "embed_chunks"}:
        from app.crawlers.embedding_pipeline import (
            embed_chunks,
            embed_text,
            embed_texts,
        )

        exports = {
            "embed_text": embed_text,
            "embed_texts": embed_texts,
            "embed_chunks": embed_chunks,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
