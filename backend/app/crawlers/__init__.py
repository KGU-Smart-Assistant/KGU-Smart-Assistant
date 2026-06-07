"""Crawler and document preparation utilities."""

from importlib import import_module

__all__ = [
    "crawl_markdown",
    "embed_chunks",
    "embed_text",
    "embed_texts",
    "prepare_markdown",
]


def __getattr__(name: str):
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

    if name == "prepare_markdown":
        return import_module("app.crawlers.prepare_markdown")

    if name == "crawl_markdown":
        return import_module("app.crawlers.crawl_markdown")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
