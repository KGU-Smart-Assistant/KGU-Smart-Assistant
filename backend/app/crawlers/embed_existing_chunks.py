from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from app.crawlers.embedding_pipeline import DEFAULT_EMBEDDING_MODEL, embed_text
from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.db.session import SessionLocal, init_db
from app.db.vector_store import get_vector_collection
from app.models import CrawlerDocument, CrawlerDocumentChunk
from app.schemas import EmbeddedChunk


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks already stored in PostgreSQL into Chroma."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--reset-collection", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit <= 0:
        print("limit must be positive", flush=True)
        return

    init_db()
    collection = get_vector_collection()
    if args.reset_collection:
        existing = collection.get(include=[])
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        print(f"reset_collection deleted={len(ids)}", flush=True)
    selected = 0
    embedded = 0
    skipped_existing = 0
    failed = 0

    with SessionLocal() as db:
        stmt = (
            select(
                CrawlerDocumentChunk,
                CrawlerDocument.published_at,
                CrawlerDocument.category,
                CrawlerDocument.department,
            )
            .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
            .where(CrawlerDocumentChunk.status == "active")
            .order_by(CrawlerDocumentChunk.last_seen_at.desc(), CrawlerDocumentChunk.chunk_id)
        )
        if args.source_name:
            stmt = stmt.where(CrawlerDocument.source_name == args.source_name)

        offset = 0
        while selected < args.limit:
            rows = db.execute(stmt.offset(offset).limit(args.batch_size)).all()
            if not rows:
                break
            offset += len(rows)

            existing_ids = _existing_ids(collection, [chunk.chunk_id for chunk, *_ in rows])
            embedded_batch: list[EmbeddedChunk] = []
            metadata_batch: list[dict[str, object]] = []
            for chunk, published_at, category, department in rows:
                if chunk.chunk_id in existing_ids:
                    skipped_existing += 1
                    continue
                if selected >= args.limit:
                    break
                selected += 1
                clean_text = clean_crawled_markdown(
                    chunk.text,
                    source_url=chunk.source_url,
                )
                if not clean_text:
                    skipped_existing += 1
                    continue
                try:
                    embedding = embed_text(clean_text, model=args.model)
                except Exception as exc:
                    failed += 1
                    print(
                        f"stopped selected={selected} embedded={embedded} "
                        f"skipped_existing={skipped_existing} failed={failed} "
                        f"error={exc.__class__.__name__}: {exc}",
                        flush=True,
                    )
                    return
                embedded_batch.append(
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        chunk_index=chunk.chunk_index,
                        text=clean_text,
                        title=chunk.title,
                        source_url=chunk.source_url,
                        source_type=chunk.source_type,
                        published_at=published_at,
                        embedding=embedding,
                        embedding_model=args.model,
                    )
                )
                metadata_batch.append(
                    _metadata(
                        embedded_batch[-1],
                        category=category,
                        department=department,
                    )
                )

            if embedded_batch:
                collection.upsert(
                    ids=[chunk.chunk_id for chunk in embedded_batch],
                    embeddings=[chunk.embedding for chunk in embedded_batch],
                    documents=[chunk.text for chunk in embedded_batch],
                    metadatas=metadata_batch,
                )
                embedded += len(embedded_batch)
                print(
                    f"progress selected={selected} embedded={embedded} "
                    f"skipped_existing={skipped_existing} chroma_count={collection.count()}",
                    flush=True,
                )

    print(
        f"done selected={selected} embedded={embedded} "
        f"skipped_existing={skipped_existing} failed={failed} "
        f"chroma_count={collection.count()}",
        flush=True,
    )


def _existing_ids(collection, chunk_ids: Iterable[str]) -> set[str]:
    ids = list(chunk_ids)
    if not ids:
        return set()
    try:
        result = collection.get(ids=ids, include=[])
    except Exception:
        return set()
    return set(result.get("ids") or [])


def _metadata(
    chunk: EmbeddedChunk,
    *,
    category: str | None,
    department: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
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
    if category:
        metadata["category"] = category
    if department:
        metadata["department"] = department
    return metadata


if __name__ == "__main__":
    main(sys.argv[1:])
