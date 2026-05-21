from __future__ import annotations

import argparse
from datetime import datetime

from sqlalchemy import select

from app.crawlers.embedding_pipeline import DEFAULT_EMBEDDING_MODEL
from app.crawlers.reclassify_existing_documents import _sync_chroma_rows
from app.db.session import SessionLocal, init_db
from app.models import CrawlerDocument, CrawlerDocumentChunk


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed existing stored crawler chunks into Chroma without recrawling."
    )
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reembed", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    init_db()

    total_seen = 0
    total_upserts = 0
    total_skipped = 0
    offset = 0

    while True:
        with SessionLocal() as db:
            stmt = (
                select(CrawlerDocumentChunk, CrawlerDocument.domain, CrawlerDocument.published_at)
                .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
                .where(CrawlerDocument.status.in_(("active", "updated")))
                .where(CrawlerDocumentChunk.status.in_(("active", "updated")))
                .order_by(CrawlerDocumentChunk.chunk_id)
                .offset(offset)
                .limit(args.batch_size)
            )
            if args.sources:
                stmt = stmt.where(CrawlerDocument.source_name.in_(args.sources))
            if args.limit is not None:
                remaining = max(args.limit - total_seen, 0)
                if remaining <= 0:
                    break
                stmt = stmt.limit(min(args.batch_size, remaining))

            rows = db.execute(stmt).all()
            if not rows:
                break

            chroma_rows: list[tuple[CrawlerDocumentChunk, str, datetime | None]] = [
                (chunk, domain or "", published_at)
                for chunk, domain, published_at in rows
            ]
            stats = _sync_chroma_rows(
                chroma_rows,
                reembed=args.reembed,
                embedding_model=args.embedding_model,
            )
            total_seen += len(rows)
            total_upserts += stats.chroma_upserts
            total_skipped += stats.skipped_missing_text
            offset += len(rows)
            print(
                f"embed_existing_progress chunks_seen={total_seen} "
                f"chroma_upserts={total_upserts} skipped_missing_text={total_skipped}",
                flush=True,
            )

    print(
        f"embed_existing_done chunks_seen={total_seen} "
        f"chroma_upserts={total_upserts} skipped_missing_text={total_skipped} "
        f"reembed={args.reembed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
