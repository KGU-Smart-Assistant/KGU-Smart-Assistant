from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, select

from app.crawlers.chunking_pipeline import chunk_document
from app.db.crawler_store import _hash_text
from app.db.session import SessionLocal, init_db
from app.models import CrawlerDocument, CrawlerDocumentChunk
from app.schemas import Document


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild PostgreSQL chunks from stored documents using current cleanup rules."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    init_db()

    rebuilt_documents = 0
    deleted_chunks = 0
    inserted_chunks = 0

    with SessionLocal() as db:
        stmt = (
            select(CrawlerDocument)
            .where(CrawlerDocument.status == "active")
            .order_by(CrawlerDocument.last_seen_at.desc(), CrawlerDocument.doc_id)
            .limit(args.limit)
        )
        if args.source_name:
            stmt = stmt.where(CrawlerDocument.source_name == args.source_name)

        documents = db.execute(stmt).scalars().all()
        for row in documents:
            document = _to_schema(row)
            chunks = chunk_document(
                document=document,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            if args.dry_run:
                print(
                    f"dry_run doc_id={row.doc_id} source={row.source_name} chunks={len(chunks)}",
                    flush=True,
                )
                rebuilt_documents += 1
                inserted_chunks += len(chunks)
                continue

            old_count = db.query(CrawlerDocumentChunk).filter(
                CrawlerDocumentChunk.doc_id == row.doc_id
            ).count()
            db.execute(
                delete(CrawlerDocumentChunk).where(CrawlerDocumentChunk.doc_id == row.doc_id)
            )
            deleted_chunks += old_count
            for chunk in chunks:
                db.add(
                    CrawlerDocumentChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        content_hash=_hash_text(chunk.text),
                        title=chunk.title,
                        source_url=chunk.source_url,
                        source_type=chunk.source_type,
                        status="active",
                        last_seen_at=row.last_seen_at,
                    )
                )
            rebuilt_documents += 1
            inserted_chunks += len(chunks)
            print(
                f"rebuilt doc_id={row.doc_id} old_chunks={old_count} new_chunks={len(chunks)}",
                flush=True,
            )

        if not args.dry_run:
            db.commit()

    print(
        f"done documents={rebuilt_documents} deleted_chunks={deleted_chunks} "
        f"inserted_chunks={inserted_chunks} dry_run={args.dry_run}",
        flush=True,
    )


def _to_schema(row: CrawlerDocument) -> Document:
    return Document(
        doc_id=row.doc_id,
        source_type=row.source_type,
        source_url=row.source_url,
        title=row.title,
        content=row.content,
        category=row.category,
        department=row.department,
        author_department=row.author_department,
        published_at=row.published_at,
        attachment_urls=[],
        collected_at=row.collected_at,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
