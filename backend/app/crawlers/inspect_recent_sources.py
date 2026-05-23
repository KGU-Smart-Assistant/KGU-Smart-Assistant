from __future__ import annotations

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import CrawlerDocument, CrawlerDocumentChunk


SOURCES = [
    "remote_learning_guides",
    "counseling_center_guides",
    "disability_support_guides",
    "dormitory_guides",
    "library_guides",
    "international_student_support_guides",
]


def main() -> None:
    with SessionLocal() as db:
        rows = db.execute(
            select(
                CrawlerDocument.source_name,
                CrawlerDocument.title,
                CrawlerDocument.source_url,
                CrawlerDocumentChunk.chunk_index,
                CrawlerDocumentChunk.text,
            )
            .join(CrawlerDocumentChunk, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
            .where(CrawlerDocument.source_name.in_(SOURCES))
            .where(CrawlerDocument.status.in_(("active", "updated")))
            .where(CrawlerDocumentChunk.status.in_(("active", "updated")))
            .order_by(CrawlerDocument.source_name, CrawlerDocument.title, CrawlerDocumentChunk.chunk_index)
        ).all()

    for source, title, url, index, text in rows:
        snippet = " ".join((text or "").split())[:450]
        print(f"--- {source} | {title} | chunk {index}")
        print(url)
        print(snippet)
    print(f"total_chunks {len(rows)}")


if __name__ == "__main__":
    main()
