from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.crawlers.embedding_pipeline import DEFAULT_EMBEDDING_MODEL, embed_text
from app.db.session import SessionLocal
from app.db.vector_store import upsert_embedded_chunks
from app.models import CrawlerDocument, CrawlerDocumentChunk
from app.schemas import EmbeddedChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed selected crawler DB chunks into Chroma.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _metadata_json(chunk: CrawlerDocumentChunk) -> dict[str, Any]:
    metadata = dict(chunk.metadata_json or {})
    if chunk.vector_point_id:
        metadata["vector_point_id"] = chunk.vector_point_id
    if chunk.embedding_eligibility:
        metadata["embedding_eligibility"] = chunk.embedding_eligibility
    if chunk.chunk_quality_status:
        metadata["chunk_quality_status"] = chunk.chunk_quality_status
    if chunk.document_quality_status:
        metadata["document_quality_status"] = chunk.document_quality_status
    return metadata


def _embedded_chunk(chunk: CrawlerDocumentChunk, *, model: str) -> EmbeddedChunk:
    text = chunk.embedding_text or chunk.text
    return EmbeddedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        chunk_index=int(chunk.chunk_index),
        text=text,
        title=chunk.title,
        source_url=chunk.source_url,
        source_type=chunk.source_type or "html",
        domain=chunk.domain,
        department=chunk.department,
        published_at=None,
        content=chunk.content,
        embedding_text=text,
        chunk_text_hash=chunk.chunk_text_hash,
        section_title=chunk.section_title,
        section_path=list(chunk.section_path or []),
        section_kind=chunk.section_kind,
        source_name=chunk.source_name,
        embedding_eligibility=chunk.embedding_eligibility,
        chunk_quality_status=chunk.chunk_quality_status,
        document_quality_status=chunk.document_quality_status,
        content_token_count=chunk.content_token_count,
        embedding_token_count=chunk.embedding_token_count,
        valid_until=chunk.valid_until,
        chunk_valid_until=chunk.chunk_valid_until,
        prepared_body_hash=chunk.prepared_body_hash,
        prepared_artifact_hash=chunk.prepared_artifact_hash,
        metadata_json=_metadata_json(chunk),
        embedding=embed_text(text=text, model=model),
        embedding_model=model,
    )


def main() -> None:
    args = parse_args()
    source_names = tuple(args.source)
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        stmt = (
            select(CrawlerDocumentChunk)
            .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
            .where(CrawlerDocumentChunk.source_name.in_(source_names))
            .where(CrawlerDocumentChunk.status.in_(("active", "updated")))
            .where(CrawlerDocument.status.in_(("active", "updated")))
            .where(CrawlerDocumentChunk.embedding_eligibility == "ELIGIBLE")
            .where(CrawlerDocumentChunk.chunk_quality_status.in_(("PASSED", "WARNING")))
            .where(CrawlerDocumentChunk.document_quality_status.in_(("PASSED", "WARNING")))
            .order_by(CrawlerDocumentChunk.source_name, CrawlerDocumentChunk.doc_id, CrawlerDocumentChunk.chunk_index)
        )
        if args.limit is not None:
            stmt = stmt.limit(args.limit)
        chunks = list(db.execute(stmt).scalars())
        embedded = [_embedded_chunk(chunk, model=args.model) for chunk in chunks]
        stored = upsert_embedded_chunks(embedded)
        for chunk in chunks:
            chunk.embedded_at = now
            chunk.embedding_model = args.model
            chunk.embedding_version = args.model
            chunk.index_fingerprint = chunk.index_fingerprint
        db.commit()

    print(
        json.dumps(
            {
                "sources": list(source_names),
                "eligible_chunks": len(chunks),
                "upserted_vectors": stored,
                "embedded_at": now.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
