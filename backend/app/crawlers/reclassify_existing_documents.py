from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from sqlalchemy import select

from app.crawlers.embedding_pipeline import DEFAULT_EMBEDDING_MODEL, embed_text
from app.db.session import SessionLocal, init_db
from app.db.vector_store import get_vector_collection
from app.models import CrawlerDocument, CrawlerDocumentChunk, CrawlerSource
from app.schemas import EmbeddedChunk
from app.services.domain_taxonomy import classify_domain, normalize_domain


@dataclass(frozen=True)
class SyncStats:
    documents_seen: int = 0
    documents_updated: int = 0
    chunks_seen: int = 0
    chroma_upserts: int = 0
    skipped_missing_text: int = 0

    def add(self, other: "SyncStats") -> "SyncStats":
        return SyncStats(
            documents_seen=self.documents_seen + other.documents_seen,
            documents_updated=self.documents_updated + other.documents_updated,
            chunks_seen=self.chunks_seen + other.chunks_seen,
            chroma_upserts=self.chroma_upserts + other.chroma_upserts,
            skipped_missing_text=self.skipped_missing_text + other.skipped_missing_text,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reclassify stored crawler documents into the domain taxonomy and sync Chroma metadata."
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Regenerate embeddings while upserting Chroma. Default reuses existing Chroma embeddings and updates metadata only.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    init_db()
    stats = reclassify_existing_documents(
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        reembed=args.reembed,
        embedding_model=args.embedding_model,
    )
    print(
        "domain_reclassify "
        f"documents_seen={stats.documents_seen} "
        f"documents_updated={stats.documents_updated} "
        f"chunks_seen={stats.chunks_seen} "
        f"chroma_upserts={stats.chroma_upserts} "
        f"skipped_missing_text={stats.skipped_missing_text} "
        f"dry_run={args.dry_run} "
        f"reembed={args.reembed}",
        flush=True,
    )


def reclassify_existing_documents(
    *,
    batch_size: int = 100,
    limit: int | None = None,
    dry_run: bool = False,
    reembed: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> SyncStats:
    total = SyncStats()
    offset = 0
    while True:
        with SessionLocal() as db:
            stmt = (
                select(CrawlerDocument, CrawlerSource.domain)
                .join(CrawlerSource, CrawlerSource.name == CrawlerDocument.source_name)
                .where(CrawlerDocument.status.in_(("active", "updated")))
                .order_by(CrawlerDocument.doc_id)
                .offset(offset)
                .limit(batch_size)
            )
            if limit is not None:
                remaining = max(limit - total.documents_seen, 0)
                if remaining <= 0:
                    break
                stmt = stmt.limit(min(batch_size, remaining))

            rows = db.execute(stmt).all()
            if not rows:
                break

            batch_stats = _process_document_batch(
                db=db,
                rows=rows,
                dry_run=dry_run,
                reembed=reembed,
                embedding_model=embedding_model,
            )
            if not dry_run:
                db.commit()
            total = total.add(batch_stats)
            offset += len(rows)
    return total


def _process_document_batch(
    *,
    db,
    rows: Iterable[tuple[CrawlerDocument, str | None]],
    dry_run: bool,
    reembed: bool,
    embedding_model: str,
) -> SyncStats:
    stats = SyncStats()
    chroma_rows: list[tuple[CrawlerDocumentChunk, str, datetime | None]] = []
    for document, source_domain in rows:
        domain = _classify_document_row(document=document, source_domain=source_domain)
        updated = domain != document.domain
        if updated and not dry_run:
            document.domain = domain
        chunks = db.execute(
            select(CrawlerDocumentChunk)
            .where(CrawlerDocumentChunk.doc_id == document.doc_id)
            .where(CrawlerDocumentChunk.status == "active")
            .order_by(CrawlerDocumentChunk.chunk_index)
        ).scalars().all()
        chroma_rows.extend((chunk, domain, document.published_at) for chunk in chunks)
        stats = stats.add(
            SyncStats(
                documents_seen=1,
                documents_updated=1 if updated else 0,
                chunks_seen=len(chunks),
            )
        )

    if not dry_run and chroma_rows:
        chroma_stats = _sync_chroma_rows(
            chroma_rows,
            reembed=reembed,
            embedding_model=embedding_model,
        )
        stats = stats.add(chroma_stats)
    return stats


def _classify_document_row(*, document: CrawlerDocument, source_domain: str | None) -> str:
    return classify_domain(
        title=document.title,
        content=document.content,
        source_url=document.source_url,
        source_name=document.source_name,
        source_domain=normalize_domain(source_domain),
        legacy_category=document.domain,
    ).domain


def _sync_chroma_rows(
    rows: list[tuple[CrawlerDocumentChunk, str, datetime | None]],
    *,
    reembed: bool,
    embedding_model: str,
) -> SyncStats:
    collection = get_vector_collection()
    ids = [chunk.chunk_id for chunk, _domain, _published_at in rows]
    existing = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
    existing_by_id = _index_chroma_get(existing)

    upsert_ids = []
    upsert_embeddings = []
    upsert_documents = []
    upsert_metadatas = []
    skipped_missing_text = 0

    for chunk, domain, published_at in rows:
        current = existing_by_id.get(chunk.chunk_id, {})
        text = chunk.text or current.get("document") or ""
        if not text:
            skipped_missing_text += 1
            continue
        if reembed or current.get("embedding") is None:
            embedding = embed_text(text=text, model=embedding_model)
            model = embedding_model
        else:
            embedding = current["embedding"]
            model = str((current.get("metadata") or {}).get("embedding_model") or embedding_model)

        metadata = _metadata_for_chunk(chunk=chunk, domain=domain, published_at=published_at, embedding_model=model)
        upsert_ids.append(chunk.chunk_id)
        upsert_embeddings.append(embedding)
        upsert_documents.append(text)
        upsert_metadatas.append(metadata)

    if upsert_ids:
        collection.upsert(
            ids=upsert_ids,
            embeddings=upsert_embeddings,
            documents=upsert_documents,
            metadatas=upsert_metadatas,
        )
    return SyncStats(chroma_upserts=len(upsert_ids), skipped_missing_text=skipped_missing_text)


def _index_chroma_get(result: dict) -> dict[str, dict]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []
    indexed = {}
    for index, chunk_id in enumerate(ids):
        indexed[chunk_id] = {
            "document": documents[index] if index < len(documents) else None,
            "metadata": metadatas[index] if index < len(metadatas) else None,
            "embedding": embeddings[index] if index < len(embeddings) else None,
        }
    return indexed


def _metadata_for_chunk(
    *,
    chunk: CrawlerDocumentChunk,
    domain: str,
    published_at: datetime | None,
    embedding_model: str,
) -> dict:
    metadata = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "source_url": chunk.source_url,
        "source_type": chunk.source_type,
        "domain": domain,
        "embedding_model": embedding_model,
    }
    if published_at:
        metadata["published_at"] = published_at.isoformat()
    return metadata


if __name__ == "__main__":
    main()
