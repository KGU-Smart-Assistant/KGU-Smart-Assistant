# Storage Boundary

This project should treat PostgreSQL as the canonical data store and Chroma as the retrieval index.

## Current State

- Ingest currently crawls documents, deduplicates them, chunks them, stores source/document/chunk/attachment/report rows in PostgreSQL when `--store-db` is provided, optionally embeds chunks, and writes embedded chunks to Chroma only when `--store-vectors` is provided.
- The API search path reads from Chroma through `app/db/vector_store.py` and `app/services/search_service.py`.
- PostgreSQL is already used for structured campus data such as places and contacts.
- Alembic owns PostgreSQL schema migrations. The initial revision creates campus DB tables and crawler storage tables.
- `Base.metadata.create_all()` still exists for legacy local setup paths, but new schema changes should be added through Alembic revisions.

## PostgreSQL Responsibilities

Store data that must be authoritative, auditable, updateable, or relational:

- Source records: source name, category, department, seed URLs, crawl settings, last crawl status, and crawl timestamps.
- Canonical documents: `doc_id`, source URL, title, source type, category, department, author department, published date, collected date, content hash, and normalized content.
- Document lifecycle fields: status such as `active`, `updated`, `skipped`, or `failed`, plus `last_seen_at` for recrawl tracking.
- Document type: `doc_type` values such as `notice`, `faq`, `file`, or `calendar`.
- Attachment metadata: attachment URL, filename, file type, parent `doc_id`, extraction status, and error reason when extraction fails.
- Chunk records if we need auditability: `chunk_id`, `doc_id`, chunk index, text, chunking version, and content hash.
- Ingest reports: per-source document counts, duplicate counts, chunk counts, embedding counts, storage counts, status, and status reason.

PostgreSQL should be the place to answer exact operational questions like "which sources failed?", "when was this notice collected?", "what changed since the last crawl?", and "which document owns this chunk?".

## Chroma Responsibilities

Store data needed for semantic retrieval:

- Chroma ID: `chunk_id`.
- Embedding vector generated from chunk text.
- Retrieval document text: the chunk text used as RAG context.
- Minimal metadata needed to filter and display search results: `doc_id`, `chunk_index`, title, source URL, category, department, embedding model, published date, and optionally source name.

Chroma should not be the canonical store for full crawler output, raw attachments, source configuration, crawl reports, or relational application data. It can duplicate chunk text for fast RAG because the current search API returns context directly from Chroma.

## Practical Rule

- If the data is needed for lifecycle, debugging, deduplication, re-crawling, permissions, or exact lookup, put it in PostgreSQL.
- If the data is needed to find semantically similar chunks, put it in Chroma.
- If both are needed, PostgreSQL owns the canonical row and Chroma stores a retrievable copy keyed by the same `chunk_id` and `doc_id`.

## Implemented First Cut

The current first cut stores these crawler tables:

- `crawler_sources`
- `crawler_documents`
- `crawler_document_chunks`
- `crawler_attachments`
- `crawler_ingest_runs`

`run_ingest.py --store-db` now uses this flow:

```text
crawl -> deduplicate -> chunk -> upsert source/documents/chunks/attachments/report to PostgreSQL -> embed -> upsert vectors to Chroma
```
