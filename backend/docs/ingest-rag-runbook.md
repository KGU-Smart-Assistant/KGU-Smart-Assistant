# Ingest and RAG Runbook

## Docker Stack

Start the backend, Postgres, and Chroma services:

```bash
docker compose up -d --build
```

Check the services:

```bash
docker compose ps
curl http://localhost:8000/api/v1/health
curl http://localhost:8001/api/v2/heartbeat
```

Apply PostgreSQL migrations:

```bash
docker compose exec backend alembic upgrade head
```

The backend uses Chroma server mode in Docker Compose:

```env
VECTOR_STORE_MODE=http
VECTOR_STORE_HOST=chroma
VECTOR_STORE_PORT=8000
```

The Chroma container uses `chroma.docker.yaml`, listens on port `8000` inside Docker, and persists index data under `/data`.
That directory is backed by the Docker volume `chroma_data`.

## Scoped Ingest

Do not run the full source list as the first validation step.
The full ingest can take a long time because pagination, attachments, ZIP extraction, HWP/HWPX parsing, and OCR can all run in one batch.

Run one source at a time:

```bash
docker compose exec backend python -m app.crawlers.run_ingest \
  --source youth_studies_notice \
  --max-pages 5 \
  --max-pagination-pages 2 \
  --store-vectors
```

Useful options:

```text
--source <name>              Run one named source.
--source-prefix <prefix>     Run sources whose names start with a prefix.
--limit <n>                  Limit the number of matching sources.
--skip-embed                 Crawl, deduplicate, and chunk without embedding.
--max-pages <n>              Override crawler max_pages for this run.
--max-pagination-pages <n>   Override pagination page limit for this run.
--store-db                   Store source, documents, chunks, attachments, and ingest reports in PostgreSQL.
--store-vectors              Store embedded chunks in Chroma.
```

`--store-vectors` requires embedding.
Do not combine it with `--skip-embed`.

Reports are written to:

```text
.tmp/ingest-reports/
```

## Validated Commands

Fast source validation without embedding:

```bash
docker compose exec backend python -m app.crawlers.run_ingest \
  --source youth_studies_notice \
  --skip-embed \
  --max-pages 5 \
  --max-pagination-pages 2
```

Vector storage validation:

```bash
docker compose exec backend python -m app.crawlers.run_ingest \
  --source youth_studies_notice \
  --max-pages 5 \
  --max-pagination-pages 2 \
  --store-vectors
```

PostgreSQL plus Chroma storage validation:

```bash
docker compose exec backend python -m app.crawlers.run_ingest \
  --source youth_studies_notice \
  --max-pages 5 \
  --max-pagination-pages 2 \
  --store-db \
  --store-vectors
```

The storage flow is:

```text
crawl -> deduplicate -> chunk -> PostgreSQL upsert -> embed -> Chroma upsert
```

Check Chroma count from the backend container:

```bash
docker compose exec backend python -c "from app.db.vector_store import count_embedded_chunks; print(count_embedded_chunks())"
```

## Search API

Call vector search through the backend:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"youth studies notice\",\"top_k\":2,\"category\":\"notice\"}"
```

Expected shape:

```json
{
  "query": "youth studies notice",
  "results": [
    {
      "chunk_id": "...",
      "doc_id": "...",
      "score": 0.64,
      "text": "...",
      "title": "...",
      "source_url": "..."
    }
  ]
}
```

## Chat RAG

General chat intent uses vector search before asking Gemini to answer.
Map and phone intents still use the existing DB-backed services.

```bash
curl -X POST http://localhost:8000/api/v1/chat/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"summarize the latest youth studies notice\"}"
```

RAG answers include source URLs.
If retrieved chunks do not contain enough detail, the answer should say what can be verified and point to the source instead of inventing missing details.

## Storage Boundary

Use PostgreSQL as the canonical store for crawled document lifecycle data and Chroma as the semantic retrieval index.
See `docs/storage-boundary.md` for the detailed split.

## Known Slow Source

`dimensional_art_materials` is attachment-heavy and becomes slow when `max_pages` is raised above `1`.

Observed behavior:

```text
--max-pages 1 --max-pagination-pages 1
status=no_content_discovered

--max-pages 3 --max-pagination-pages 1
long-running source, exceeded 5 minutes
```

Treat it as a separate attachment/OCR performance investigation target.

## Local Python Notes

The project currently has an existing virtual environment at:

```text
kGU-chatbot/Scripts/python.exe
```

For local crawl validation, use that interpreter because it contains crawler dependencies such as `crawl4ai` and `docling`.
