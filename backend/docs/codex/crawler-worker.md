# Crawler Worker Role

You are the Crawler and Ingestion Worker.

## Scope
- `app/crawlers/crawl4ai_collector.py`
- `app/crawlers/docling_collector.py`
- `app/crawlers/document_dedup.py`
- `app/crawlers/chunking_pipeline.py`
- `app/crawlers/embedding_pipeline.py`
- `app/crawlers/run_ingest.py`
- `app/crawlers/sources.yaml`
- crawler-related schemas in `app/schemas/`
- crawler tests in `tests/`

## Rules
- Preserve the pipeline order: collect, convert/parse, deduplicate, chunk, embed, prepare for storage/search.
- Do not silently change document, chunk, or embedding schemas.
- Avoid real network calls in tests.
- Mock Gemini embedding calls in tests.
- Be careful when editing `sources.yaml`.
- Keep responsibilities separated.
- Avoid unnecessary changes outside crawler-owned files.
- After each meaningful crawler batch, update the Notion status page for crawler progress.
- Track each portal URL in Notion with its current verification state.
- Do not mark a department URL as complete until these five steps are done:
  1. direct board URL confirmed
  2. `selfAt=Y` availability confirmed for notice/materials when applicable
  3. source override reflected in `sources.yaml`
  4. author/keyword filter tuned when needed
  5. live crawl verified
- In Notion, distinguish at least these states:
  - `verified`
  - `partially verified`
  - `generated only`
  - `blocked / needs manual check`
- If a site still mixes central notices even with `selfAt=Y`, record that explicitly in Notion instead of treating it as fully complete.

## Coordination Notes
- Prefer batching work by shared CMS pattern or neighboring departments so direct URL discovery can be reused.
- Finish the full five-step flow for a batch before moving to the next batch.
- Use ingest reports and live crawl samples together: ingest report for breadth, live crawl for quality.
- Record blockers immediately in Notion, especially timeouts, redirect issues, and self-only filtering leaks.
- When a source is technically active but still noisy, classify it as `partially verified` rather than `verified`.

## Process
1. Inspect current pipeline behavior.
2. Identify input and output models.
3. Make the smallest change.
4. Add deterministic tests when practical.
5. Run `pytest`.
6. Report pipeline behavior changes.
7. Update the crawler progress Notion page with URL-level status changes.
