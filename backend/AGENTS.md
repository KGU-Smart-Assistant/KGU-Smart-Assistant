# AGENTS.md

## Repository Purpose
This repository is the backend repository for the project.

The backend currently provides:
- FastAPI web API server
- Gemini-based chat response generation
- Offline crawling and document ingestion pipeline
- Document deduplication, chunking, and embedding pipeline
- Future search and storage integration points

Codex must act as a careful backend engineering assistant. Keep changes small, reviewable, and aligned with the current architecture.

## Tech Stack
- Web API: FastAPI, Uvicorn
- Configuration: `pydantic-settings` with `.env`
- AI provider: Google Gemini via `google-genai`
- Crawling: `crawl4ai`
- Document conversion: `docling`
- HTML / HTTP: `beautifulsoup4`, `requests`, `httpx`
- Testing: `pytest`
- Container: Docker, Docker Compose

Notes:
- `chromadb`, `openai`, `langchain`, and `tiktoken` exist in dependencies but are not part of the current main code path.
- `app/services/search_service.py` is not fully implemented.
- `app/db/base.py` is currently empty.
- Alembic manages PostgreSQL schema migrations.

## Current Architecture
Main online API layer:
- `app/main.py`
- `app/api/v1/router.py`
- `app/api/v1/endpoints/`
- `app/services/`
- `app/schemas/`

Offline crawling and ingestion layer:
- `app/crawlers/crawl4ai_collector.py`
- `app/crawlers/docling_collector.py`
- `app/crawlers/document_dedup.py`
- `app/crawlers/chunking_pipeline.py`
- `app/crawlers/embedding_pipeline.py`
- `app/crawlers/run_ingest.py`
- `app/crawlers/sources.yaml`

Future storage and search layer:
- `app/db/`
- `app/services/search_service.py`

Tests:
- `tests/`

Preserve this separation:
- API endpoints belong in `app/api/v1/endpoints/`
- Business logic belongs in `app/services/`
- Request and response models belong in `app/schemas/`
- Crawling and ingestion logic belongs in `app/crawlers/`
- DB and search implementation belongs in `app/db/` and `app/services/search_service.py`

## Core Rules
- Inspect existing files before editing.
- Make the smallest safe change.
- Do not rewrite large parts of the repository unless explicitly requested.
- Do not modify unrelated files.
- Do not remove existing behavior unless the task clearly requires it.
- Do not add dependencies unless necessary.
- Do not modify `requirements.txt` without explaining why.
- Do not commit secrets, API keys, tokens, credentials, or real `.env` values.
- Do not log Gemini API keys, sensitive prompts, tokens, or private configuration.
- Do not assume unused dependencies are safe to remove.
- Do not implement a full DB or vector store layer unless explicitly requested.
- Do not run destructive commands.

## Collaboration Rules
- Respect file ownership implied by the task scope.
- Avoid touching another teammate's area unless the task requires it.
- If a task spans multiple areas, keep the touched file set minimal and clearly report cross-area changes.
- Prefer additive changes over broad refactors during active collaboration.

## Commands
Run API server:
```bash
uvicorn app.main:app --reload
```

Run with Docker:
```bash
docker compose up --build
```

Run ingestion pipeline:
```bash
python -m app.crawlers.run_ingest
```

Run tests:
```bash
pytest
```

Or:
```bash
python -m pytest
```

Check available Gemini models:
```bash
```

Apply DB migrations:
```bash
alembic upgrade head
```

Create a new migration after model changes:
```bash
alembic revision --autogenerate -m "describe change"
```

## Branch, Commit, and PR Workflow
- Workflow style: `main` / `develop` / `feature/*`
- Create feature work from `develop`
- Open PRs into `develop`
- Do not work directly on `main`
- Current CI is expected to run for PRs targeting `develop`
- Before opening or updating a PR, run `git status --short` and review the staged diff.
- Only stage files that are directly related to the requested task.
- Do not include unrelated local edits in a PR. If unrelated edits already exist, leave them unstaged and mention them in the final response.
- If a PR branch has conflicts with `develop`, resolve the conflicts on the PR branch and push the branch update. Do not merge the PR unless explicitly requested.
- Prefer `git push --force-with-lease` after a rebase; do not use unsafe force pushes.

Branch naming examples:
- `feature/<task-name>`
- `fix/<task-name>`
- `refactor/<task-name>`
- `test/<task-name>`

Commit message convention:
- `feat: add search request validation`
- `fix: handle gemini timeout safely`
- `refactor: simplify chunking pipeline flow`

## Local Files and PR Safety Rules
- Never commit real local configuration files such as `.env`, `.env.*`, `.envrc`, local IDE settings, shell profiles, or machine-specific paths.
- Never commit secrets, API keys, tokens, passwords, cookies, private certificates, or generated credentials.
- Never commit local runtime data such as `app.db`, SQLite journals, database dumps, local vector-store files, crawler output, cache folders, logs, or temporary files unless the task explicitly requires a reviewed fixture.
- Treat generated local folders such as `.tmp/`, `.crawl4ai-data/`, `.pytest_cache/`, `__pycache__/`, `servers/`, `venv/`, and `.venv/` as non-PR files.
- Before committing, check `git diff --cached --name-only` and verify that no personal local files or unrelated files are staged.
- If a tracked file contains local-only data, do not modify or stage it for a PR without explicit approval.
- Use `.env.example` for documented configuration keys and safe placeholder values only. Do not copy real `.env` values into examples, docs, tests, or PR descriptions.
- CI-only secrets must come from GitHub Actions secrets or harmless dummy values. Do not encode real credentials in workflow files.

## API Rules
- Follow the existing FastAPI router structure.
- Keep endpoint logic thin.
- Move reusable logic into `app/services/`.
- Add or update Pydantic schemas in `app/schemas/`.
- Do not change existing API contracts unless explicitly required.

## Gemini Rules
- Use configuration values from `app/core/config.py`.
- Avoid hardcoding API keys, secrets, or new model names unless required.
- Handle provider failures safely.
- Prefer mocks or stubs for Gemini-dependent tests.

## Crawling and Ingestion Rules
- Preserve the pipeline order: collect, deduplicate, chunk, embed, prepare for storage/search.
- Do not silently change document, chunk, or embedding schemas.
- Avoid real network-dependent tests.
- Keep crawling, parsing, deduplication, chunking, and embedding responsibilities separated.

## Search and DB Rules
- Treat `app/services/search_service.py` as the search service boundary.
- Do not assume ChromaDB is already integrated.
- Do not introduce LangChain or OpenAI usage unless explicitly required.
- Use Alembic revisions for schema changes.
- Document any DB or vector-store design decision in the final response.

## Test Rules
- Add or update tests when behavior changes and it is practical.
- Prefer deterministic tests.
- Mock external API and network calls.
- Add regression tests for bug fixes when practical.
- If tests are not added, explain why.

## Completion Checklist
- Summary of changes
- Files changed
- Commands run
- Test results
- API changes, if any
- DB or search changes, if any
- Risks or TODOs

Detailed role guidance lives in `docs/codex/`.
