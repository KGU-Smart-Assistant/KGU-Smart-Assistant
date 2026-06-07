# KGU Smart Assistant Backend

## Run

Run backend commands from this directory:

```powershell
docker compose up -d --build
```

Expected repository layout:

```text
KGU SmartAssistant/
  backend/
    docker-compose.yml
  frontend/
    frontend-app/
```

The compose file builds the frontend from `../frontend/frontend-app`.

## Local Runtime Data

Local runs may create:

```text
backend/.tmp/
```

Docker volumes:

```text
postgres_data
chroma_data
frontend_node_modules
frontend_next
```

These are local runtime data and should not be committed.

## Crawl Markdown

The crawler uses Crawl4AI and writes Markdown files that can be reviewed and passed to `prepare_markdown`.

Recommended: run crawling inside the dedicated Docker service so the browser/runtime dependencies stay isolated from the main backend image.

`--max-pages` limits how many LIST pages are traversed for a source, not how many detail pages can be discovered from those lists.

Start the crawler container:

```powershell
docker compose --profile crawler up -d --build crawler
```

Run the crawl + prepare pipeline inside the container:

```powershell
docker compose --profile crawler run --rm crawler `
  python -m app.crawlers.run_pipeline `
  --config app/crawlers/sources.yaml `
  --crawl-output-dir data/crawled_markdown/run-local `
  --prepared-output-dir data/prepared_markdown/run-local `
  --force
```

You can also enter the running container and invoke the crawler manually:

```powershell
docker compose --profile crawler exec crawler sh
```

Direct Python execution still works for local development, but it requires the crawler dependencies in the host environment.

```powershell
python -m app.crawlers.crawl_markdown `
  --config app/crawlers/sources.yaml `
  --output-dir data/crawled_markdown/run-local `
  --force
```

Install optional crawler dependencies before running it locally:

```powershell
pip install -r requirements-crawlers.txt
python -m playwright install chromium
```

## URLs

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Chroma: http://localhost:8001
