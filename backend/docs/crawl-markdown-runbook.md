# Crawl Markdown Runbook

Use the Crawl4AI Markdown crawler to create reviewable Markdown input for `prepare_markdown`.

```powershell
cd backend
pip install -r requirements-crawlers.txt
python -m playwright install chromium
python -m app.crawlers.crawl_markdown `
  --config app/crawlers/sources.example.yaml `
  --output-dir data/crawled_markdown/run-local `
  --force
```

Outputs:

- `final/**.md`: frontmatter Markdown files.
- `manifest/crawl_classification_seed.csv`: seed CSV with the columns expected by `prepare_markdown`.
- `manifest/crawl_report.json`: crawl counts and failures.

This crawler does not write PostgreSQL, call embedding APIs, or write Chroma. Those steps remain downstream.
