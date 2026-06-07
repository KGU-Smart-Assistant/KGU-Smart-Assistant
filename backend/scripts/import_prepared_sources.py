from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.crawlers.crawl_markdown import load_sources_config
from app.crawlers.run_pipeline import (
    CrawlPreparePipelineOptions,
    _store_pipeline_result_to_db,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import selected prepared markdown sources into crawler DB tables.")
    parser.add_argument("--config", type=Path, default=Path("app/crawlers/sources.yaml"))
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {source.casefold() for source in args.source}
    sources = [source for source in load_sources_config(args.config) if source.name.casefold() in requested]
    found = {source.name.casefold() for source in sources}
    missing = sorted(requested - found)
    if missing:
        raise SystemExit(f"Unknown source(s): {', '.join(missing)}")

    now = datetime.now(timezone.utc)
    options = CrawlPreparePipelineOptions(
        config_path=args.config,
        prepared_output_dir=args.prepared_dir,
        source_names=tuple(args.source),
        store_db=True,
        run_id=args.run_id,
    )
    report = _store_pipeline_result_to_db(
        options=options,
        sources=sources,
        prepared_output_dir=args.prepared_dir,
        run_id=args.run_id or f"prepared-import-{now.strftime('%Y%m%d%H%M%S')}",
        started_at=now,
        completed_at=now,
        db_session=None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
