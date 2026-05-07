import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import yaml

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.crawlers import (
    chunk_documents,
    Crawl4AICollectorConfig,
    DoclingCollectorConfig,
    collect_documents_with_crawl4ai,
    select_latest_documents,
)

DEFAULT_INCLUDE_PATTERNS = (
    "notice",
    "bbs",
    "board",
    "academic",
    "scholarship",
    "faq",
    "download",
    "dn.php",
    "contents.do",
    ".pdf",
    ".docx",
)

DEFAULT_EXCLUDE_PATTERNS = (
    "login",
    "logout",
    "signup",
    "search.do",
    "javascript:",
    "mailto:",
)

ATTACHMENT_SOURCE_TYPES = {"pdf", "docx", "hwp", "hwpx", "zip", "file"}


def embed_chunks(*args, **kwargs):
    from app.crawlers import embed_chunks as _embed_chunks

    return _embed_chunks(*args, **kwargs)


def upsert_embedded_chunks(*args, **kwargs):
    from app.db.vector_store import upsert_embedded_chunks as _upsert_embedded_chunks

    return _upsert_embedded_chunks(*args, **kwargs)


def delete_embedded_chunks_for_documents(*args, **kwargs):
    from app.db.vector_store import (
        delete_embedded_chunks_for_documents as _delete_embedded_chunks_for_documents,
    )

    return _delete_embedded_chunks_for_documents(*args, **kwargs)


def store_ingest_source_result(*args, **kwargs):
    from app.db.crawler_store import (
        store_ingest_source_result as _store_ingest_source_result,
    )
    from app.db.session import SessionLocal, init_db

    init_db()
    with SessionLocal() as db:
        return _store_ingest_source_result(db, *args, **kwargs)


def load_sources_config(config_path: Path) -> List[Dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    sources = list(config.get("sources", []))
    sources.extend(
        _expand_generated_sources(
            department_sites=config.get("department_sites", []),
            source_blueprints=config.get("source_blueprints", []),
        )
    )
    return sources


def _expand_generated_sources(
    department_sites: List[Dict[str, Any]],
    source_blueprints: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    expanded_sources: List[Dict[str, Any]] = []

    for site in department_sites:
        slug = site.get("slug")
        base_url = site.get("base_url")
        site_type = site.get("site_type", "portal")
        source_overrides = site.get("source_overrides", {}) or {}
        if not slug or not base_url:
            continue

        for blueprint in source_blueprints:
            applies_to = blueprint.get("applies_to", ["portal"])
            if isinstance(applies_to, str):
                applies_to = [applies_to]
            if site_type not in applies_to:
                continue

            source = {
                key: value
                for key, value in blueprint.items()
                if key not in {"applies_to", "name_suffix", "seed_url_template"}
            }
            source["name"] = f"{slug}_{blueprint['name_suffix']}"
            source["department"] = site.get("department", slug)
            seed_url_template = blueprint.get("seed_url_template")
            if seed_url_template:
                source["seed_urls"] = [seed_url_template.format(base_url=base_url.rstrip("/"))]
            else:
                source["seed_urls"] = source.get("seed_urls") or [base_url]

            override = source_overrides.get(blueprint["name_suffix"], {})
            if override:
                source.update(override)
            expanded_sources.append(source)

    return expanded_sources


def build_crawler_config(source: Dict[str, Any]) -> Crawl4AICollectorConfig:
    seed_urls = source["seed_urls"]
    allowed_path_prefixes = source.get("allowed_path_prefixes")
    if allowed_path_prefixes is None:
        allowed_path_prefixes = _derive_allowed_path_prefixes(seed_urls)

    return Crawl4AICollectorConfig(
        seed_urls=seed_urls,
        max_pages=source.get("max_pages", 20),
        max_depth=source.get("max_depth", 2),
        max_pagination_pages=source.get("max_pagination_pages", 200),
        min_published_at=(
            datetime(source["min_published_year"], 1, 1)
            if source.get("min_published_year") is not None
            else None
        ),
        category=source.get("category"),
        department=source.get("department"),
        include_patterns=tuple(source.get("include_patterns", DEFAULT_INCLUDE_PATTERNS)),
        follow_patterns=(
            tuple(source["follow_patterns"]) if "follow_patterns" in source else None
        ),
        collect_patterns=(
            tuple(source["collect_patterns"]) if "collect_patterns" in source else None
        ),
        exclude_patterns=tuple(source.get("exclude_patterns", DEFAULT_EXCLUDE_PATTERNS)),
        allowed_path_prefixes=(
            tuple(allowed_path_prefixes) if allowed_path_prefixes is not None else None
        ),
        collect_seed_pages=source.get("collect_seed_pages", True),
        allowed_keyword_filters=(
            tuple(source["allowed_keyword_filters"])
            if "allowed_keyword_filters" in source
            else None
        ),
        blocked_keyword_filters=(
            tuple(source["blocked_keyword_filters"])
            if "blocked_keyword_filters" in source
            else None
        ),
        allowed_author_department_filters=(
            tuple(source["allowed_author_department_filters"])
            if "allowed_author_department_filters" in source
            else None
        ),
        blocked_author_department_filters=(
            tuple(source["blocked_author_department_filters"])
            if "blocked_author_department_filters" in source
            else None
        ),
        docling_config=DoclingCollectorConfig(
            category=source.get("category"),
            department=source.get("department"),
            skip_unsupported=source.get("skip_unsupported", False),
            skip_images=source.get("skip_images", False),
        ),
    )


def collect_source_documents(
    crawler_config: Crawl4AICollectorConfig,
    *,
    timeout_seconds: int,
):
    if timeout_seconds <= 0:
        return collect_documents_with_crawl4ai(crawler_config)

    context_name = "fork" if "fork" in mp.get_all_start_methods() else None
    context = mp.get_context(context_name) if context_name else mp.get_context()
    pool = context.Pool(processes=1)
    try:
        result = pool.apply_async(collect_documents_with_crawl4ai, (crawler_config,))
        documents = result.get(timeout_seconds)
    except mp.TimeoutError as exc:
        pool.terminate()
        pool.join()
        raise TimeoutError(
            f"source collection exceeded {timeout_seconds} seconds"
        ) from exc
    except Exception:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
        return documents


def _derive_allowed_path_prefixes(seed_urls: List[str]) -> List[str]:
    prefixes: List[str] = []
    for seed_url in seed_urls:
        parsed = urlparse(seed_url)
        path = parsed.path or "/"
        if path == "/":
            prefix = "/"
        else:
            first_segment = path.strip("/").split("/", 1)[0]
            prefix = f"/{first_segment}/" if first_segment else "/"
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def default_report_output_dir() -> Path:
    return PROJECT_ROOT / ".tmp" / "ingest-reports"


def write_ingest_report(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"ingest-report-{timestamp}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def classify_source_report(
    *,
    category: str | None,
    raw_documents: int,
    documents: int,
    exact_duplicates_removed: int,
    version_duplicates_removed: int,
) -> Dict[str, str]:
    if raw_documents == 0:
        return {
            "status": "no_content_discovered",
            "reason": "No crawlable documents were discovered from the source.",
        }

    if documents == 0:
        if exact_duplicates_removed or version_duplicates_removed:
            return {
                "status": "fully_deduplicated",
                "reason": "Documents were discovered but removed as duplicates or older versions.",
            }
        if category == "faq":
            return {
                "status": "empty_board_or_filtered",
                "reason": "FAQ board was empty or filtered out as a non-detail page.",
            }
        return {
            "status": "filtered_out_or_empty",
            "reason": "Documents were discovered but filtered out or reduced to zero usable items.",
        }

    return {
        "status": "ok",
        "reason": "Source produced usable documents.",
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the crawler ingestion pipeline.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Limit ingestion to the named source. Can be provided multiple times.",
    )
    parser.add_argument(
        "--source-prefix",
        action="append",
        dest="source_prefixes",
        help="Limit ingestion to sources whose names start with the prefix. Can be provided multiple times.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        dest="exclude_sources",
        help="Exclude the named source from ingestion. Can be provided multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many matching sources.",
    )
    parser.add_argument(
        "--source-timeout-seconds",
        type=int,
        default=0,
        help="Skip a source if collection takes longer than this many seconds. 0 disables the timeout.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding even if a source enables it.",
    )
    parser.add_argument(
        "--store-vectors",
        action="store_true",
        help="Persist embedded chunks to the configured vector store.",
    )
    parser.add_argument(
        "--store-db",
        action="store_true",
        help="Persist crawled documents, chunks, attachments, and ingest reports to PostgreSQL.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Override per-source max_pages for this run.",
    )
    parser.add_argument(
        "--max-pagination-pages",
        type=int,
        help="Override per-source max_pagination_pages for this run.",
    )
    return parser.parse_args(argv)


def select_sources(
    sources: List[Dict[str, Any]],
    *,
    names: List[str] | None = None,
    prefixes: List[str] | None = None,
    exclude_names: List[str] | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    selected_sources = sources

    if names:
        allowed_names = set(names)
        selected_sources = [
            source for source in selected_sources if source.get("name") in allowed_names
        ]

    if prefixes:
        selected_sources = [
            source
            for source in selected_sources
            if any(source.get("name", "").startswith(prefix) for prefix in prefixes)
        ]

    if exclude_names:
        blocked_names = set(exclude_names)
        selected_sources = [
            source
            for source in selected_sources
            if source.get("name") not in blocked_names
        ]

    if limit is not None and limit >= 0:
        selected_sources = selected_sources[:limit]

    return selected_sources


def apply_runtime_overrides(
    source: Dict[str, Any],
    *,
    skip_embed: bool,
    max_pages: int | None,
    max_pagination_pages: int | None,
) -> Dict[str, Any]:
    overridden = dict(source)
    if skip_embed:
        overridden["embed"] = False
    if max_pages is not None:
        overridden["max_pages"] = max_pages
    if max_pagination_pages is not None:
        overridden["max_pagination_pages"] = max_pagination_pages
    return overridden


def select_embedding_chunks(
    chunks: List[Any],
    *,
    embedding_limit: int | None,
    attachment_embedding_limit: int | None,
) -> List[Any]:
    if not chunks:
        return []
    if embedding_limit is None and attachment_embedding_limit is None:
        return chunks

    selected: List[Any] = []
    seen_chunk_ids: set[str] = set()

    main_limit = embedding_limit if embedding_limit is not None else len(chunks)
    if main_limit > 0:
        for chunk in chunks:
            if len(selected) >= main_limit:
                break
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)

    if attachment_embedding_limit is not None and attachment_embedding_limit > 0:
        attachment_count = 0
        for chunk in chunks:
            if attachment_count >= attachment_embedding_limit:
                break
            if chunk.chunk_id in seen_chunk_ids:
                continue
            if not _is_attachment_chunk(chunk):
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            attachment_count += 1

    return selected


def _is_attachment_chunk(chunk: Any) -> bool:
    source_type = getattr(chunk, "source_type", "")
    if source_type in ATTACHMENT_SOURCE_TYPES:
        return True
    return "downloadbbsfile.do" in getattr(chunk, "source_url", "").lower()


def main(argv: List[str] | None = None) -> None:
    args = parse_args([] if argv is None else argv)
    if args.skip_embed and args.store_vectors:
        raise ValueError("--store-vectors cannot be used together with --skip-embed.")

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    config_path = Path(__file__).resolve().with_name("sources.yaml")

    sources = load_sources_config(config_path)
    if not sources:
        print("No sources configured in app/crawlers/sources.yaml")
        return
    sources = select_sources(
        sources,
        names=args.sources,
        prefixes=args.source_prefixes,
        exclude_names=args.exclude_sources,
        limit=args.limit,
    )
    if not sources:
        print("No sources matched the requested filters.")
        return

    total_documents = 0
    total_chunks = 0
    total_embedded_chunks = 0
    total_stored_chunks = 0
    total_db_documents = 0
    total_db_chunks = 0
    source_reports: List[Dict[str, Any]] = []
    source_status_counts: Dict[str, int] = {}

    for configured_source in sources:
        source_started_at = datetime.now()
        source = apply_runtime_overrides(
            configured_source,
            skip_embed=args.skip_embed,
            max_pages=args.max_pages,
            max_pagination_pages=args.max_pagination_pages,
        )
        print(f"[{source['name']}] start", flush=True)
        try:
            crawler_config = build_crawler_config(source)
            raw_documents = collect_source_documents(
                crawler_config,
                timeout_seconds=args.source_timeout_seconds,
            )
            dedup_result = select_latest_documents(raw_documents)
            documents = dedup_result.documents
            chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
            embedded_count = 0
            stored_count = 0

            if source.get("embed", True) and chunks:
                embedding_limit = source.get("embedding_limit")
                attachment_embedding_limit = source.get("attachment_embedding_limit")
                target_chunks = select_embedding_chunks(
                    chunks,
                    embedding_limit=embedding_limit,
                    attachment_embedding_limit=attachment_embedding_limit,
                )
                embedded_chunks = embed_chunks(target_chunks)
                embedded_count = len(embedded_chunks)
                total_embedded_chunks += embedded_count
                if args.store_vectors and embedded_chunks:
                    delete_embedded_chunks_for_documents(documents)
                    stored_count = upsert_embedded_chunks(
                        embedded_chunks,
                        category=source.get("category"),
                        department=source.get("department"),
                    )
                    total_stored_chunks += stored_count

            total_documents += len(documents)
            total_chunks += len(chunks)

            source_summary = classify_source_report(
                category=source.get("category"),
                raw_documents=dedup_result.total_input,
                documents=len(documents),
                exact_duplicates_removed=dedup_result.exact_duplicates_removed,
                version_duplicates_removed=dedup_result.version_duplicates_removed,
            )
            source_status_counts[source_summary["status"]] = (
                source_status_counts.get(source_summary["status"], 0) + 1
            )
            source_report = {
                "name": source["name"],
                "category": source.get("category"),
                "department": source.get("department"),
                "seed_urls": source["seed_urls"],
                "raw_documents": dedup_result.total_input,
                "documents": len(documents),
                "exact_duplicates_removed": dedup_result.exact_duplicates_removed,
                "version_duplicates_removed": dedup_result.version_duplicates_removed,
                "chunks": len(chunks),
                "embedded_chunks": embedded_count,
                "stored_chunks": stored_count,
                "db_documents": 0,
                "db_chunks": 0,
                "status": source_summary["status"],
                "status_reason": source_summary["reason"],
            }
            if args.store_db:
                stored_db = store_ingest_source_result(
                    run_id=run_id,
                    source=source,
                    documents=documents,
                    chunks=chunks,
                    source_report=source_report,
                    started_at=source_started_at,
                    completed_at=datetime.now(),
                )
                source_report["db_documents"] = stored_db["documents"]
                source_report["db_chunks"] = stored_db["chunks"]
                total_db_documents += stored_db["documents"]
                total_db_chunks += stored_db["chunks"]

            source_reports.append(source_report)

            print(
                f"[{source['name']}] raw_documents={dedup_result.total_input} "
                f"documents={len(documents)} "
                f"exact_duplicates_removed={dedup_result.exact_duplicates_removed} "
                f"version_duplicates_removed={dedup_result.version_duplicates_removed} "
                f"chunks={len(chunks)} embedded_chunks={embedded_count} stored_chunks={stored_count} "
                f"db_documents={source_report['db_documents']} db_chunks={source_report['db_chunks']} "
                f"status={source_summary['status']}",
                flush=True,
            )
        except Exception as exc:
            source_status_counts["error"] = source_status_counts.get("error", 0) + 1
            source_report = {
                "name": source["name"],
                "category": source.get("category"),
                "department": source.get("department"),
                "seed_urls": source["seed_urls"],
                "raw_documents": 0,
                "documents": 0,
                "exact_duplicates_removed": 0,
                "version_duplicates_removed": 0,
                "chunks": 0,
                "embedded_chunks": 0,
                "stored_chunks": 0,
                "db_documents": 0,
                "db_chunks": 0,
                "status": "error",
                "status_reason": f"{exc.__class__.__name__}: {exc}",
            }
            source_reports.append(source_report)
            print(
                f"[{source['name']}] raw_documents=0 documents=0 "
                "exact_duplicates_removed=0 version_duplicates_removed=0 "
                "chunks=0 embedded_chunks=0 stored_chunks=0 db_documents=0 db_chunks=0 "
                f"status=error error={exc.__class__.__name__}: {exc}",
                flush=True,
            )

    print(f"total_documents={total_documents}", flush=True)
    print(f"total_chunks={total_chunks}", flush=True)
    print(f"total_embedded_chunks={total_embedded_chunks}", flush=True)
    print(f"total_stored_chunks={total_stored_chunks}", flush=True)
    print(f"total_db_documents={total_db_documents}", flush=True)
    print(f"total_db_chunks={total_db_chunks}", flush=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "run_id": run_id,
        "source_count": len(sources),
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "total_embedded_chunks": total_embedded_chunks,
        "total_stored_chunks": total_stored_chunks,
        "total_db_documents": total_db_documents,
        "total_db_chunks": total_db_chunks,
        "source_status_counts": source_status_counts,
        "sources": source_reports,
    }
    report_path = write_ingest_report(report, default_report_output_dir())
    print(f"ingest_report={report_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
