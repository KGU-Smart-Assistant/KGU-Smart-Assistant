from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from app.crawlers.crawl_markdown import DEFAULT_USER_AGENT, CrawlOptions, crawl_markdown, load_sources_config
from app.crawlers.prepare_markdown import PrepareOptions, prepare_markdown
from app.schemas import Document, DocumentChunk


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = BACKEND_ROOT / "data"
DEFAULT_CRAWLED_MARKDOWN_ROOT = DATA_ROOT / "crawled_markdown"
DEFAULT_PREPARED_MARKDOWN_ROOT = DATA_ROOT / "prepared_markdown"


@dataclass(frozen=True)
class CrawlPreparePipelineOptions:
    config_path: Path
    crawl_output_dir: Path | None = None
    prepared_output_dir: Path | None = None
    classification_csv: Path | None = None
    classification_effective_date: str = "2026-06-05"
    processing_effective_date: str = "2026-06-05"
    force: bool = False
    actions: tuple[str, ...] = ("KEEP",)
    max_pages: int | None = None
    source_names: tuple[str, ...] = ()
    user_agent: str | None = None
    store_db: bool = False
    run_id: str | None = None


def run_crawl_prepare_pipeline(
    options: CrawlPreparePipelineOptions,
    *,
    crawler: Any | None = None,
    db_session: Any | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    crawl_output_dir, prepared_output_dir = _resolve_output_dirs(options, started_at=started_at)
    sources = load_sources_config(options.config_path)
    if options.source_names:
        allowed = {name.casefold() for name in options.source_names}
        sources = [source for source in sources if source.name.casefold() in allowed]
    crawl_result = crawl_markdown(
        CrawlOptions(
            sources=sources,
            output_dir=crawl_output_dir,
            force=options.force,
            max_pages=options.max_pages,
            user_agent=options.user_agent or DEFAULT_USER_AGENT,
        ),
        crawler=crawler,
    )

    classification_csv = options.classification_csv or (crawl_output_dir / "manifest" / "crawl_classification_seed.csv")
    prepare_result = prepare_markdown(
        PrepareOptions(
            input_dir=crawl_output_dir,
            classification_csv=classification_csv,
            output_dir=prepared_output_dir,
            actions=options.actions,
            classification_effective_date=options.classification_effective_date,
            processing_effective_date=options.processing_effective_date,
            page_type=_prepare_page_type(sources),
            force=options.force,
            dump_report=True,
        )
    )

    completed_at = datetime.now(timezone.utc)
    report = {
        "crawl": crawl_result,
        "prepare": prepare_result,
        "config_path": str(options.config_path),
        "crawl_output_dir": str(crawl_output_dir),
        "prepared_output_dir": str(prepared_output_dir),
        "classification_csv": str(classification_csv),
    }
    if options.store_db:
        report["db_store"] = _store_pipeline_result_to_db(
            options=options,
            sources=sources,
            prepared_output_dir=prepared_output_dir,
            run_id=options.run_id or f"crawl-prepare-{uuid4().hex}",
            started_at=started_at,
            completed_at=completed_at,
            db_session=db_session,
        )
    (prepared_output_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (prepared_output_dir / "manifest" / "pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _resolve_output_dirs(options: CrawlPreparePipelineOptions, *, started_at: datetime) -> tuple[Path, Path]:
    if options.crawl_output_dir and options.prepared_output_dir:
        return options.crawl_output_dir, options.prepared_output_dir

    run_slug = _output_run_slug(options, started_at=started_at)
    return (
        options.crawl_output_dir or (DEFAULT_CRAWLED_MARKDOWN_ROOT / run_slug),
        options.prepared_output_dir or (DEFAULT_PREPARED_MARKDOWN_ROOT / run_slug),
    )


def _output_run_slug(options: CrawlPreparePipelineOptions, *, started_at: datetime) -> str:
    if options.run_id:
        return _safe_path_name(options.run_id)
    source_part = "-".join(_safe_path_name(source) for source in options.source_names if source) or "all-sources"
    timestamp = started_at.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"crawl4ai-{timestamp}-{source_part}"


def _safe_path_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in value.strip())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:120] or "run"


def _prepare_page_type(sources: list[Any]) -> str:
    page_types = {str(getattr(source, "page_type", "LIST_PAGE")).upper() for source in sources}
    return "SINGLE_PAGE" if page_types == {"SINGLE_PAGE"} else "LIST_PAGE"


def _store_pipeline_result_to_db(
    *,
    options: CrawlPreparePipelineOptions,
    sources: list[Any],
    prepared_output_dir: Path,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    db_session: Any | None,
) -> dict[str, Any]:
    from app.db.crawler_store import store_ingest_source_result

    document_rows = _load_prepared_documents(prepared_output_dir, collected_at=completed_at)
    chunk_rows = _load_prepared_chunks(prepared_output_dir)
    documents_by_source: dict[str, list[Document]] = {}
    chunks_by_source: dict[str, list[DocumentChunk]] = {}
    for source_name, document in document_rows:
        documents_by_source.setdefault(source_name, []).append(document)
    for source_name, chunk in chunk_rows:
        chunks_by_source.setdefault(source_name, []).append(chunk)

    close_session = False
    db = db_session
    if db is None:
        from app.db.session import SessionLocal

        db = SessionLocal()
        close_session = True

    try:
        stored: dict[str, Any] = {}
        for source in sources:
            source_documents = documents_by_source.get(source.name, [])
            source_chunks = chunks_by_source.get(source.name, [])
            source_report = {
                "status": "ok",
                "status_reason": "Source prepared successfully.",
                "raw_documents": len(source_documents),
                "documents": len(source_documents),
                "exact_duplicates_removed": 0,
                "version_duplicates_removed": 0,
                "chunks": len(source_chunks),
                "embedded_chunks": 0,
                "stored_chunks": 0,
            }
            stored[source.name] = store_ingest_source_result(
                db,
                run_id=run_id,
                source={
                    "name": source.name,
                    "seed_urls": source.urls,
                    "domain": source.domain,
                    "department": source.department,
                },
                documents=source_documents,
                chunks=source_chunks,
                source_report=source_report,
                started_at=started_at,
                completed_at=completed_at,
                run_type="crawl_prepare",
            )
        return {
            "run_id": run_id,
            "documents": len(document_rows),
            "chunks": len(chunk_rows),
            "sources": stored,
        }
    finally:
        if close_session:
            db.close()


def _load_prepared_documents(prepared_output_dir: Path, *, collected_at: datetime) -> list[tuple[str, Document]]:
    documents: list[tuple[str, Document]] = []
    for path in sorted((prepared_output_dir / "final").glob("*/*.md")):
        metadata, body = _split_markdown_frontmatter(path.read_text(encoding="utf-8-sig"))
        if not body.strip():
            continue
        attachment_urls, attachment_metadata = _prepared_attachments(metadata)
        documents.append(
            (
                str(metadata["source_name"]),
                Document(
                    doc_id=str(metadata["doc_id"]),
                    source_type=str(metadata.get("source_type") or "html"),
                    source_url=str(metadata["source_url"]),
                    title=str(metadata["title"]),
                    content=body.strip(),
                    domain=_optional_str(metadata.get("domain")),
                    department=_optional_str(metadata.get("department")),
                    author_department=_optional_str(metadata.get("author_department")),
                    published_at=_optional_datetime(metadata.get("published_at")),
                    collected_at=_optional_datetime(metadata.get("collected_at")) or collected_at,
                    attachment_urls=attachment_urls,
                    attachment_metadata=attachment_metadata,
                ),
            )
        )
    return documents


def _load_prepared_chunks(prepared_output_dir: Path) -> list[tuple[str, DocumentChunk]]:
    path = prepared_output_dir / "chunks" / "chunks.jsonl"
    if not path.exists():
        return []
    chunks: list[tuple[str, DocumentChunk]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chunks.append(
            (
                str(row["source_name"]),
                DocumentChunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row.get("embedding_text") or row.get("text") or row["content"]),
                    title=str(row["title"]),
                    source_url=str(row["source_url"]),
                    source_type="html",
                    domain=_optional_str(row.get("domain")),
                    department=_optional_str(row.get("department")),
                    published_at=_optional_datetime(row.get("published_at")),
                    content=_optional_str(row.get("content")),
                    embedding_text=_optional_str(row.get("embedding_text") or row.get("text")),
                    chunk_text_hash=_optional_str(row.get("chunk_text_hash")),
                    section_title=_optional_str(row.get("section_title")),
                    section_path=[str(value) for value in row.get("section_path") or []],
                    section_kind=_optional_str(row.get("section_kind")),
                    source_name=_optional_str(row.get("source_name")),
                    embedding_eligibility=_optional_str(row.get("embedding_eligibility")),
                    chunk_quality_status=_optional_str(row.get("chunk_quality_status")),
                    document_quality_status=_optional_str(row.get("document_quality_status")),
                    content_token_count=_optional_int(row.get("content_token_count")),
                    embedding_token_count=_optional_int(row.get("embedding_token_count")),
                    valid_until=_optional_datetime(row.get("valid_until")),
                    chunk_valid_until=_optional_datetime(row.get("chunk_valid_until")),
                    prepared_body_hash=_optional_str(row.get("prepared_body_hash")),
                    prepared_artifact_hash=_optional_str(row.get("prepared_artifact_hash")),
                    metadata_json=_chunk_metadata_json(row),
                ),
            )
        )
    return chunks


def _split_markdown_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    _prefix, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter) or {}, body.strip()


def _prepared_attachments(metadata: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, str]]]:
    urls = [str(url) for url in metadata.get("attachment_urls") or [] if url]
    attachment_metadata = metadata.get("attachment_metadata") or {}
    for attachment in metadata.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        url = attachment.get("canonical_attachment_url") or attachment.get("original_attachment_url")
        if not url:
            continue
        url = str(url)
        urls.append(url)
        filename = attachment.get("filename")
        if filename:
            attachment_metadata.setdefault(url, {})["filename"] = str(filename)
    return list(dict.fromkeys(urls)), attachment_metadata


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _chunk_metadata_json(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "warning_codes",
        "preview",
        "processing_effective_date",
        "tokenizer_name",
        "tokenizer_version",
        "quality_gate_version",
        "review_status",
        "initial_filter_status",
        "page_type",
        "document_type",
        "content_char_count",
        "embedding_char_count",
    )
    return {key: row.get(key) for key in keys if row.get(key) not in (None, "")}


def _parse_args() -> CrawlPreparePipelineOptions:
    parser = argparse.ArgumentParser(description="Run crawl_markdown and prepare_markdown in one pass.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--crawl-output-dir",
        type=Path,
        help="Raw Crawl4AI markdown output directory. Defaults to backend/data/crawled_markdown/<run>.",
    )
    parser.add_argument(
        "--prepared-output-dir",
        type=Path,
        help="Prepared markdown output directory. Defaults to backend/data/prepared_markdown/<run>.",
    )
    parser.add_argument("--classification-csv", type=Path)
    parser.add_argument("--classification-effective-date", default="2026-06-05")
    parser.add_argument("--processing-effective-date", default="2026-06-05")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action", dest="actions", action="append")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--source", dest="source_names", action="append")
    parser.add_argument("--user-agent")
    parser.add_argument("--store-db", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    return CrawlPreparePipelineOptions(
        config_path=args.config,
        crawl_output_dir=args.crawl_output_dir,
        prepared_output_dir=args.prepared_output_dir,
        classification_csv=args.classification_csv,
        classification_effective_date=args.classification_effective_date,
        processing_effective_date=args.processing_effective_date,
        force=args.force,
        actions=tuple(args.actions) if args.actions else ("KEEP",),
        max_pages=args.max_pages,
        source_names=tuple(args.source_names or ()),
        user_agent=args.user_agent,
        store_db=args.store_db,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    print(json.dumps(run_crawl_prepare_pipeline(_parse_args()), ensure_ascii=False, indent=2))
