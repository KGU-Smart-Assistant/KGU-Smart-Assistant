from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import yaml

from app.db.crawler_identity import normalize_url


SCHEMA_VERSION = "prepared-markdown-v1"
PREPROCESS_VERSION = "markdown-preprocess-v1"
CHUNKING_VERSION = "semantic-markdown-v1"
QUALITY_GATE_VERSION = "prepared-quality-v1"
URL_NORMALIZATION_VERSION = "crawler-identity-normalize-url-v1"
TOKENIZER_NAME = "cl100k_base"
MAX_EMBEDDING_CHARS = 1600
MAX_EMBEDDING_TOKENS = 700

REQUIRED_CLASSIFICATION_COLUMNS = {
    "recommended_action",
    "doc_id",
    "source_name",
    "source_url",
    "final_md_path",
    "title",
    "domain",
    "department",
    "filter_reason",
    "confidence",
}

ALLOWED_DOCUMENT_WARNINGS = {
    "LOW_TEXT_BUT_HAS_CONTACT",
    "LOW_TEXT_BUT_HAS_ATTACHMENT_NAME",
    "DERIVED_HEADING_RULE_APPLIED",
    "MINOR_MARKDOWN_REPAIRED",
}
ALLOWED_CHUNK_WARNINGS = {
    "SHORT_BUT_COMPLETE",
    "SINGLE_SECTION_DOCUMENT",
    "UNSPLITTABLE_TABLE",
}

PAGINATION_NOISE_PATTERNS = (
    re.compile(r"처음\s*페이지.*이전\s*10\s*페이지.*다음\s*10\s*페이지.*끝\s*페이지"),
    re.compile(r"first\s*page.*prev.*next.*last\s*page", re.IGNORECASE),
    re.compile(r"^총\s*게시물\s*:\s*\d+\s*페이지\s*:\s*\d+\s*/\s*\d+\s*$"),
)

NAVIGATION_MENU_TERMS = (
    "게시물 검색",
    "처음 페이지",
    "이전 10 페이지",
    "다음 10 페이지",
    "끝 페이지",
    "재무회계팀소개",
    "등록금납부안내",
    "상조회안내",
    "결산공고",
)

TIME_SENSITIVE_TERMS = (
    "신청 기간",
    "신청기간",
    "접수 기간",
    "접수기간",
    "제출기한",
    "제출 기한",
    "모집 기간",
    "모집기간",
    "행사 기간",
    "행사기간",
    "납부 기간",
    "납부기간",
)

TIME_SENSITIVE_DATE_PATTERN = re.compile(
    r"(?:20\d{2}\s*[.\-/년]\s*\d{1,2}\s*[.\-/월]\s*\d{0,2}|"
    r"\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*(?:일)?|"
    r"~|까지|마감|기한)"
)


@dataclass(frozen=True)
class PrepareOptions:
    input_dir: Path
    classification_csv: Path
    output_dir: Path
    actions: tuple[str, ...]
    classification_effective_date: str
    processing_effective_date: str
    review_decisions_csv: Path | None = None
    source_name: str | None = None
    document_type: str | None = None
    page_type: str = "LIST_PAGE"
    manual_review_only: bool = False
    force: bool = False
    limit: int | None = None
    sample_size: int | None = None
    keep_staging: bool = False
    dry_run: bool = False
    dump_report: bool = False


@dataclass(frozen=True)
class MarkdownCandidate:
    path: Path
    metadata: dict[str, Any]
    body: str
    source_content_hash: str


def prepare_markdown(options: PrepareOptions) -> dict[str, Any]:
    rows = _read_classification_rows(options.classification_csv)
    classification_manifest_hash = _classification_manifest_hash(rows)
    rows = _filter_rows(rows, options)
    if options.limit is not None:
        rows = rows[: options.limit]
    if options.sample_size is not None:
        rows = rows[: options.sample_size]

    review_decisions = _read_review_decisions(options.review_decisions_csv)
    output_root = options.output_dir
    tmp_root = output_root.with_name(f"{output_root.name}.tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if output_root.exists() and options.force:
        shutil.rmtree(output_root)
    elif output_root.exists() and not options.dry_run:
        raise FileExistsError(f"Output directory already exists: {output_root}")

    if options.dry_run:
        return _dry_run_mapping_report(rows=rows, options=options, classification_manifest_hash=classification_manifest_hash)

    (tmp_root / "manifest").mkdir(parents=True, exist_ok=True)
    (tmp_root / "final").mkdir(parents=True, exist_ok=True)
    (tmp_root / "manual_review" / "final").mkdir(parents=True, exist_ok=True)
    (tmp_root / "manual_review" / "chunk_preview").mkdir(parents=True, exist_ok=True)
    (tmp_root / "chunks").mkdir(parents=True, exist_ok=True)
    staging_root = tmp_root / "staging_chunks"
    staging_root.mkdir(parents=True, exist_ok=True)

    mapping_rows: list[dict[str, Any]] = []
    processing_rows: list[dict[str, Any]] = []
    document_quality_rows: list[dict[str, Any]] = []
    chunk_quality_rows: list[dict[str, Any]] = []
    review_queue_rows: list[dict[str, Any]] = []
    published_chunks: list[dict[str, Any]] = []
    prepared_artifact_hashes: list[str] = []

    for source_row in rows:
        result = _process_row(
            row=source_row,
            options=options,
            tmp_root=tmp_root,
            staging_root=staging_root,
            review_decisions=review_decisions,
            classification_manifest_hash=classification_manifest_hash,
        )
        mapping_rows.append(result["mapping"])
        processing_rows.append(result["processing"])
        document_quality_rows.append(result["document_quality"])
        chunk_quality_rows.append(result["chunk_quality"])
        if result.get("review_queue"):
            review_queue_rows.append(result["review_queue"])
        published_chunks.extend(result["published_chunks"])
        if result.get("prepared_artifact_hash"):
            prepared_artifact_hashes.append(result["prepared_artifact_hash"])

    published_chunks = sorted(published_chunks, key=lambda item: (item["source_name"], item["canonical_doc_key_hash"], item["chunk_index"]))
    chunks_jsonl = "".join(f"{_canonical_json(chunk)}\n" for chunk in published_chunks)
    chunks_jsonl_hash = _sha256_text(chunks_jsonl)
    _write_text(tmp_root / "chunks" / "chunks.jsonl", chunks_jsonl)

    report_context = {
        "classification_manifest_hash": classification_manifest_hash,
        "classification_effective_date": options.classification_effective_date,
        "processing_effective_date": options.processing_effective_date,
        "preprocess_version": PREPROCESS_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "quality_gate_version": QUALITY_GATE_VERSION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_version": _tokenizer_version(),
        "schema_version": SCHEMA_VERSION,
        "url_normalization_version": URL_NORMALIZATION_VERSION,
        "rebuild_id": output_root.name,
        "chunks_jsonl_hash": chunks_jsonl_hash,
    }
    _write_csv(tmp_root / "manifest" / "mapping_report.csv", mapping_rows)
    _write_csv(tmp_root / "manifest" / "processing_report.csv", _add_context(processing_rows, report_context))
    _write_csv(tmp_root / "manifest" / "document_quality_report.csv", _add_context(document_quality_rows, report_context))
    _write_csv(tmp_root / "manifest" / "chunk_quality_report.csv", _add_context(chunk_quality_rows, report_context))
    _write_csv(tmp_root / "manifest" / "review_queue.csv", review_queue_rows)
    _snapshot_or_template_review_decisions(tmp_root, options.review_decisions_csv)
    _write_json(
        tmp_root / "manifest" / "version_report.json",
        {
            **report_context,
            "prepared_artifact_hashes": sorted(prepared_artifact_hashes),
            "published_chunk_count": len(published_chunks),
            "processed_document_count": len(rows),
        },
    )
    _write_text(
        tmp_root / "manifest" / "input_classification.hash.txt",
        f"{classification_manifest_hash}\n",
    )

    if not options.keep_staging and staging_root.exists():
        shutil.rmtree(staging_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        tmp_root.rename(output_root)
    except PermissionError:
        shutil.move(str(tmp_root), str(output_root))
    return {
        "processed_rows": len(rows),
        "published_chunks": len(published_chunks),
        "classification_manifest_hash": classification_manifest_hash,
        "chunks_jsonl_hash": chunks_jsonl_hash,
    }


def _process_row(
    *,
    row: dict[str, str],
    options: PrepareOptions,
    tmp_root: Path,
    staging_root: Path,
    review_decisions: dict[tuple[str, str], dict[str, str]],
    classification_manifest_hash: str,
) -> dict[str, Any]:
    normalized_source_url = normalize_url(row["source_url"])
    canonical_doc_key = _canonical_doc_key(row["source_name"], normalized_source_url)
    canonical_doc_key_hash = _sha256_text(canonical_doc_key)
    base_report = {
        "doc_id": row["doc_id"],
        "source_name": row["source_name"],
        "title": row["title"],
        "initial_filter_status": row["recommended_action"],
        "canonical_doc_key": canonical_doc_key,
        "canonical_doc_key_hash": canonical_doc_key_hash,
    }
    mapping = {**base_report, "mapping_status": "NOT_CHECKED", "match_method": "", "mapping_reason": ""}
    processing = {**base_report, "processing_status": "PENDING", "prepared_artifact_hash": "", "prepared_body_hash": ""}
    document_quality = {**base_report, "document_quality_status": "NOT_CHECKED", "document_warning_codes": "", "document_quality_reason": ""}
    chunk_quality = {**base_report, "chunk_status": "NOT_CREATED", "chunk_quality_status": "NOT_CHECKED", "chunk_warning_codes": "", "chunk_quality_reason": "", "embedding_eligibility": "NOT_ELIGIBLE"}

    candidate = _locate_candidate(row=row, input_dir=options.input_dir)
    if isinstance(candidate, str):
        mapping.update({"mapping_status": candidate, "mapping_reason": "Could not safely map classification row to Markdown."})
        processing["processing_status"] = "SKIPPED"
        return _empty_result(mapping, processing, document_quality, chunk_quality)

    source_content_hash = candidate.source_content_hash
    mapping.update(
        {
            "mapping_status": "MATCHED",
            "match_method": _match_method(row=row, metadata=candidate.metadata, normalized_source_url=normalized_source_url),
            "source_content_hash": source_content_hash,
        }
    )
    if mapping["match_method"] == "HASH_MISMATCH":
        mapping["mapping_status"] = "HASH_MISMATCH"
        processing["processing_status"] = "SKIPPED"
        return _empty_result(mapping, processing, document_quality, chunk_quality)

    date_blocked = _dates_require_recheck(
        row=row,
        classification_effective_date=options.classification_effective_date,
        processing_effective_date=options.processing_effective_date,
    )
    review_status, review_reason = _review_status(
        row=row,
        canonical_doc_key=canonical_doc_key,
        source_content_hash=source_content_hash,
        classification_manifest_hash=classification_manifest_hash,
        review_decisions=review_decisions,
        date_blocked=date_blocked,
    )

    attachment_urls = _attachment_urls(candidate.metadata)
    title = _title_from_body(candidate.body) or row["title"]
    prepared_body = preprocess_markdown_body(
        body=candidate.body,
        title=title,
        attachment_urls=attachment_urls,
    )
    prepared_body_hash = _hash_body(prepared_body)
    metadata = _prepared_metadata(
        row=row,
        candidate=candidate,
        canonical_doc_key=canonical_doc_key,
        canonical_doc_key_hash=canonical_doc_key_hash,
        normalized_source_url=normalized_source_url,
        source_content_hash=source_content_hash,
        prepared_body_hash=prepared_body_hash,
        classification_manifest_hash=classification_manifest_hash,
        options=options,
        review_status=review_status,
        review_reason=review_reason,
    )
    public_metadata = _public_prepared_metadata(metadata)
    metadata["metadata_hash"] = _metadata_hash(public_metadata)
    public_metadata["metadata_hash"] = metadata["metadata_hash"]
    artifact = _serialize_markdown(public_metadata, prepared_body)
    prepared_artifact_hash = _sha256_text(artifact)
    processing.update(
        {
            "processing_status": "PREPROCESSED",
            "source_content_hash": source_content_hash,
            "prepared_body_hash": prepared_body_hash,
            "metadata_hash": metadata["metadata_hash"],
            "prepared_artifact_hash": prepared_artifact_hash,
            "review_status": review_status,
            "review_reason": review_reason,
        }
    )

    doc_quality_status, doc_warnings, doc_reason = _document_quality(prepared_body, metadata, date_blocked=date_blocked)
    document_quality.update(
        {
            "document_quality_status": doc_quality_status,
            "document_warning_codes": "|".join(doc_warnings),
            "document_quality_reason": doc_reason,
        }
    )
    metadata["document_quality_status"] = doc_quality_status
    metadata["document_warning_codes"] = doc_warnings
    output_path = _prepared_markdown_path(
        tmp_root=tmp_root,
        row=row,
        canonical_doc_key_hash=canonical_doc_key_hash,
        title=title,
        manual=row["recommended_action"] == "MANUAL_REVIEW" and review_status != "APPROVED_KEEP",
    )
    _write_text(output_path, artifact)

    if row["recommended_action"] == "MANUAL_REVIEW" and review_status != "APPROVED_KEEP":
        preview_chunks = _make_chunks(
            body=prepared_body,
            metadata=metadata,
            prepared_body_hash=prepared_body_hash,
            prepared_artifact_hash=prepared_artifact_hash,
            preview=True,
        )
        preview_path = tmp_root / "manual_review" / "chunk_preview" / row["source_name"] / f"{canonical_doc_key_hash}.chunks.json"
        _write_json(preview_path, preview_chunks)
        chunk_quality.update(
            {
                "chunk_status": "PREVIEW_CREATED",
                "chunk_quality_status": "NOT_CHECKED",
                "embedding_eligibility": "BLOCKED_BY_REVIEW",
            }
        )
        return {
            "mapping": mapping,
            "processing": processing,
            "document_quality": document_quality,
            "chunk_quality": chunk_quality,
            "review_queue": _review_queue_row(base_report, review_status, review_reason, source_content_hash, classification_manifest_hash),
            "published_chunks": [],
            "prepared_artifact_hash": prepared_artifact_hash,
        }

    if review_status in {"APPROVED_EXCLUDE", "RECHECK_REQUIRED", "PENDING", "PROMOTION_CANDIDATE"}:
        chunk_quality.update({"embedding_eligibility": "BLOCKED_BY_REVIEW"})
        return _finished_no_chunks(mapping, processing, document_quality, chunk_quality, prepared_artifact_hash)
    if doc_quality_status == "FAILED" or (doc_quality_status == "WARNING" and not _all_allowed(doc_warnings, ALLOWED_DOCUMENT_WARNINGS)):
        chunk_quality.update({"embedding_eligibility": "BLOCKED_BY_QUALITY"})
        return _finished_no_chunks(mapping, processing, document_quality, chunk_quality, prepared_artifact_hash)

    staging_chunks = _make_chunks(
        body=prepared_body,
        metadata=metadata,
        prepared_body_hash=prepared_body_hash,
        prepared_artifact_hash=prepared_artifact_hash,
        preview=False,
    )
    staging_chunks = _drop_attachment_only_chunks_when_possible(staging_chunks)
    staging_chunks = _drop_noninformative_chunks_when_possible(staging_chunks)
    staging_chunks = _drop_expired_time_sensitive_chunks_when_possible(staging_chunks)
    staging_path = staging_root / row["source_name"] / f"{canonical_doc_key_hash}.chunks.json"
    _write_json(staging_path, staging_chunks)
    chunk_status, chunk_q_status, chunk_warnings, chunk_reason, eligibility = _chunk_quality(staging_chunks)
    chunk_quality.update(
        {
            "chunk_status": chunk_status,
            "chunk_quality_status": chunk_q_status,
            "chunk_warning_codes": "|".join(chunk_warnings),
            "chunk_quality_reason": chunk_reason,
            "embedding_eligibility": eligibility,
        }
    )
    published_chunks: list[dict[str, Any]] = []
    if eligibility == "ELIGIBLE":
        published_chunks = staging_chunks
        chunk_quality["chunk_status"] = "PUBLISHED"
        published_path = tmp_root / "chunks" / row["source_name"] / f"{canonical_doc_key_hash}.chunks.json"
        _write_json(published_path, published_chunks)
        processing["processing_status"] = "CHUNKED"
    else:
        processing["processing_status"] = "CHUNK_FAILED"
    return {
        "mapping": mapping,
        "processing": processing,
        "document_quality": document_quality,
        "chunk_quality": chunk_quality,
        "review_queue": None,
        "published_chunks": published_chunks,
        "prepared_artifact_hash": prepared_artifact_hash,
    }


def _read_classification_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_CLASSIFICATION_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"Classification CSV is missing required columns: {', '.join(missing)}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _filter_rows(rows: list[dict[str, str]], options: PrepareOptions) -> list[dict[str, str]]:
    actions = set(options.actions)
    if options.manual_review_only:
        actions = {"MANUAL_REVIEW"}
    filtered = [
        row
        for row in rows
        if row.get("recommended_action") in actions
        and (not options.source_name or row.get("source_name") == options.source_name)
        and (not options.document_type or row.get("document_type") == options.document_type)
    ]
    return sorted(filtered, key=lambda row: (row.get("source_name", ""), row.get("doc_id", ""), normalize_url(row.get("source_url", ""))))


def _dry_run_mapping_report(
    *,
    rows: list[dict[str, str]],
    options: PrepareOptions,
    classification_manifest_hash: str,
) -> dict[str, Any]:
    output_root = options.output_dir
    if output_root.exists() and options.force:
        shutil.rmtree(output_root)
    elif output_root.exists() and options.dump_report:
        raise FileExistsError(f"Output directory already exists: {output_root}")

    mapping_rows: list[dict[str, Any]] = []
    by_status: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        normalized_source_url = normalize_url(row["source_url"])
        canonical_doc_key = _canonical_doc_key(row["source_name"], normalized_source_url)
        canonical_doc_key_hash = _sha256_text(canonical_doc_key)
        mapping: dict[str, Any] = {
            "doc_id": row["doc_id"],
            "source_name": row["source_name"],
            "title": row["title"],
            "initial_filter_status": row["recommended_action"],
            "canonical_doc_key": canonical_doc_key,
            "canonical_doc_key_hash": canonical_doc_key_hash,
            "mapping_status": "NOT_CHECKED",
            "match_method": "",
            "mapping_reason": "",
        }
        candidate = _locate_candidate(row=row, input_dir=options.input_dir)
        if isinstance(candidate, str):
            mapping.update({"mapping_status": candidate, "mapping_reason": "Could not safely map classification row to Markdown."})
        else:
            match_method = _match_method(row=row, metadata=candidate.metadata, normalized_source_url=normalized_source_url)
            mapping.update(
                {
                    "mapping_status": "HASH_MISMATCH" if match_method == "HASH_MISMATCH" else "MATCHED",
                    "match_method": match_method,
                    "source_content_hash": candidate.source_content_hash,
                }
            )
        status = str(mapping["mapping_status"])
        source = row["source_name"]
        by_status[status] = by_status.get(status, 0) + 1
        by_source.setdefault(source, {})
        by_source[source][status] = by_source[source].get(status, 0) + 1
        mapping_rows.append(mapping)

    summary_rows = [
        {"source_name": source, **counts, "total": sum(counts.values())}
        for source, counts in sorted(by_source.items())
    ]
    if options.dump_report:
        manifest_root = output_root / "manifest"
        manifest_root.mkdir(parents=True, exist_ok=True)
        _write_csv(manifest_root / "mapping_report.csv", mapping_rows)
        _write_csv(manifest_root / "mapping_summary_by_source.csv", summary_rows)
        _write_json(
            manifest_root / "dry_run_summary.json",
            {
                "classification_manifest_hash": classification_manifest_hash,
                "rows": len(rows),
                "mapping_status_counts": by_status,
                "source_count": len(by_source),
            },
        )
    return {
        "rows": len(rows),
        "classification_manifest_hash": classification_manifest_hash,
        "mapping_status_counts": by_status,
        "source_count": len(by_source),
    }


def _classification_manifest_hash(rows: list[dict[str, str]]) -> str:
    normalized = [
        {key: " ".join(str(value or "").split()) for key, value in sorted(row.items())}
        for row in rows
    ]
    normalized.sort(key=lambda row: (row.get("source_name", ""), row.get("doc_id", ""), normalize_url(row.get("source_url", ""))))
    return _sha256_json(normalized)


def _locate_candidate(*, row: dict[str, str], input_dir: Path) -> MarkdownCandidate | str:
    try:
        path = _resolve_locator(input_dir=input_dir, locator=row.get("final_md_path", ""))
    except ValueError:
        return "FAILED"
    if not path.exists():
        return "UNMATCHED"
    try:
        text = _read_text(path)
        frontmatter, body = _split_frontmatter(text)
        metadata = yaml.safe_load(frontmatter) or {}
    except Exception:
        return "FAILED"
    return MarkdownCandidate(
        path=path,
        metadata=metadata,
        body=body,
        source_content_hash=_hash_body(body),
    )


def _resolve_locator(*, input_dir: Path, locator: str) -> Path:
    if not locator:
        raise ValueError("empty final_md_path")
    normalized = locator.replace("\\", "/")
    raw_path = Path(normalized)
    if raw_path.is_absolute():
        raise ValueError("absolute final_md_path is not allowed")
    parts = list(raw_path.parts)
    if parts and parts[0] == input_dir.name:
        parts = parts[1:]
    candidate = (input_dir / Path(*parts)).resolve()
    root = input_dir.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("final_md_path escapes input-dir")
    return candidate


def _match_method(*, row: dict[str, str], metadata: dict[str, Any], normalized_source_url: str) -> str:
    if str(metadata.get("doc_id") or "") == row["doc_id"]:
        if str(metadata.get("source_name") or "") != row["source_name"]:
            return "HASH_MISMATCH"
        if normalize_url(str(metadata.get("source_url") or "")) != normalized_source_url:
            return "HASH_MISMATCH"
        return "DOC_ID"
    if str(metadata.get("source_name") or "") == row["source_name"] and normalize_url(str(metadata.get("source_url") or "")) == normalized_source_url:
        return "SOURCE_URL"
    return "HASH_MISMATCH"


def _review_status(
    *,
    row: dict[str, str],
    canonical_doc_key: str,
    source_content_hash: str,
    classification_manifest_hash: str,
    review_decisions: dict[tuple[str, str], dict[str, str]],
    date_blocked: bool,
) -> tuple[str, str]:
    if date_blocked:
        return "RECHECK_REQUIRED", "classification_and_processing_dates_differ"
    action = row["recommended_action"]
    decision = review_decisions.get((row["doc_id"], canonical_doc_key))
    if decision:
        if decision.get("source_content_hash") != source_content_hash or decision.get("classification_manifest_hash") != classification_manifest_hash:
            return "RECHECK_REQUIRED", "review_decision_hash_mismatch"
        if decision.get("decision") == "APPROVED_KEEP":
            return "APPROVED_KEEP", decision.get("reason", "")
        if decision.get("decision") == "APPROVED_EXCLUDE":
            return "APPROVED_EXCLUDE", decision.get("reason", "")
    if action == "KEEP":
        return "NOT_REQUIRED", ""
    if action == "MANUAL_REVIEW":
        if _looks_like_promotion_candidate(row):
            return "PROMOTION_CANDIDATE", "manual_review_candidate_requires_human_approval"
        return "PENDING", "manual_review_requires_human_approval"
    return "PENDING", "not_in_embedding_scope"


def _looks_like_promotion_candidate(row: dict[str, str]) -> bool:
    text = " ".join([row.get("domain", ""), row.get("document_type", ""), row.get("title", ""), row.get("snippet", "")])
    return any(keyword in text.casefold() for keyword in ("faq", "메일", "전화", "연락", "자료", "양식", "신청서"))


def _dates_require_recheck(*, row: dict[str, str], classification_effective_date: str, processing_effective_date: str) -> bool:
    if classification_effective_date == processing_effective_date:
        return False
    return any(row.get(key) for key in ("relevant_end_date", "title_period_end", "all_detected_end_date"))


def _prepared_metadata(
    *,
    row: dict[str, str],
    candidate: MarkdownCandidate,
    canonical_doc_key: str,
    canonical_doc_key_hash: str,
    normalized_source_url: str,
    source_content_hash: str,
    prepared_body_hash: str,
    classification_manifest_hash: str,
    options: PrepareOptions,
    review_status: str,
    review_reason: str,
) -> dict[str, Any]:
    attachments = [
        {
            "original_attachment_url": url,
            "canonical_attachment_url": normalize_url(url),
            "filename": _filename_from_attachment_url(url),
        }
        for url in _attachment_urls(candidate.metadata)
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "rebuild_id": options.output_dir.name,
        "classification_manifest_hash": classification_manifest_hash,
        "classification_effective_date": options.classification_effective_date,
        "processing_effective_date": options.processing_effective_date,
        "preprocess_version": PREPROCESS_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "quality_gate_version": QUALITY_GATE_VERSION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_version": _tokenizer_version(),
        "url_normalization_version": URL_NORMALIZATION_VERSION,
        "doc_id": row["doc_id"],
        "source_name": row["source_name"],
        "page_type": options.page_type,
        "document_type": row.get("document_type") or None,
        "domain": row.get("domain") or None,
        "department": row.get("department") or None,
        "title": row["title"],
        "source_url": row["source_url"],
        "normalized_source_url": normalized_source_url,
        "canonical_doc_key": canonical_doc_key,
        "canonical_doc_key_hash": canonical_doc_key_hash,
        "initial_filter_status": row["recommended_action"],
        "filter_reason": row.get("filter_reason") or None,
        "confidence": row.get("confidence") or None,
        "review_status": review_status,
        "review_reason": review_reason,
        "source_content_hash": source_content_hash,
        "prepared_body_hash": prepared_body_hash,
        "published_at": row.get("published_at_metadata") or candidate.metadata.get("published_at"),
        "valid_until": row.get("relevant_end_date") or row.get("title_period_end") or None,
        "attachments": attachments,
    }
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def _document_quality(body: str, metadata: dict[str, Any], *, date_blocked: bool) -> tuple[str, list[str], str]:
    if date_blocked:
        return "FAILED", [], "classification and processing effective dates differ for dated document"
    normalized = " ".join(body.split())
    if not normalized:
        return "FAILED", [], "empty prepared body"
    if not re.search(r"(?m)^#\s+\S+", body):
        return "FAILED", [], "missing H1"
    if re.search(r"(?m)^#\s*(공지사항|자료실|FAQ)\s*$", body):
        return "FAILED", [], "generic H1 remains"
    lowered = body.casefold()
    if "downloadbbsfile" in lowered or "javascript:" in lowered:
        return "FAILED", [], "unsafe URL remains in body"
    warnings: list[str] = []
    mojibake_status, mojibake_code = _mojibake_signal(body)
    if mojibake_status == "FAILED":
        return "FAILED", [mojibake_code], "suspicious broken Korean encoding pattern detected"
    if mojibake_status == "WARNING":
        return "WARNING", [mojibake_code], "abnormal character ratio requires manual review"
    if len(normalized) < 180:
        if _has_contact(body):
            warnings.append("LOW_TEXT_BUT_HAS_CONTACT")
        elif metadata.get("attachments"):
            warnings.append("LOW_TEXT_BUT_HAS_ATTACHMENT_NAME")
        else:
            return "FAILED", [], "too short without searchable signal"
    return ("WARNING" if warnings else "PASSED"), warnings, ""


def _make_chunks(
    *,
    body: str,
    metadata: dict[str, Any],
    prepared_body_hash: str,
    prepared_artifact_hash: str,
    preview: bool,
) -> list[dict[str, Any]]:
    sections = _sections(body)
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for section_title, section_path, section_text in sections:
        prefix = _chunk_prefix(metadata["title"], section_path)
        max_content_chars = max(240, MAX_EMBEDDING_CHARS - len(prefix))
        max_content_tokens = max(120, MAX_EMBEDDING_TOKENS - _token_count(prefix))
        for piece in _split_long_text(section_text, max_chars=max_content_chars, max_tokens=max_content_tokens):
            content = piece.strip()
            embedding_text = _chunk_text(metadata["title"], section_path, content)
            chunk_text_hash = _sha256_text(embedding_text)
            chunk = {
                "chunk_id": _chunk_id(metadata["canonical_doc_key"], prepared_body_hash, chunk_index),
                "doc_id": metadata["doc_id"],
                "canonical_doc_key": _canonical_doc_key_parts(metadata),
                "canonical_doc_key_hash": metadata["canonical_doc_key_hash"],
                "source_name": metadata["source_name"],
                "source_url": metadata["source_url"],
                "title": metadata["title"],
                "section_title": section_title,
                "section_path": section_path,
                "chunk_index": chunk_index,
                "content": content,
                "embedding_text": embedding_text,
                # Deprecated compatibility alias. New indexing code should read embedding_text.
                "text": embedding_text,
                "embedding_char_count": len(embedding_text),
                "embedding_token_count": _token_count(embedding_text),
                "char_count": len(embedding_text),
                "token_count": _token_count(embedding_text),
                "content_char_count": len(content),
                "content_token_count": _token_count(content),
                "chunk_text_hash": chunk_text_hash,
                "page_type": metadata.get("page_type"),
                "document_type": metadata.get("document_type"),
                "domain": metadata.get("domain"),
                "department": metadata.get("department"),
                "published_at": metadata.get("published_at"),
                "valid_until": metadata.get("valid_until"),
                "section_kind": _section_kind(section_title, content),
                "is_time_sensitive": _is_time_sensitive_section(section_title, content),
                "chunk_valid_until": metadata.get("valid_until") if _is_time_sensitive_section(section_title, content) else None,
                "processing_effective_date": metadata.get("processing_effective_date"),
                "initial_filter_status": metadata.get("initial_filter_status"),
                "review_status": metadata.get("review_status"),
                "document_quality_status": metadata.get("document_quality_status", "NOT_CHECKED"),
                "chunk_quality_status": "NOT_CHECKED",
                "embedding_eligibility": "NOT_ELIGIBLE" if preview else "PENDING",
                "warning_codes": [],
                "chunking_version": CHUNKING_VERSION,
                "quality_gate_version": QUALITY_GATE_VERSION,
                "tokenizer_name": TOKENIZER_NAME,
                "tokenizer_version": _tokenizer_version(),
                "prepared_body_hash": prepared_body_hash,
                "prepared_artifact_hash": prepared_artifact_hash,
                "preview": preview,
            }
            chunks.append(chunk)
            chunk_index += 1
    chunks = _merge_short_chunks(chunks, metadata["canonical_doc_key"], prepared_body_hash)
    return _enforce_embedding_limits(chunks, metadata["canonical_doc_key"], prepared_body_hash)


def _merge_short_chunks(
    chunks: list[dict[str, Any]],
    canonical_doc_key: str,
    prepared_body_hash: str,
) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(chunks):
        chunk = dict(chunks[index])
        should_merge_forward = (
            index + 1 < len(chunks)
            and (
                _is_label_only_chunk(chunk)
                or _is_question_only_chunk(chunk)
                or (chunk["char_count"] < 180 and not _is_semantic_chunk(chunk))
            )
        )
        if should_merge_forward:
            next_chunk = chunks[index + 1]
            chunk["content"] = f"{chunk['content'].rstrip()}\n\n{next_chunk['content'].strip()}"
            if str(chunk["section_title"]) == "본문" and _is_semantic_chunk(next_chunk):
                chunk["section_title"] = next_chunk["section_title"]
                chunk["section_path"] = next_chunk["section_path"]
            else:
                chunk["section_title"] = (
                    str(chunk["section_title"])
                    if chunk["section_title"] == next_chunk["section_title"]
                    else f"{chunk['section_title']} / {next_chunk['section_title']}"
                )
                chunk["section_path"] = _merge_section_paths(chunk["section_path"], next_chunk["section_path"])
            _refresh_chunk_text(chunk)
            index += 2
        else:
            index += 1
        merged.append(chunk)
    for new_index, chunk in enumerate(merged):
        _refresh_chunk_identity(chunk, canonical_doc_key, prepared_body_hash, new_index)
    return merged


def _chunk_quality(chunks: list[dict[str, Any]]) -> tuple[str, str, list[str], str, str]:
    if not chunks:
        return "FAILED", "FAILED", [], "no chunks created", "BLOCKED_BY_QUALITY"
    aggregate_warnings: list[str] = []
    for chunk in chunks:
        text = chunk["embedding_text"]
        failure_reason = _chunk_failure_reason(chunk)
        if failure_reason:
            status, warning, reason, eligibility = failure_reason
            chunk["chunk_quality_status"] = status
            chunk["embedding_eligibility"] = eligibility
            chunk["warning_codes"] = [warning] if warning else []
            return "FAILED", status, [warning] if warning else [], reason, eligibility
        chunk_warnings: list[str] = []
        if "첨부파일:" in text and len(" ".join(text.split())) < 220:
            return "FAILED", "FAILED", ["ATTACHMENT_NAME_ONLY"], "attachment name only chunk is not publishable", "BLOCKED_BY_ATTACHMENT"
        if chunk["char_count"] < 180:
            if _has_contact(text):
                chunk_warnings.append("SHORT_BUT_COMPLETE")
            elif _is_semantic_chunk(chunk):
                chunk_warnings.append("SHORT_BUT_COMPLETE")
            elif len(chunks) == 1:
                chunk_warnings.append("SINGLE_SECTION_DOCUMENT")
            elif chunk["content_char_count"] >= 40:
                chunk_warnings.append("SHORT_BUT_COMPLETE")
            else:
                chunk["chunk_quality_status"] = "FAILED"
                chunk["embedding_eligibility"] = "BLOCKED_BY_QUALITY"
                chunk["warning_codes"] = []
                return "FAILED", "FAILED", [], "chunk below min size", "BLOCKED_BY_QUALITY"
        unique_chunk_warnings = sorted(set(chunk_warnings))
        if unique_chunk_warnings and not _all_allowed(unique_chunk_warnings, ALLOWED_CHUNK_WARNINGS):
            chunk["chunk_quality_status"] = "WARNING"
            chunk["embedding_eligibility"] = "BLOCKED_BY_QUALITY"
            chunk["warning_codes"] = unique_chunk_warnings
            return "FAILED", "WARNING", unique_chunk_warnings, "unallowed chunk warning", "BLOCKED_BY_QUALITY"
        chunk["chunk_quality_status"] = "WARNING" if unique_chunk_warnings else "PASSED"
        chunk["embedding_eligibility"] = "ELIGIBLE"
        chunk["warning_codes"] = unique_chunk_warnings
        aggregate_warnings.extend(unique_chunk_warnings)
    unique_warnings = sorted(set(aggregate_warnings))
    return "VALIDATED", "WARNING" if unique_warnings else "PASSED", unique_warnings, "", "ELIGIBLE"


def _drop_attachment_only_chunks_when_possible(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks
    filtered = [chunk for chunk in chunks if not _is_attachment_name_only_chunk(chunk)]
    if not filtered:
        return chunks
    for new_index, chunk in enumerate(filtered):
        _refresh_chunk_identity(chunk, _canonical_json(chunk["canonical_doc_key"]), chunk["prepared_body_hash"], new_index)
    return filtered


def _drop_noninformative_chunks_when_possible(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks
    filtered = [
        chunk
        for chunk in chunks
        if _has_contact(str(chunk.get("content") or "")) or (not _is_label_only_chunk(chunk) and not _is_question_only_chunk(chunk))
    ]
    if not filtered:
        return chunks
    for new_index, chunk in enumerate(filtered):
        _refresh_chunk_identity(chunk, _canonical_json(chunk["canonical_doc_key"]), chunk["prepared_body_hash"], new_index)
    return filtered


def _drop_expired_time_sensitive_chunks_when_possible(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks
    filtered = [chunk for chunk in chunks if not _is_expired_time_sensitive_chunk(chunk)]
    if not filtered:
        return chunks
    for new_index, chunk in enumerate(filtered):
        _refresh_chunk_identity(chunk, _canonical_json(chunk["canonical_doc_key"]), chunk["prepared_body_hash"], new_index)
    return filtered


def _is_attachment_name_only_chunk(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("embedding_text") or "")
    return "첨부파일:" in text and len(" ".join(text.split())) < 220


def _chunk_failure_reason(chunk: dict[str, Any]) -> tuple[str, str, str, str] | None:
    text = str(chunk.get("embedding_text") or "")
    content = str(chunk.get("content") or "")
    if _has_pagination_noise(text) or _has_navigation_menu_noise(text):
        return "FAILED", "PAGINATION_NOISE_REMAINING", "pagination or menu noise remains", "BLOCKED_BY_QUALITY"
    if _has_attachment_placeholder(content):
        return "FAILED", "ATTACHMENT_PLACEHOLDER_ONLY", "attachment placeholder remains in chunk", "BLOCKED_BY_ATTACHMENT"
    if chunk["embedding_char_count"] > MAX_EMBEDDING_CHARS or chunk["embedding_token_count"] > MAX_EMBEDDING_TOKENS:
        return "FAILED", "", "chunk exceeds max size", "BLOCKED_BY_QUALITY"
    if _is_label_only_chunk(chunk):
        return "FAILED", "LABEL_ONLY_CHUNK", "label-only chunk is not publishable", "BLOCKED_BY_QUALITY"
    if _is_question_only_chunk(chunk):
        return "FAILED", "QUESTION_ONLY_CHUNK", "question-only chunk is not publishable", "BLOCKED_BY_QUALITY"
    if _is_expired_time_sensitive_chunk(chunk):
        return "FAILED", "EXPIRED_TIME_SENSITIVE_CHUNK", "expired time-sensitive chunk is not publishable", "BLOCKED_BY_QUALITY"
    return None


def _enforce_embedding_limits(
    chunks: list[dict[str, Any]],
    canonical_doc_key: str,
    prepared_body_hash: str,
) -> list[dict[str, Any]]:
    limited: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk["embedding_char_count"] <= MAX_EMBEDDING_CHARS and chunk["embedding_token_count"] <= MAX_EMBEDDING_TOKENS:
            limited.append(chunk)
            continue
        prefix = _chunk_prefix(chunk["title"], chunk["section_path"])
        max_content_chars = max(240, MAX_EMBEDDING_CHARS - len(prefix))
        max_content_tokens = max(120, MAX_EMBEDDING_TOKENS - _token_count(prefix))
        for piece in _split_long_text(chunk["content"], max_chars=max_content_chars, max_tokens=max_content_tokens):
            next_chunk = dict(chunk)
            next_chunk["content"] = piece.strip()
            _refresh_chunk_text(next_chunk)
            limited.append(next_chunk)
    for new_index, chunk in enumerate(limited):
        _refresh_chunk_identity(chunk, canonical_doc_key, prepared_body_hash, new_index)
    return limited


def _refresh_chunk_text(chunk: dict[str, Any]) -> None:
    chunk["section_kind"] = _section_kind(str(chunk.get("section_title") or ""), str(chunk.get("content") or ""))
    chunk["is_time_sensitive"] = _is_time_sensitive_section(str(chunk.get("section_title") or ""), str(chunk.get("content") or ""))
    chunk["chunk_valid_until"] = chunk.get("valid_until") if chunk["is_time_sensitive"] else None
    chunk["embedding_text"] = _chunk_text(str(chunk["title"]), list(chunk["section_path"]), str(chunk["content"]))
    chunk["text"] = chunk["embedding_text"]


def _refresh_chunk_identity(chunk: dict[str, Any], canonical_doc_key: str, prepared_body_hash: str, chunk_index: int) -> None:
    _refresh_chunk_text(chunk)
    chunk["chunk_index"] = chunk_index
    chunk["chunk_id"] = _chunk_id(canonical_doc_key, prepared_body_hash, chunk_index)
    chunk["embedding_char_count"] = len(chunk["embedding_text"])
    chunk["embedding_token_count"] = _token_count(chunk["embedding_text"])
    chunk["char_count"] = chunk["embedding_char_count"]
    chunk["token_count"] = chunk["embedding_token_count"]
    chunk["content_char_count"] = len(chunk["content"])
    chunk["content_token_count"] = _token_count(chunk["content"])
    chunk["chunk_text_hash"] = _sha256_text(chunk["embedding_text"])


def _section_kind(section_title: str, content: str) -> str:
    combined = f"{section_title}\n{content}"
    if _is_time_sensitive_section(section_title, content):
        return "TIME_SENSITIVE_ACTION"
    if "첨부파일:" in combined:
        return "ATTACHMENT_INFO"
    if _has_contact(combined):
        return "CONTACT"
    return "STABLE_REFERENCE"


def _is_time_sensitive_section(section_title: str, content: str) -> bool:
    combined = " ".join(f"{section_title}\n{content}".split())
    return any(term in combined for term in TIME_SENSITIVE_TERMS) and bool(TIME_SENSITIVE_DATE_PATTERN.search(combined))


def _is_expired_time_sensitive_chunk(chunk: dict[str, Any]) -> bool:
    if not chunk.get("is_time_sensitive"):
        return False
    valid_until = _parse_iso_date(str(chunk.get("chunk_valid_until") or chunk.get("valid_until") or ""))
    processing_date = _parse_iso_date(str(chunk.get("processing_effective_date") or ""))
    return bool(valid_until and processing_date and valid_until < processing_date)


def _parse_iso_date(value: str) -> date | None:
    if not value:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0))
    except ValueError:
        return None


def _has_pagination_noise(text: str) -> bool:
    compact = " ".join(str(text).split())
    return any(pattern.search(compact) for pattern in PAGINATION_NOISE_PATTERNS)


def _has_navigation_menu_noise(text: str) -> bool:
    compact = " ".join(str(text).split())
    return any(term in compact for term in NAVIGATION_MENU_TERMS)


def _has_attachment_placeholder(text: str) -> bool:
    return any(_is_attachment_placeholder(line.strip().lstrip("- ").strip()) for line in str(text).splitlines())


def _is_label_only_chunk(chunk: dict[str, Any]) -> bool:
    content = " ".join(str(chunk.get("content") or "").split())
    if not content:
        return True
    if _has_contact(content):
        return False
    normalized = re.sub(r"^[\d.)\s]+", "", content).strip()
    return len(normalized) <= 24 and bool(_semantic_label_title(normalized))


def _is_question_only_chunk(chunk: dict[str, Any]) -> bool:
    content = " ".join(str(chunk.get("content") or "").split())
    if not content or len(content) > 160:
        return False
    if not _faq_question_title(content):
        return False
    answer_markers = ("A", "A.", "답", "답변", "처리", "가능", "신청", "제출", "문의")
    remainder = content.rstrip("?").strip()
    return not any(marker in remainder for marker in answer_markers)


def _sections(body: str) -> list[tuple[str, list[str], str]]:
    lines = body.splitlines()
    title = ""
    sections: list[tuple[str, list[str], list[str]]] = []
    current_title = "본문"
    current_path = ["본문"]
    current_lines: list[str] = []
    for line in lines:
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            heading = match.group(2).strip()
            if match.group(1) == "#":
                title = heading
                current_path = [heading, "본문"]
                continue
            if current_lines:
                sections.append((current_title, current_path, current_lines))
            current_title = heading
            current_path = [title, heading] if title else [heading]
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_path, current_lines))
    if not sections:
        section_title = title or "본문"
        sections = [(section_title, [section_title], [body])]
    expanded: list[tuple[str, list[str], str]] = []
    for section_title, section_path, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        expanded.extend(_semantic_subsections(section_title, section_path, section_text))
    return expanded


def _semantic_subsections(section_title: str, section_path: list[str], section_text: str) -> list[tuple[str, list[str], str]]:
    lines = section_text.splitlines()
    starts = _semantic_section_starts(lines)
    if not starts:
        return [(section_title, section_path, section_text)]

    sections: list[tuple[str, list[str], str]] = []
    if starts[0][0] > 0:
        intro = "\n".join(lines[: starts[0][0]]).strip()
        if intro:
            sections.append((section_title, section_path, intro))

    for index, (line_index, title) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(lines)
        content = "\n".join(lines[line_index:end]).strip()
        if not content:
            continue
        path = [part for part in [*section_path, title] if part]
        sections.append((title, _dedupe_path(path), content))
    return sections


def _semantic_section_starts(lines: list[str]) -> list[tuple[int, str]]:
    starts: list[tuple[int, str]] = []
    seen_indexes: set[int] = set()
    for index, line in enumerate(lines):
        stripped = " ".join(line.strip().split())
        if not stripped:
            continue
        title = _faq_question_title(stripped) or _semantic_label_title(stripped)
        if not title or index in seen_indexes:
            continue
        starts.append((index, title))
        seen_indexes.add(index)
    return starts


def _faq_question_title(line: str) -> str | None:
    if re.match(r"^(Q|Q\.|Q\d+|문)\s*[:.)：]\s*.+", line, flags=re.IGNORECASE):
        return _trim_section_title(line)
    if len(line) <= 120 and line.endswith("?") and not line.startswith(("-", "*")):
        return _trim_section_title(line)
    return None


def _semantic_label_title(line: str) -> str | None:
    normalized = re.sub(r"^[\-*•\d.)\s]+", "", line).strip()
    normalized = normalized.strip("<>[]【】")
    label_patterns = (
        r"^(신청|접수)\s*(방법|절차|기간|대상)",
        r"^(지원|신청)\s*(자격|대상|조건)",
        r"^(제출|구비)\s*(서류|자료)",
        r"^(선발|지급)\s*(기준|방법|대상)",
        r"^(유의|주의)\s*사항",
        r"^(문의|문의처|담당자|연락처)",
        r"^(일정|행사\s*기간|운영\s*기간)",
        r"^(필수\s*제출|선택\s*제출)",
    )
    if any(re.match(pattern, normalized) for pattern in label_patterns):
        return _trim_section_title(normalized)
    return None


def _is_semantic_chunk(chunk: dict[str, Any]) -> bool:
    title = str(chunk.get("section_title") or "")
    candidates = [part.strip() for part in title.split("/") if part.strip()]
    candidates.append(title)
    return any(_faq_question_title(candidate) or _semantic_label_title(candidate) for candidate in candidates)


def _trim_section_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value[:120]


def _dedupe_path(path: list[str]) -> list[str]:
    result: list[str] = []
    for part in path:
        if part and part not in result:
            result.append(part)
    return result


def preprocess_markdown_body(*, body: str, title: str, attachment_urls: list[str]) -> str:
    lines = body.splitlines()
    lines = _trim_menu_to_actual_detail(lines, title)
    lines = _promote_heading_after_generic_heading(lines)
    lines = _remove_attachment_url_lines(lines, attachment_urls)
    lines = _remove_navigation_noise_lines(lines)
    lines = _trim_navigation_tail(lines)
    lines = _normalize_author_lines(lines)
    lines = _drop_duplicate_title_after_h1(lines, title)
    content = _collapse_blank_lines(lines)
    content = _ensure_attachment_filename_section(content, attachment_urls)
    content = _remove_attachment_placeholder_lines(content)
    return content.strip()


def _trim_menu_to_actual_detail(lines: list[str], title: str) -> list[str]:
    generic_titles = {"공지사항", "자료실", "faq", "# 공지사항", "# 자료실", "# faq"}
    title_is_generic = _normalize_title(title) in {_normalize_title(value) for value in generic_titles}
    for index, line in enumerate(lines):
        stripped = line.strip()
        normalized = _normalize_title(stripped)
        if not stripped or stripped.startswith(("*", "_", "|")):
            continue
        if normalized in {_normalize_title(value) for value in generic_titles}:
            continue
        if not (stripped.startswith("[") or stripped.startswith("【") or title_is_generic):
            continue
        lookahead = "\n".join(lines[index + 1 : index + 5])
        if not re.search(r"_?작성자_?|작성일|등록일", lookahead):
            continue
        detail_title = re.sub(r"^#{1,6}\s*", "", stripped).strip()
        return [f"# {detail_title}", *lines[index + 1 :]]
    return lines


def _promote_heading_after_generic_heading(lines: list[str]) -> list[str]:
    generic = {_normalize_title(value) for value in ("공지사항", "자료실", "FAQ")}
    for index, line in enumerate(lines):
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if not match or _normalize_title(match.group(1)) not in generic:
            continue
        for next_index in range(index + 1, min(index + 6, len(lines))):
            next_match = re.match(r"^#{1,6}\s+(.+)$", lines[next_index].strip())
            if not next_match:
                continue
            actual_title = next_match.group(1).strip()
            if _normalize_title(actual_title) in generic:
                continue
            return lines[:index] + [f"# {actual_title}"] + lines[next_index + 1 :]
    return lines


def _remove_attachment_url_lines(lines: list[str], attachment_urls: list[str]) -> list[str]:
    attachment_set = {url.strip() for url in attachment_urls if url}
    cleaned: list[str] = []
    skip_attachment_url_block = False
    for line in lines:
        stripped = line.strip()
        lowered = stripped.casefold()
        if "첨부파일 다운로드 url" in lowered or "다운로드 url" in lowered:
            skip_attachment_url_block = True
            continue
        if skip_attachment_url_block and (stripped.startswith("- http") or stripped.startswith("http")):
            continue
        skip_attachment_url_block = False
        if "downloadbbsfile.do" in lowered:
            continue
        if stripped in attachment_set or stripped.lstrip("- ").strip() in attachment_set:
            continue
        cleaned.append(line)
    return cleaned


def _remove_navigation_noise_lines(lines: list[str]) -> list[str]:
    noise_exact = {"이전글", "다음글", "목록", "공지사항", "자료실", "게시물 검색"}
    cleaned: list[str] = []
    for line in lines:
        stripped = " ".join(line.strip().split())
        if stripped in noise_exact or stripped == "야야야":
            continue
        if _has_pagination_noise(stripped):
            continue
        if stripped in NAVIGATION_MENU_TERMS:
            continue
        if re.match(r"^총게시물\s*:\s*_?\d+_?\s*건\s*페이지\s*:\s*_?\d+_?\s*/\s*\d+\s*$", stripped):
            continue
        if stripped in {"링크:", "링크"}:
            continue
        cleaned.append(line)
    return cleaned


def _trim_navigation_tail(lines: list[str]) -> list[str]:
    tail_markers = ("목록보기 본문출력", "본문출력 < 이전글다음글", "부서: 소개", "부서:소개")
    for index, line in enumerate(lines):
        stripped = " ".join(line.strip().split())
        if any(marker in stripped for marker in tail_markers):
            return lines[:index]
    return lines


def _normalize_author_lines(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(작성자|작성부서|담당부서|부서)\s*[:：]?\s+(.+)$", stripped)
        if match:
            normalized.append(f"{match.group(1)}: {' '.join(match.group(2).split())}")
            continue
        normalized.append(line)
    return normalized


def _drop_duplicate_title_after_h1(lines: list[str], title: str) -> list[str]:
    if not lines:
        return lines
    h1_index = next((index for index, line in enumerate(lines) if line.strip().startswith("# ")), None)
    if h1_index is None:
        return lines
    h1_title = _normalize_title(lines[h1_index])
    next_index = h1_index + 1
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    if next_index < len(lines) and _normalize_title(lines[next_index]) in {h1_title, _normalize_title(title)}:
        return lines[:next_index] + lines[next_index + 1 :]
    return lines


def _collapse_blank_lines(lines: list[str]) -> str:
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        if not line.strip():
            if compacted and not previous_blank:
                compacted.append("")
            previous_blank = True
            continue
        compacted.append(line.strip())
        previous_blank = False
    return "\n".join(compacted)


def _ensure_attachment_filename_section(content: str, attachment_urls: list[str]) -> str:
    filenames = [_filename_from_attachment_url(url) for url in attachment_urls]
    filenames = [filename for filename in filenames if filename and not _is_attachment_placeholder(filename) and filename not in content]
    if not filenames:
        return content
    lines = [content.rstrip(), "", "첨부파일:"]
    lines.extend(f"- {filename}" for filename in filenames)
    return "\n".join(lines)


def _filename_from_attachment_url(url: str) -> str | None:
    path_name = Path(url.split("?", 1)[0]).name
    if path_name and "." in path_name and not path_name.casefold().startswith("downloadbbsfile"):
        return path_name[:300]
    query_match = re.search(r"(?:atchmnflNo|fileNo|fileId)=([^&]+)", url, flags=re.IGNORECASE)
    if query_match:
        return f"첨부파일-{query_match.group(1)}"
    return None


def _remove_attachment_placeholder_lines(content: str) -> str:
    lines: list[str] = []
    pending_attachment_heading = False
    kept_attachment_item = False
    for line in content.splitlines():
        stripped = line.strip()
        if _is_attachment_heading(stripped):
            pending_attachment_heading = True
            kept_attachment_item = False
            continue
        if pending_attachment_heading and stripped.startswith("- "):
            item = stripped[2:].strip()
            if _is_attachment_placeholder(item):
                continue
            if not kept_attachment_item:
                lines.append("첨부파일:")
            lines.append(line)
            kept_attachment_item = True
            continue
        if pending_attachment_heading:
            pending_attachment_heading = False
            kept_attachment_item = False
        lines.append(line)
    return _collapse_blank_lines(lines)


def _is_attachment_heading(value: str) -> bool:
    return value in {"첨부파일:", "첨부파일"}


def _is_attachment_placeholder(value: str) -> bool:
    return bool(re.fullmatch(r"첨부파일-?\d+", value.strip(), flags=re.IGNORECASE))


def _title_from_body(body: str) -> str | None:
    generic = {_normalize_title(value) for value in ("공지사항", "자료실", "FAQ")}
    for line in body.splitlines():
        stripped = line.strip()
        match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if match and _normalize_title(match.group(1)) not in generic:
            return match.group(1).strip()
    return None


def _normalize_title(value: str) -> str:
    value = re.sub(r"^#{1,6}\s*", "", value or "").strip()
    return re.sub(r"\s+", " ", value).casefold()


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1].strip(), parts[2].strip()


def _split_long_text(text: str, *, max_chars: int, max_tokens: int = 650) -> list[str]:
    if len(text) <= max_chars and _token_count(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    current = ""
    for block in re.split(r"(\n\s*\n)", text):
        if not block.strip():
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block.strip()
        if len(candidate) <= max_chars and _token_count(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = block.strip()
        while len(current) > max_chars or _token_count(current) > max_tokens:
            split_at = _find_split_point(current, max_chars=max_chars, max_tokens=max_tokens)
            parts.append(current[:split_at].strip())
            current = current[split_at:].strip()
    if current:
        parts.append(current)
    return parts


def _find_split_point(text: str, *, max_chars: int, max_tokens: int) -> int:
    hard_limit = min(max_chars, len(text))
    while hard_limit > 200 and _token_count(text[:hard_limit]) > max_tokens:
        hard_limit = int(hard_limit * 0.85)
    for marker in ("\n", ". ", " "):
        split_at = text.rfind(marker, 0, hard_limit)
        if split_at >= 180:
            return split_at + (1 if marker != " " else 0)
    return max(180, hard_limit)


def _chunk_text(title: str, section_path: list[str], text: str) -> str:
    return f"{_chunk_prefix(title, section_path)}{text.strip()}"


def _chunk_prefix(title: str, section_path: list[str]) -> str:
    section_label = " > ".join(part for part in section_path if part and part != title) or "본문"
    return f"문서제목: {title}\n구간: {section_label}\n\n"


def _chunk_id(canonical_doc_key: str, prepared_body_hash: str, chunk_index: int) -> str:
    return _sha256_text(
        _canonical_json(
            [
                canonical_doc_key,
                CHUNKING_VERSION,
                prepared_body_hash,
                chunk_index,
            ]
        )
    )


def _canonical_doc_key_parts(metadata: dict[str, Any]) -> list[str]:
    return [str(metadata["source_name"]), str(metadata["normalized_source_url"])]


def _public_prepared_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    public_metadata = dict(metadata)
    public_metadata.pop("normalized_source_url", None)
    return public_metadata


def _merge_section_paths(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for part in [*left, *right]:
        if part and part not in merged:
            merged.append(part)
    if len(merged) > 2 and "본문" in merged:
        merged = [part for part in merged if part != "본문"]
    return merged


def _read_review_decisions(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {
            (row.get("doc_id", "").strip(), row.get("canonical_doc_key", "").strip()): {key: (value or "").strip() for key, value in row.items()}
            for row in rows
        }


def _snapshot_or_template_review_decisions(tmp_root: Path, source: Path | None) -> None:
    target = tmp_root / "manifest" / "review_decisions.csv"
    if source and source.exists():
        target.write_text(source.read_text(encoding="utf-8-sig"), encoding="utf-8", newline="\n")
        return
    _write_csv(
        target,
        [],
        fieldnames=["doc_id", "canonical_doc_key", "decision", "reviewer", "reviewed_at", "source_content_hash", "classification_manifest_hash", "reason"],
    )


def _review_queue_row(
    base_report: dict[str, str],
    review_status: str,
    review_reason: str,
    source_content_hash: str,
    classification_manifest_hash: str,
) -> dict[str, str]:
    return {
        **base_report,
        "review_status": review_status,
        "review_reason": review_reason,
        "source_content_hash": source_content_hash,
        "classification_manifest_hash": classification_manifest_hash,
    }


def _empty_result(mapping: dict[str, Any], processing: dict[str, Any], document_quality: dict[str, Any], chunk_quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "mapping": mapping,
        "processing": processing,
        "document_quality": document_quality,
        "chunk_quality": chunk_quality,
        "review_queue": None,
        "published_chunks": [],
        "prepared_artifact_hash": None,
    }


def _finished_no_chunks(
    mapping: dict[str, Any],
    processing: dict[str, Any],
    document_quality: dict[str, Any],
    chunk_quality: dict[str, Any],
    prepared_artifact_hash: str,
) -> dict[str, Any]:
    return {
        "mapping": mapping,
        "processing": processing,
        "document_quality": document_quality,
        "chunk_quality": chunk_quality,
        "review_queue": None,
        "published_chunks": [],
        "prepared_artifact_hash": prepared_artifact_hash,
    }


def _prepared_markdown_path(*, tmp_root: Path, row: dict[str, str], canonical_doc_key_hash: str, title: str, manual: bool) -> Path:
    base = tmp_root / ("manual_review/final" if manual else "final") / row["source_name"]
    return base / f"{canonical_doc_key_hash}_{_safe_filename(title)}.md"


def _serialize_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
    return _lf(f"---\n{frontmatter}\n---\n\n{body.strip()}\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_lf(text), encoding="utf-8", newline="\n")


def _read_text(path: Path) -> str:
    return _lf(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, f"{_canonical_json(value)}\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _add_context(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**row, **context} for row in rows]


def _canonical_doc_key(source_name: str, normalized_source_url: str) -> str:
    return _canonical_json([source_name, normalized_source_url])


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_body(body: str) -> str:
    return _sha256_text(_normalize_hash_text(body))


def _metadata_hash(metadata: dict[str, Any]) -> str:
    excluded = {"metadata_hash", "prepared_artifact_hash", "processed_at", "absolute_path", "tmp_run_id"}
    return _sha256_json({key: value for key, value in metadata.items() if key not in excluded})


def _normalize_hash_text(value: str) -> str:
    return _lf(value).strip()


def _lf(value: str) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return (value or "document")[:48]


def _attachment_urls(metadata: dict[str, Any]) -> list[str]:
    return [str(url) for url in metadata.get("attachment_urls") or [] if url]


def _has_contact(text: str) -> bool:
    return bool(re.search(r"[\w.-]+@[\w.-]+|\b0\d{1,2}-\d{3,4}-\d{4}\b", text))


def _mojibake_signal(text: str) -> tuple[str, str]:
    if "\ufffd" in text:
        return "FAILED", "UNICODE_REPLACEMENT_CHARACTER_FOUND"
    suspicious_patterns = (
        "臾몄",
        "援ш",
        "蹂몃",
        "泥⑤",
        "寃쎄",
        "?쒕",
        "?덈",
        "?숈",
        "?묒",
        "?좎",
    )
    pattern_hits = sum(text.count(pattern) for pattern in suspicious_patterns)
    question_hangul_noise = len(re.findall(r"\?[가-힣]", text))
    question_marks = text.count("?")
    cjk_noise = len(re.findall(r"[一-龥豈-﫿]", text))
    hangul = len(re.findall(r"[가-힣]", text))
    normalized_len = max(1, len("".join(text.split())))
    if pattern_hits >= 2 or question_hangul_noise >= 6:
        return "FAILED", "KNOWN_MOJIBAKE_PATTERN_FOUND"
    if question_marks >= 25 and question_marks / normalized_len >= 0.08:
        return "WARNING", "ABNORMAL_CHARACTER_RATIO"
    if cjk_noise >= 12 and cjk_noise > max(4, hangul * 2):
        return "WARNING", "ABNORMAL_CHARACTER_RATIO"
    return "NONE", ""


def _has_mojibake(text: str) -> bool:
    return _mojibake_signal(text)[0] in {"FAILED", "WARNING"}


def _all_allowed(warnings: Iterable[str], allowed: set[str]) -> bool:
    return all(warning in allowed for warning in warnings)


def _token_count(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding(TOKENIZER_NAME).encode(text))
    except Exception:
        return max(1, int(len(text) / 2.2 + 0.999))


def _tokenizer_version() -> str:
    try:
        import tiktoken

        return getattr(tiktoken, "__version__", "unknown")
    except Exception:
        return "fallback-len-div-2.2"


def _parse_args() -> PrepareOptions:
    parser = argparse.ArgumentParser(description="Prepare existing Markdown dumps for offline chunk publishing.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", default=["KEEP", "MANUAL_REVIEW"])
    parser.add_argument("--classification-effective-date", required=True)
    parser.add_argument("--processing-effective-date", required=True)
    parser.add_argument("--review-decisions-csv", type=Path)
    parser.add_argument("--source-name")
    parser.add_argument("--page-type", default="LIST_PAGE")
    parser.add_argument("--document-type")
    parser.add_argument("--manual-review-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dump-report", action="store_true")
    parser.add_argument("--preprocess", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quality-gate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dump-final", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dump-chunks", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    _validate_date(args.classification_effective_date, "--classification-effective-date")
    _validate_date(args.processing_effective_date, "--processing-effective-date")
    return PrepareOptions(
        input_dir=args.input_dir,
        classification_csv=args.classification_csv,
        output_dir=args.output_dir,
        actions=tuple(args.actions),
        classification_effective_date=args.classification_effective_date,
        processing_effective_date=args.processing_effective_date,
        review_decisions_csv=args.review_decisions_csv,
        source_name=args.source_name,
        page_type=args.page_type,
        document_type=args.document_type,
        manual_review_only=args.manual_review_only,
        force=args.force,
        limit=args.limit,
        sample_size=args.sample_size,
        keep_staging=args.keep_staging,
        dry_run=args.dry_run,
        dump_report=args.dump_report,
    )


def _validate_date(value: str, name: str) -> None:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD") from exc


def main() -> None:
    result = prepare_markdown(_parse_args())
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
