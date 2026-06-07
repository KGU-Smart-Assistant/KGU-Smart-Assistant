from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.schemas import Document, DocumentChunk


CHUNKER_VERSION = "markdown-semantic-v1"
EMBEDDING_VERSION = "v1"
EMBEDDING_TEMPLATE_VERSION = "metadata-prefix-v1"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}

UNSTABLE_QUERY_PARAMS = {
    "page",
    "pageindex",
    "pageno",
    "searchcnd",
    "searchwrd",
    "session",
    "sessionid",
    "timestamp",
    "token",
}

STABLE_ID_QUERY_PARAMS = {"bbsno", "nttno", "key"}


def normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered in TRACKING_QUERY_PARAMS:
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items, key=lambda item: (item[0].casefold(), item[1])), doseq=True)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            query,
            "",
        )
    )


def normalize_stable_url(url: str) -> str | None:
    parsed = urlparse(normalize_url(url))
    if not parsed.scheme or not parsed.netloc:
        return None

    query = {
        key.casefold(): value
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_QUERY_PARAMS
        and key.casefold() not in UNSTABLE_QUERY_PARAMS
    }
    path = parsed.path.casefold()
    stable_items: list[tuple[str, str]] = []
    if "selectbbsnttview.do" in path:
        for key in ("bbsno", "nttno", "key"):
            if query.get(key):
                stable_items.append((key, query[key]))
        if not (query.get("bbsno") and query.get("nttno")):
            return None
    elif "contents.do" in path:
        if not query.get("key"):
            return None
        stable_items.append(("key", query["key"]))
    else:
        meaningful_query_keys = set(query)
        if meaningful_query_keys - STABLE_ID_QUERY_PARAMS:
            return None
        stable_items = sorted(query.items())

    stable_query = urlencode(sorted(stable_items), doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", stable_query, ""))


def canonical_doc_key(document: Document, *, system: str, domain: str | None, department: str | None) -> str:
    normalized_url = normalize_url(document.source_url)
    parsed = urlparse(normalized_url)
    query = {
        key.casefold(): value
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    }
    path = parsed.path.casefold()
    if "selectbbsnttview.do" in path and query.get("bbsno") and query.get("nttno"):
        return f"kyonggi_bbs:{query['bbsno']}:{query['nttno']}"

    stable_url = normalize_stable_url(normalized_url)
    if stable_url:
        return f"url:{system}:{sha256_text(stable_url)}"

    scoped_hash = content_hash(document.title, document.content)
    return "content_hash:{system}:{domain}:{department}:{hash}".format(
        system=system,
        domain=domain or "unknown",
        department=department or "unknown",
        hash=scoped_hash,
    )


def canonical_doc_id(canonical_key: str) -> str:
    return f"crawl-{sha256_text(canonical_key)[:24]}"


def source_system(source: dict[str, Any] | str | None, source_url: str | None = None) -> str:
    source_name = source.get("name", "") if isinstance(source, dict) else str(source or "")
    url = source_url or ""
    lowered = f"{source_name} {url}".casefold()
    if "job.kyonggi.ac.kr" in lowered or "career" in lowered:
        return "career_portal"
    if "library" in lowered:
        return "library"
    if "kyonggi.ac.kr" in lowered or source_name:
        return "kyonggi"
    return "unknown"


def canonicalize_documents_and_chunks(
    *,
    source: dict[str, Any],
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> tuple[list[Document], list[DocumentChunk]]:
    old_to_new_identity: dict[str, tuple[str, str]] = {}
    canonical_documents: list[Document] = []
    system = source_system(source)
    source_domain = source.get("domain") or source.get("category")
    for document in documents:
        domain = document.domain or source_domain
        department = document.department or source.get("department")
        key = canonical_doc_key(document, system=system, domain=domain, department=department)
        doc_id = canonical_doc_id(key)
        fingerprint = index_fingerprint(content_hash_value=content_hash(document.title, document.content))
        old_to_new_identity[document.doc_id] = (doc_id, fingerprint)
        canonical_documents.append(document.model_copy(update={"doc_id": doc_id}))

    canonical_chunks: list[DocumentChunk] = []
    for chunk in chunks:
        doc_id, fingerprint = old_to_new_identity.get(
            chunk.doc_id,
            (chunk.doc_id, index_fingerprint(content_hash_value=sha256_text(chunk.text))),
        )
        chunk_payload = chunk.model_dump()
        chunk_payload["doc_id"] = doc_id
        chunk_payload["chunk_id"] = vector_point_id(
            doc_id=doc_id,
            index_fingerprint_value=fingerprint,
            chunk_index=chunk.chunk_index,
        )
        canonical_chunks.append(DocumentChunk(**chunk_payload))
    return canonical_documents, canonical_chunks


def content_hash(title: str, content: str) -> str:
    return sha256_text(f"{normalize_text(title)}\n{normalize_text(content)}")


def metadata_hash(*, domains: Iterable[str | None], departments: Iterable[str | None], source_urls: Iterable[str]) -> str:
    metadata = {
        "domains": sorted({value for value in domains if value}),
        "departments": sorted({value for value in departments if value}),
        "source_urls": sorted({normalize_url(value) for value in source_urls if value}),
    }
    return sha256_json(metadata)


def attachment_hash(attachments: Iterable[dict[str, Any]]) -> str:
    normalized = []
    for attachment in attachments:
        normalized.append(
            {
                "attachment_url": normalize_url(str(attachment.get("attachment_url") or "")),
                "filename": attachment.get("filename"),
                "file_type": attachment.get("file_type"),
            }
        )
    return sha256_json(sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)))


def index_fingerprint(
    *,
    content_hash_value: str,
    embedding_model: str | None = None,
    chunker_version: str = CHUNKER_VERSION,
    embedding_version: str = EMBEDDING_VERSION,
    embedding_template_version: str = EMBEDDING_TEMPLATE_VERSION,
) -> str:
    payload = {
        "content_hash": content_hash_value,
        "chunker_version": chunker_version,
        "embedding_model": embedding_model or DEFAULT_EMBEDDING_MODEL,
        "embedding_template_version": embedding_template_version,
        "embedding_version": embedding_version,
    }
    return sha256_json(payload)


def vector_point_id(*, doc_id: str, index_fingerprint_value: str, chunk_index: int) -> str:
    return sha256_text(f"{doc_id}:{index_fingerprint_value}:{chunk_index}")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
