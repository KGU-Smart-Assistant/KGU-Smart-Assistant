from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from app.schemas import Document

SOURCE_TYPE_PRIORITY = {
    "html": 6,
    "markdown": 5,
    "pdf": 4,
    "docx": 3,
    "hwp": 3,
    "hwpx": 3,
    "zip": 2,
    "image": 2,
    "file": 1,
    "notice": 0,
}


@dataclass
class DocumentDedupResult:
    documents: List[Document]
    total_input: int
    total_output: int
    exact_duplicates_removed: int
    version_duplicates_removed: int


def select_latest_documents(documents: List[Document]) -> DocumentDedupResult:
    """Remove duplicate documents and keep the latest/best version."""
    if not documents:
        return DocumentDedupResult(
            documents=[],
            total_input=0,
            total_output=0,
            exact_duplicates_removed=0,
            version_duplicates_removed=0,
        )

    exact_groups: Dict[Tuple[str, str, str, str], List[Document]] = {}
    for document in documents:
        exact_groups.setdefault(_build_exact_key(document), []).append(document)

    exact_survivors: List[Document] = []
    exact_duplicates_removed = 0

    for group in exact_groups.values():
        sorted_group = sorted(group, key=_document_rank_key, reverse=True)
        exact_survivors.append(sorted_group[0])
        exact_duplicates_removed += max(0, len(sorted_group) - 1)

    version_groups: Dict[Tuple[str, str, str], List[Document]] = {}
    for document in exact_survivors:
        version_groups.setdefault(_build_version_key(document), []).append(document)

    deduplicated: List[Document] = []
    version_duplicates_removed = 0

    for group in version_groups.values():
        kept_documents: List[Document] = []
        for candidate in sorted(group, key=_document_rank_key, reverse=True):
            matched_document = next(
                (existing for existing in kept_documents if _is_same_document(existing, candidate)),
                None,
            )
            if matched_document is not None:
                version_duplicates_removed += 1
                continue
            kept_documents.append(candidate)

        deduplicated.extend(kept_documents)

    deduplicated.sort(key=_document_rank_key, reverse=True)

    return DocumentDedupResult(
        documents=deduplicated,
        total_input=len(documents),
        total_output=len(deduplicated),
        exact_duplicates_removed=exact_duplicates_removed,
        version_duplicates_removed=version_duplicates_removed,
    )


def _build_exact_key(document: Document) -> Tuple[str, str, str, str]:
    return (
        _normalize_title(document.title),
        _normalize_value(document.category),
        _normalize_value(document.department),
        _hash_text(_normalize_content(document.content)),
    )


def _build_version_key(document: Document) -> Tuple[str, str, str]:
    return (
        _normalize_title(document.title),
        _normalize_value(document.category),
        _normalize_value(document.department),
    )


def _is_same_document(left: Document, right: Document) -> bool:
    if _normalize_title(left.title) != _normalize_title(right.title):
        return False

    left_content = _normalize_content(left.content)
    right_content = _normalize_content(right.content)

    if left_content == right_content:
        return True

    minimum_length = min(len(left_content), len(right_content))
    if minimum_length < 200:
        return False

    shorter, longer = sorted((left_content, right_content), key=len)
    if shorter and shorter in longer and len(shorter) / len(longer) >= 0.85:
        return True

    similarity = SequenceMatcher(a=left_content, b=right_content).ratio()
    if similarity >= 0.92:
        return True

    left_path = urlparse(left.source_url).path.lower()
    right_path = urlparse(right.source_url).path.lower()
    same_path = left_path and left_path == right_path
    return same_path and similarity >= 0.75


def _document_rank_key(document: Document) -> Tuple[int, datetime, int, int, str]:
    published_at = document.published_at or datetime.min
    collected_at = document.collected_at
    source_priority = SOURCE_TYPE_PRIORITY.get(document.source_type, 0)
    content_length = len(_normalize_content(document.content))
    return (
        1 if document.published_at else 0,
        published_at,
        source_priority,
        content_length,
        collected_at.isoformat(),
    )


def _normalize_title(value: str) -> str:
    lowered = value.casefold()
    return re.sub(r"\s+", " ", lowered).strip()


def _normalize_content(value: str) -> str:
    lowered = value.casefold()
    lowered = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _normalize_value(value: str | None) -> str:
    return (value or "").casefold().strip()


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()
