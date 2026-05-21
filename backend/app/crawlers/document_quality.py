from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List

from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.schemas import Document

ATTACHMENT_SOURCE_TYPES = {"pdf", "hwp", "hwpx", "docx", "file", "image"}
MIN_CONTENT_CHARS = 80
MIN_UNIQUE_TOKENS = 8
LOW_INFORMATION_MAX_CHARS = 500
MIN_SEARCHABLE_CHUNK_CHARS = 30

NAVIGATION_NOISE_MARKERS = (
    "로그인",
    "회원가입",
    "개인정보처리방침",
    "사이트맵",
    "통합검색",
    "추천검색어",
    "인기 검색어",
    "go to top",
    "copyright",
)

LOW_VALUE_CHUNK_PATTERNS = (
    re.compile(r"^\s*(내역|목록|첨부파일|붙임|별첨)\s*$", re.IGNORECASE),
    re.compile(r"^\s*[A-Z]\s+.+(신청서|확인서|동의서|서약서|양식)\s*$"),
)

GARBLED_MARKERS = (
    "\ufffd",
    "\uf071",
    "\uf09e",
    "쀀",
    "쓉",
    "�",
)

MOJIBAKE_MARKERS = (
    "濡",
    "媛",
    "瑗",
    "吏",
    "湲",
    "??",
)


@dataclass(frozen=True)
class DocumentQualityResult:
    documents: List[Document]
    total_input: int
    total_output: int
    removed_short: int
    removed_low_information: int
    removed_navigation_noise: int
    attachment_link_fallbacks: int = 0


def filter_quality_documents(documents: Iterable[Document]) -> DocumentQualityResult:
    kept: List[Document] = []
    removed_short = 0
    removed_low_information = 0
    removed_navigation_noise = 0
    attachment_link_fallbacks = 0
    input_documents = list(documents)

    for document in input_documents:
        normalized = normalize_document_text(document)
        if _is_listing_noise(document, normalized):
            removed_navigation_noise += 1
            continue
        if should_use_attachment_link_fallback(document, normalized):
            kept.append(
                document.model_copy(
                    update={"content": build_attachment_fallback_text(document)}
                )
            )
            attachment_link_fallbacks += 1
            continue
        if _is_navigation_noise(document.content) or _is_navigation_noise(normalized):
            removed_navigation_noise += 1
            continue
        if len(normalized) < MIN_CONTENT_CHARS:
            removed_short += 1
            continue
        if (
            len(normalized) < LOW_INFORMATION_MAX_CHARS
            and _unique_token_count(normalized) < MIN_UNIQUE_TOKENS
        ):
            removed_low_information += 1
            continue
        if normalized == " ".join(document.content.split()):
            kept.append(document)
        else:
            kept.append(document.model_copy(update={"content": normalized}))

    return DocumentQualityResult(
        documents=kept,
        total_input=len(input_documents),
        total_output=len(kept),
        removed_short=removed_short,
        removed_low_information=removed_low_information,
        removed_navigation_noise=removed_navigation_noise,
        attachment_link_fallbacks=attachment_link_fallbacks,
    )


def normalize_document_text(document: Document) -> str:
    if "원문 링크:" in document.content and "첨부파일 본문을 안정적으로 추출하지 못했습니다" in document.content:
        return " ".join(document.content.split())
    cleaned = clean_crawled_markdown(document.content, source_url=document.source_url)
    return " ".join(cleaned.split())


def should_use_attachment_link_fallback(
    document: Document,
    normalized_text: str | None = None,
) -> bool:
    if document.source_type in ATTACHMENT_SOURCE_TYPES:
        return True
    if not document.attachment_urls:
        return False
    normalized = normalized_text or normalize_document_text(document)
    return _is_download_link_page(normalized)


def build_attachment_fallback_text(document: Document) -> str:
    attachment_urls = "\n".join(f"- {url}" for url in document.attachment_urls)
    if attachment_urls:
        attachment_urls = f"\n다운로드 URL:\n{attachment_urls}"
    return (
        "첨부파일 본문을 안정적으로 추출하지 못했습니다. 원문 링크 또는 다운로드 URL에서 직접 확인하세요.\n"
        f"제목: {sanitize_title(document.title)}\n"
        f"원문 링크: {document.source_url}"
        f"{attachment_urls}"
    )
    return (
        "첨부파일 본문을 안정적으로 추출하지 못했습니다. "
        "원문 링크에서 파일을 직접 확인하세요.\n"
        f"제목: {sanitize_title(document.title)}\n"
        f"원문 링크: {document.source_url}"
    )


def sanitize_title(title: str) -> str:
    cleaned = re.sub(r"\*\s*\[HOME\]\([^)]+\)", "", title, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\]\(https?://[^)]+\)", "", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned or cleaned.casefold() == "home":
        return "제목 없음"
    return cleaned[:300]


def _is_download_link_page(text: str) -> bool:
    lowered = text.casefold()
    has_curriculum_context = any(
        keyword in text
        for keyword in (
            "교육과정",
            "교과과정",
            "이수체계도",
            "졸업",
            "전공",
        )
    )
    has_download_context = any(
        keyword in lowered
        for keyword in (
            "download",
            "preview",
            "다운로드",
            "미리보기",
            "바로가기",
        )
    )
    return has_curriculum_context and has_download_context


def is_searchable_chunk_text(text: str) -> bool:
    normalized = " ".join(text.split())
    if len(normalized) < MIN_SEARCHABLE_CHUNK_CHARS and _looks_like_low_value_short_text(
        normalized
    ):
        return False
    if any(pattern.match(normalized) for pattern in LOW_VALUE_CHUNK_PATTERNS):
        return False
    if is_garbled_text(normalized):
        return False
    if (
        len(normalized) < LOW_INFORMATION_MAX_CHARS
        and _unique_token_count(normalized) < 3
        and not re.fullmatch(r"[A-Za-z0-9_-]+", normalized)
        and not (normalized.endswith((".", "?", "!")) and _unique_token_count(normalized) >= 2)
    ):
        return False
    return True


def is_garbled_text(text: str) -> bool:
    if any(marker in text for marker in GARBLED_MARKERS):
        return True

    compact = "".join(text.split())
    if not compact:
        return False

    private_or_control = sum(
        1
        for char in compact
        if 0xE000 <= ord(char) <= 0xF8FF or (ord(char) < 32 and char not in "\n\t")
    )
    if private_or_control / len(compact) > 0.08:
        return True

    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    if marker_count >= 3 and len(text) < 1200:
        return True
    if text.count("?") >= 6 and text.count("?") / max(len(text), 1) > 0.03:
        return True
    if _looks_like_bad_cjk_noise(text):
        return True
    if _looks_like_bad_numeric_table_noise(text):
        return True
    return False


def _is_navigation_noise(text: str) -> bool:
    lowered = text.casefold()
    marker_count = sum(marker.casefold() in lowered for marker in NAVIGATION_NOISE_MARKERS)
    return marker_count >= 2 and len(text) < 1200


def _is_listing_noise(document: Document, normalized_text: str) -> bool:
    lowered_url = document.source_url.casefold()
    lowered_text = normalized_text.casefold()
    if _is_career_listing_or_calendar_url(lowered_url):
        return True
    if "contents.do?key=9346" in lowered_url:
        return True
    if "page=list" in lowered_url:
        return True
    if "book_idx=" in lowered_url:
        return True
    if "비밀번호 입력" in normalized_text and "비밀번호 확인" in normalized_text:
        return True
    if lowered_text.count("학사공지") >= 5 and lowered_text.count("|") >= 20:
        return True
    if lowered_text.count("대여불가") >= 5 and lowered_text.count("|") >= 20:
        return True
    return False


def _is_career_listing_or_calendar_url(lowered_url: str) -> bool:
    if "job.kyonggi.ac.kr" not in lowered_url:
        return False
    if "/calendar" in lowered_url:
        return True
    return "/list/" in lowered_url and "/view/" not in lowered_url


def _looks_like_low_value_short_text(text: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return False
    if text.endswith((".", "?", "!")) and _unique_token_count(text) >= 2:
        return False
    if _unique_token_count(text) >= 3:
        return False
    if len(text) >= 18 and _unique_token_count(text) >= 2:
        return False
    return True


def _looks_like_bad_cjk_noise(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 80:
        return False

    cjk = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    hangul = sum(1 for char in compact if "\uac00" <= char <= "\ud7a3")
    ascii_letters = sum(1 for char in compact if char.isascii() and char.isalpha())
    digits = sum(1 for char in compact if char.isdigit())
    informative = cjk + hangul + ascii_letters + digits
    if informative == 0:
        return False

    return cjk >= 20 and cjk > hangul * 1.5 and cjk / informative > 0.15


def _looks_like_bad_numeric_table_noise(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 150:
        return False

    cjk = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    hangul = sum(1 for char in compact if "\uac00" <= char <= "\ud7a3")
    digits = sum(1 for char in compact if char.isdigit())
    informative = cjk + hangul + digits
    if informative == 0:
        return False

    return hangul < 5 and cjk >= 5 and digits >= 120 and digits / informative > 0.75


def _unique_token_count(text: str) -> int:
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", text.casefold())
    return len(set(token for token in tokens if len(token) >= 2))
