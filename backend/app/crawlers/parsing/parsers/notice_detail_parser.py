import re
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.crawlers.parsing.parsers.base import BaseParser
from app.crawlers.parsing.schemas import ParseContext, ParsedDocument

DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".hwp",
    ".hwpx",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
)
SKIP_TITLE_TOKENS = ("二쇰찓", "닫기", "language", "login", "로그", "검색")
GENERIC_TITLE_TOKENS = ("공지사항 -", "자료실 -", "faq -")
METADATA_TITLE_PREFIXES = ("_작성자", "_작성일")
METADATA_LABEL_TOKENS = ("작성자", "등록일", "작성일", "부서", "담당부서")


class NoticeDetailParser(BaseParser):
    def parse(self, result, context: ParseContext) -> ParsedDocument | None:
        markdown = getattr(result, "markdown", None)
        raw_markdown = getattr(markdown, "fit_markdown", None) or getattr(
            markdown, "raw_markdown", None
        )
        if not raw_markdown:
            return None

        content = clean_crawled_markdown(raw_markdown, source_url=context.url)
        if not content:
            return None

        if self._should_skip_empty_faq_page(context.category, content):
            return None
        if context.category == "faq" and "selectbbsnttlist.do" in context.url.lower():
            return None

        title = self._extract_title_from_crawl_result(result, content)
        if title:
            title = self._sanitize_extracted_title(title)
        if (
            not title
            or title == "본문 바로가기"
            or self._looks_like_metadata_title(title)
            or self._matches_blocked_title_prefix(title, context)
        ):
            title = self._extract_fallback_title_from_content(content, context)
        if not title:
            return None

        parsed = ParsedDocument(
            title=title,
            content=content,
            published_at=self._extract_published_at(result, content),
            author_department=self._extract_author_department(result, content),
            attachment_urls=self._extract_attachment_urls(
                base_url=context.url,
                result_links=getattr(result, "links", {}) or {},
            ),
        )

        if context.allowed_keyword_filters and not self._matches_keyword_filters(
            parsed, context.allowed_keyword_filters, context.department
        ):
            return None
        if context.blocked_keyword_filters and self._matches_keyword_filters(
            parsed, context.blocked_keyword_filters, context.department
        ):
            return None
        if context.allowed_author_department_filters and not self._matches_author_department_filters(
            parsed.author_department,
            context.allowed_author_department_filters,
        ):
            return None
        if context.blocked_author_department_filters and self._matches_author_department_filters(
            parsed.author_department,
            context.blocked_author_department_filters,
        ):
            return None

        return parsed

    def _extract_title_from_crawl_result(self, result, content: str) -> Optional[str]:
        metadata = getattr(result, "metadata", None) or {}
        for key in ("title", "og:title"):
            value = metadata.get(key)
            if value:
                normalized = self._normalize_text(value)
                if (
                    normalized
                    and not self._should_skip_title(normalized)
                    and not self._looks_like_generic_board_title(normalized)
                ):
                    return normalized[:300]

        board_title_match = re.search(r"^\[.*?\]\s+(.+)$", content, re.MULTILINE)
        if board_title_match:
            normalized = self._normalize_title_candidate(board_title_match.group(1))
            if normalized and not self._should_skip_title(normalized):
                return normalized[:300]

        detail_title = self._extract_title_after_print_action(content)
        if detail_title:
            return detail_title[:300]

        for line in content.splitlines():
            normalized_line = line.strip()
            if normalized_line.startswith("_작성자") or normalized_line.startswith("_작성일"):
                continue
            if re.match(r"^!\[.*\]\(", normalized_line):
                continue

            normalized = self._normalize_text(re.sub(r"^#+\s*", "", line))
            if not normalized or self._should_skip_title(normalized):
                continue
            return normalized[:300]

        return None

    def _extract_published_at(self, result, content: str) -> Optional[datetime]:
        metadata = getattr(result, "metadata", None) or {}
        metadata_candidates = (
            metadata.get("article:published_time"),
            metadata.get("published_time"),
            metadata.get("published_at"),
            metadata.get("date"),
            metadata.get("created"),
        )
        for candidate in metadata_candidates:
            parsed = self._parse_datetime(candidate)
            if parsed is not None:
                return parsed

        line_patterns = (
            r"_?작성일_?\s*[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}(?:\s+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
            r"_?등록일_?\s*[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}(?:\s+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
            r"_?작성일_?\s*[:：]?\s*([0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일(?:\s*[0-9]{1,2}시\s*[0-9]{1,2}분(?:\s*[0-9]{1,2}초)?)?)",
            r"_?등록일_?\s*[:：]?\s*([0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일(?:\s*[0-9]{1,2}시\s*[0-9]{1,2}분(?:\s*[0-9]{1,2}초)?)?)",
        )
        for pattern in line_patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            parsed = self._parse_datetime(match.group(1))
            if parsed is not None:
                return parsed

        generic_patterns = (
            r"([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}(?:\s+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
            r"([0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일(?:\s*[0-9]{1,2}시\s*[0-9]{1,2}분(?:\s*[0-9]{1,2}초)?)?)",
        )
        for pattern in generic_patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            parsed = self._parse_datetime(match.group(1))
            if parsed is not None:
                return parsed

        return None

    def _extract_author_department(self, result, content: str) -> Optional[str]:
        metadata = getattr(result, "metadata", None) or {}
        for key in ("author", "department", "writer", "article:author"):
            candidate = metadata.get(key)
            normalized = self._normalize_text(str(candidate or ""))
            if normalized and not self._looks_like_metadata_label(normalized):
                return normalized[:200]

        line_patterns = (
            r"_?작성자_?\s*[:：]?\s*([^\n]+)",
            r"_?부서_?\s*[:：]?\s*([^\n]+)",
            r"_?담당부서_?\s*[:：]?\s*([^\n]+)",
        )
        for pattern in line_patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            normalized = self._normalize_text(match.group(1))
            if normalized and not self._looks_like_metadata_label(normalized):
                return normalized[:200]

        return None

    def _extract_attachment_urls(self, base_url: str, result_links: Dict[str, List[Dict]]) -> List[str]:
        attachment_urls: List[str] = []
        seen: Set[str] = set()
        for group in ("internal", "external"):
            for link in result_links.get(group, []):
                href = (link or {}).get("href")
                if not href:
                    continue
                normalized = urljoin(base_url, href)
                if normalized in seen or not self._looks_like_attachment_url(normalized):
                    continue
                seen.add(normalized)
                attachment_urls.append(normalized)
        return attachment_urls

    def _looks_like_attachment_url(self, url: str) -> bool:
        lowered = url.lower()
        return any(ext in lowered for ext in DOCUMENT_EXTENSIONS) or "downloadbbsfile.do" in lowered

    def _parse_datetime(self, value: object) -> Optional[datetime]:
        if not value:
            return None
        text = self._normalize_text(str(value))
        if not text:
            return None
        iso_candidate = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(iso_candidate)
        except ValueError:
            pass

        normalized = (
            text.replace("년", "-")
            .replace("월", "-")
            .replace("일", "")
            .replace("시", ":")
            .replace("분", ":")
            .replace("초", "")
            .replace(".", "-")
            .replace("/", "-")
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r":\s", ":", normalized)
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        for dt_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(normalized, dt_format)
            except ValueError:
                continue
        return None

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _normalize_title_candidate(self, value: str) -> str:
        normalized = self._normalize_text(value)
        normalized = re.sub(r"\s+이미지\s+\d+$", "", normalized)
        return normalized.strip()

    def _extract_title_after_print_action(self, content: str) -> Optional[str]:
        seen_print_action = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped in {"* 인쇄", "인쇄"}:
                seen_print_action = True
                continue
            if not seen_print_action:
                continue
            if not stripped or stripped.startswith("*") or stripped.startswith("["):
                continue
            normalized = self._sanitize_extracted_title(re.sub(r"^#+\s*", "", stripped))
            if (
                not normalized
                or self._should_skip_title(normalized)
                or self._looks_like_metadata_title(normalized)
            ):
                continue
            return normalized
        return None

    def _sanitize_extracted_title(self, value: str) -> str:
        normalized = self._normalize_text(value)
        normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized)
        normalized = re.sub(r"\]\(https?://[^)]+\)", "", normalized)
        return normalized.strip()

    def _extract_fallback_title_from_content(
        self,
        content: str,
        context: ParseContext,
    ) -> Optional[str]:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("![") or stripped.startswith("["):
                continue
            normalized = self._sanitize_extracted_title(re.sub(r"^#+\s*", "", stripped))
            if (
                not normalized
                or normalized == "본문 바로가기"
                or self._should_skip_title(normalized)
                or self._looks_like_metadata_title(normalized)
                or self._matches_blocked_title_prefix(normalized, context)
            ):
                continue
            return normalized[:300]
        return None

    def _should_skip_title(self, value: str) -> bool:
        lowered = value.lower()
        normalized = self._sanitize_extracted_title(value).strip("*_ ")
        navigation_titles = {
            "경기대학교",
            "입시홈페이지",
            "kutis",
            "통합검색",
            "로그인",
            "language",
            "주메뉴 열기",
        }
        if normalized.casefold() in navigation_titles:
            return True
        if normalized.startswith("주메뉴 열기"):
            return True
        return any(token in lowered for token in SKIP_TITLE_TOKENS)

    def _looks_like_generic_board_title(self, value: str) -> bool:
        lowered = value.casefold()
        return any(token in lowered for token in GENERIC_TITLE_TOKENS)

    def _looks_like_metadata_title(self, value: str) -> bool:
        normalized = self._normalize_text(value)
        return any(normalized.startswith(prefix) for prefix in METADATA_TITLE_PREFIXES)

    def _matches_blocked_title_prefix(self, value: str, context: ParseContext) -> bool:
        normalized = self._normalize_text(value)
        blocked_prefixes = context.parser_options.get("blocked_title_prefixes", [])
        return any(normalized.startswith(prefix) for prefix in blocked_prefixes)

    def _looks_like_metadata_label(self, value: str) -> bool:
        lowered = value.lower()
        return any(token in lowered for token in METADATA_LABEL_TOKENS)

    def _should_skip_empty_faq_page(self, category: Optional[str], content: str) -> bool:
        if category != "faq":
            return False
        normalized = self._normalize_text(content)
        return "총게시물 : 0" in normalized or "총게시물: 0" in normalized

    def _matches_keyword_filters(
        self,
        parsed: ParsedDocument,
        keywords: tuple[str, ...],
        department: Optional[str],
    ) -> bool:
        lowered_keywords = tuple(keyword.casefold() for keyword in keywords if keyword)
        if not lowered_keywords:
            return True
        candidates = (parsed.title, department, parsed.author_department)
        haystack = " ".join(self._normalize_text(str(value or "")) for value in candidates).casefold()
        return any(keyword in haystack for keyword in lowered_keywords)

    def _matches_author_department_filters(
        self,
        author_department: Optional[str],
        filters: tuple[str, ...],
    ) -> bool:
        lowered_filters = tuple(value.casefold() for value in filters if value)
        if not lowered_filters:
            return True
        normalized_author_department = self._normalize_text(str(author_department or "")).casefold()
        if not normalized_author_department:
            return False
        return any(value in normalized_author_department for value in lowered_filters)
