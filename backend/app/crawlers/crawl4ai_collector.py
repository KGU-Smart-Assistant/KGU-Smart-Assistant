import asyncio
import hashlib
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

from app.crawlers.docling_collector import (
    DoclingCollectorConfig,
    collect_documents_with_docling,
)
from app.crawlers.parsing.parser_router import ParserRouter
from app.crawlers.parsing.schemas import ParseContext
from app.schemas import Document

DOCUMENT_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".hwp": "hwp",
    ".hwpx": "hwpx",
    ".zip": "zip",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".gif": "image",
    ".webp": "image",
}

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
    ".hwp",
    ".hwpx",
    ".zip",
)

DEFAULT_EXCLUDE_PATTERNS = (
    "login",
    "logout",
    "signup",
    "search.do",
    "javascript:",
    "mailto:",
)

PARSER_ROUTER = ParserRouter()


@dataclass
class Crawl4AICollectorConfig:
    seed_urls: List[str]
    max_pages: int = 20
    max_depth: int = 2
    max_pagination_pages: int = 200
    category: Optional[str] = None
    department: Optional[str] = None
    include_patterns: Tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS
    follow_patterns: Optional[Tuple[str, ...]] = None
    collect_patterns: Optional[Tuple[str, ...]] = None
    exclude_patterns: Tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    allowed_domains: Optional[Set[str]] = None
    allowed_path_prefixes: Optional[Tuple[str, ...]] = None
    headless: bool = True
    word_count_threshold: int = 1
    page_timeout_ms: int = 30000
    collect_seed_pages: bool = True
    allowed_keyword_filters: Optional[Tuple[str, ...]] = None
    blocked_keyword_filters: Optional[Tuple[str, ...]] = None
    allowed_author_department_filters: Optional[Tuple[str, ...]] = None
    blocked_author_department_filters: Optional[Tuple[str, ...]] = None
    min_published_at: Optional[datetime] = None
    docling_config: DoclingCollectorConfig = field(default_factory=DoclingCollectorConfig)


def collect_documents_with_crawl4ai(
    config: Crawl4AICollectorConfig,
) -> List[Document]:
    """Discover same-domain pages and document links starting from seed URLs."""
    return asyncio.run(_collect_documents_with_crawl4ai(config))


async def _collect_documents_with_crawl4ai(
    config: Crawl4AICollectorConfig,
) -> List[Document]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    except ImportError as exc:
        raise RuntimeError(
            "crawl4ai is not installed. Add 'crawl4ai' to dependencies and install it first."
        ) from exc

    allowed_domains = config.allowed_domains or {
        _normalize_domain(urlparse(seed).netloc) for seed in config.seed_urls
    }

    browser_channel = os.getenv("CRAWL4AI_BROWSER_CHANNEL")
    browser_config_kwargs = {
        "headless": config.headless,
        "java_script_enabled": True,
    }
    if browser_channel:
        browser_config_kwargs["channel"] = browser_channel
        browser_config_kwargs["chrome_channel"] = browser_channel

    browser_config = BrowserConfig(
        **browser_config_kwargs,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=config.word_count_threshold,
        page_timeout=config.page_timeout_ms,
    )

    queue: Deque[Tuple[str, int]] = deque((_normalize_url(url), 0) for url in config.seed_urls)
    visited_html_urls: Set[str] = set()
    collected_doc_urls: Set[str] = set()
    attachment_parent_titles: Dict[str, str] = {}
    documents: List[Document] = []
    collected_at = datetime.now()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        while queue and _within_page_limit(visited_html_urls, config.max_pages):
            current_url, depth = queue.popleft()
            if current_url in visited_html_urls:
                continue
            if depth != 0 and not _is_allowed_url(current_url, allowed_domains, config):
                continue

            result = await crawler.arun(url=current_url, config=run_config)
            visited_html_urls.add(current_url)

            if not getattr(result, "success", False):
                continue

            should_collect_html = _should_collect_html_url(
                url=current_url,
                config=config,
                is_seed=depth == 0,
            )
            html_document: Optional[Document] = None
            if should_collect_html:
                html_document = _build_html_document(
                    url=current_url,
                    result=result,
                    category=config.category,
                    department=config.department,
                    collected_at=collected_at,
                    allowed_keyword_filters=config.allowed_keyword_filters,
                    blocked_keyword_filters=config.blocked_keyword_filters,
                    allowed_author_department_filters=config.allowed_author_department_filters,
                    blocked_author_department_filters=config.blocked_author_department_filters,
                    min_published_at=config.min_published_at,
                )
                if html_document is not None:
                    documents.append(html_document)
                    for attachment_url in html_document.attachment_urls:
                        attachment_parent_titles.setdefault(
                            attachment_url,
                            html_document.title,
                        )

            discovered_html_urls, discovered_doc_urls = _extract_links(
                base_url=current_url,
                result_links=getattr(result, "links", {}) or {},
                allowed_domains=allowed_domains,
                config=config,
            )
            if not _should_follow_discovered_links(
                url=current_url,
                result=result,
                config=config,
            ):
                discovered_html_urls = set()
                discovered_doc_urls = set()
            elif config.min_published_at is not None and _is_board_list_url(current_url):
                discovered_html_urls = {
                    next_url
                    for next_url in discovered_html_urls
                    if not _is_board_list_url(next_url)
                }
            if should_collect_html and html_document is None:
                discovered_doc_urls = set()
            discovered_html_urls.update(
                _extract_pagination_urls(
                    url=current_url,
                    result=result,
                    allowed_domains=allowed_domains,
                    config=config,
                )
            )
            collected_doc_urls.update(discovered_doc_urls)

            if depth >= config.max_depth:
                continue

            for next_url in sorted(discovered_html_urls, key=_html_url_priority):
                if next_url not in visited_html_urls:
                    queue.append((next_url, depth + 1))

    if collected_doc_urls:
        docling_config = config.docling_config
        docling_config.category = config.category or docling_config.category
        docling_config.department = config.department or docling_config.department
        attachment_documents = collect_documents_with_docling(
            sources=sorted(collected_doc_urls),
            config=docling_config,
        )
        _apply_attachment_parent_titles(attachment_documents, attachment_parent_titles)
        documents.extend(attachment_documents)

    return documents


def _apply_attachment_parent_titles(
    documents: List[Document],
    parent_titles: Dict[str, str],
) -> None:
    for document in documents:
        base_source_url = document.source_url.split("#", 1)[0]
        parent_title = parent_titles.get(base_source_url)
        if parent_title:
            document.title = parent_title


def _within_page_limit(visited_html_urls: Set[str], max_pages: int) -> bool:
    return max_pages <= 0 or len(visited_html_urls) < max_pages


def _build_html_document(
    url: str,
    result,
    category: Optional[str],
    department: Optional[str],
    collected_at: datetime,
    allowed_keyword_filters: Optional[Tuple[str, ...]] = None,
    blocked_keyword_filters: Optional[Tuple[str, ...]] = None,
    allowed_author_department_filters: Optional[Tuple[str, ...]] = None,
    blocked_author_department_filters: Optional[Tuple[str, ...]] = None,
    min_published_at: Optional[datetime] = None,
) -> Optional[Document]:
    parsed = PARSER_ROUTER.parse(
        result=result,
        context=ParseContext(
            url=url,
            category=category,
            department=department,
            allowed_keyword_filters=allowed_keyword_filters,
            blocked_keyword_filters=blocked_keyword_filters,
            allowed_author_department_filters=allowed_author_department_filters,
            blocked_author_department_filters=blocked_author_department_filters,
        ),
    )
    if parsed is None:
        return None
    if (
        min_published_at is not None
        and parsed.published_at is not None
        and parsed.published_at < min_published_at
    ):
        return None
    if (
        min_published_at is not None
        and parsed.published_at is None
        and _is_board_detail_url(url)
    ):
        return None

    return Document(
        doc_id=_build_doc_id(url),
        source_type="html",
        source_url=url,
        title=parsed.title,
        content=parsed.content,
        category=category,
        department=department,
        author_department=parsed.author_department,
        published_at=parsed.published_at,
        attachment_urls=parsed.attachment_urls,
        collected_at=collected_at,
    )


def _extract_links(
    base_url: str,
    result_links: Dict[str, List[Dict]],
    allowed_domains: Set[str],
    config: Crawl4AICollectorConfig,
) -> Tuple[Set[str], Set[str]]:
    html_urls: Set[str] = set()
    doc_urls: Set[str] = set()

    for link in result_links.get("internal", []):
        href = (link or {}).get("href")
        if not href:
            continue

        normalized = _normalize_url(urljoin(base_url, href))
        if not _is_allowed_url(normalized, allowed_domains, config):
            continue

        if _looks_like_document_url(normalized):
            doc_urls.add(normalized)
        else:
            html_urls.add(normalized)

    return html_urls, doc_urls


def _extract_pagination_urls(
    url: str,
    result,
    allowed_domains: Set[str],
    config: Crawl4AICollectorConfig,
) -> Set[str]:
    if not _is_board_list_url(url):
        return set()
    if not _should_follow_discovered_links(url=url, result=result, config=config):
        return set()

    last_page = _extract_last_page_index(result)
    if last_page is None or last_page <= 1:
        return set()

    if config.max_pagination_pages > 0:
        last_page = min(last_page, config.max_pagination_pages)
    if config.min_published_at is not None:
        current_page = _current_page_index(url)
        if current_page >= last_page:
            return set()
        page_url = _with_query_param(url, "pageIndex", str(current_page + 1))
        if _is_allowed_url(page_url, allowed_domains, config):
            return {_normalize_url(page_url)}
        return set()

    pagination_urls: Set[str] = set()
    for page_index in range(1, last_page + 1):
        page_url = _with_query_param(url, "pageIndex", str(page_index))
        if _is_allowed_url(page_url, allowed_domains, config):
            pagination_urls.add(_normalize_url(page_url))
    return pagination_urls


def _should_follow_discovered_links(
    *,
    url: str,
    result,
    config: Crawl4AICollectorConfig,
) -> bool:
    if not _is_board_list_url(url):
        return True
    if config.min_published_at is None:
        return True
    if not _has_board_detail_links(result):
        return False

    dates = _extract_dates_from_result(result)
    if not dates:
        return True
    return max(dates) >= config.min_published_at


def _has_board_detail_links(result) -> bool:
    for link_group in (getattr(result, "links", {}) or {}).values():
        for link in link_group:
            href = ((link or {}).get("href") or "").lower()
            if "selectbbsnttview.do" in href:
                return True
    return False


def _extract_dates_from_result(result) -> List[datetime]:
    markdown = getattr(result, "markdown", None)
    content = (
        getattr(markdown, "fit_markdown", None)
        or getattr(markdown, "raw_markdown", None)
        or ""
    )
    dates: List[datetime] = []
    patterns = (
        r"([0-9]{4})[./-]([0-9]{1,2})[./-]([0-9]{1,2})",
        r"([0-9]{4})\s*년\s*([0-9]{1,2})\s*월\s*([0-9]{1,2})\s*일",
    )
    for pattern in patterns:
        for year, month, day in re.findall(pattern, content):
            try:
                dates.append(datetime(int(year), int(month), int(day)))
            except ValueError:
                continue
    return dates


def _current_page_index(url: str) -> int:
    parsed = urlparse(url)
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.casefold() == "pageindex" and value.isdigit():
            return int(value)
    return 1


def _is_board_list_url(url: str) -> bool:
    return "selectbbsnttlist.do" in url.lower()


def _is_board_detail_url(url: str) -> bool:
    return "selectbbsnttview.do" in url.lower()


def _extract_last_page_index(result) -> Optional[int]:
    markdown = getattr(result, "markdown", None)
    content = (
        getattr(markdown, "fit_markdown", None)
        or getattr(markdown, "raw_markdown", None)
        or ""
    )
    total_page_match = re.search(r"페이지\s*:\s*_?\d+_?\s*/\s*_?(\d+)_?", content)
    if total_page_match:
        return int(total_page_match.group(1))

    page_indexes: Set[int] = set()
    for link_group in (getattr(result, "links", {}) or {}).values():
        for link in link_group:
            href = (link or {}).get("href") or ""
            parsed = urlparse(href)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if key.casefold() == "pageindex" and value.isdigit():
                    page_indexes.add(int(value))
    if page_indexes:
        return max(page_indexes)
    return None


def _with_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query_items = [
        (item_key, item_value)
        for item_key, item_value in parse_qsl(parsed.query, keep_blank_values=True)
        if item_key.casefold() != key.casefold()
    ]
    query_items.append((key, value))
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def _looks_like_document_url(url: str) -> bool:
    lowered = url.lower()
    return "downloadbbsfile.do" in lowered or any(ext in lowered for ext in DOCUMENT_EXTENSIONS)


def _should_collect_html_url(
    url: str,
    config: Crawl4AICollectorConfig,
    is_seed: bool,
) -> bool:
    if is_seed and not config.collect_seed_pages:
        return False

    patterns = config.collect_patterns or config.include_patterns
    if not patterns:
        return True

    lowered = url.lower()
    return any(pattern in lowered for pattern in patterns)


def _html_url_priority(url: str) -> Tuple[int, str]:
    lowered = url.lower()
    if "selectbbsnttview.do" in lowered:
        return (0, lowered)
    if "selectbbsnttlist.do" in lowered:
        return (1, lowered)
    if "contents.do" in lowered:
        return (2, lowered)
    return (3, lowered)


def _is_allowed_url(
    url: str,
    allowed_domains: Set[str],
    config: Crawl4AICollectorConfig,
) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False

    domain = _normalize_domain(parsed.netloc)
    if allowed_domains and domain not in allowed_domains:
        return False

    lowered = url.lower()
    if any(pattern in lowered for pattern in config.exclude_patterns):
        return False

    if _looks_like_document_url(url):
        return True

    if config.allowed_path_prefixes:
        normalized_path = parsed.path if parsed.path.endswith("/") else f"{parsed.path}/"
        if not any(normalized_path.startswith(prefix) for prefix in config.allowed_path_prefixes):
            return False

    patterns = config.follow_patterns or config.include_patterns
    if not patterns:
        return True

    return any(pattern in lowered for pattern in patterns)


def _normalize_domain(domain: str) -> str:
    normalized = domain.lower().strip()
    if normalized.startswith("www."):
        return normalized[4:]
    return normalized


def _normalize_url(url: str) -> str:
    clean_url, _ = urldefrag(url)
    parsed = urlparse(clean_url)
    normalized_params = ""
    if parsed.params and not re.fullmatch(r"jsessionid=[^/?#]+", parsed.params, flags=re.IGNORECASE):
        normalized_params = parsed.params
    normalized_query = _normalize_query_for_path(parsed.path, parsed.query)
    rebuilt = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            normalized_params,
            normalized_query,
            parsed.fragment,
        )
    )
    return rebuilt.rstrip("/")


def _normalize_query_for_path(path: str, query: str) -> str:
    query_items = parse_qsl(query, keep_blank_values=True)
    lowered_path = path.lower()
    if "selectbbsnttview.do" in lowered_path:
        return _canonical_board_detail_query(query_items)
    if "selectbbsnttlist.do" in lowered_path:
        return _canonical_board_list_query(query_items)
    return urlencode(sorted(query_items))


def _canonical_board_detail_query(query_items: List[Tuple[str, str]]) -> str:
    allowed_keys = {
        "bbsno",
        "key",
        "nttno",
        "selfat",
    }
    canonical_items: Dict[str, Tuple[str, str]] = {}
    for key, value in query_items:
        lowered_key = key.casefold()
        if lowered_key not in allowed_keys:
            continue
        canonical_items.setdefault(lowered_key, (key, value))
    return urlencode([canonical_items[key] for key in sorted(canonical_items)])


def _canonical_board_list_query(query_items: List[Tuple[str, str]]) -> str:
    allowed_keys = {
        "bbsno",
        "key",
        "pageindex",
        "selfat",
        "sf.dc",
        "sf.dcof",
        "sf.of1",
        "sf.of2",
    }
    canonical_items: Dict[str, Tuple[str, str]] = {}
    for key, value in query_items:
        lowered_key = key.casefold()
        if lowered_key not in allowed_keys:
            continue
        canonical_items.setdefault(lowered_key, (key, value))
    return urlencode([canonical_items[key] for key in sorted(canonical_items)])


def _build_doc_id(source_url: str) -> str:
    digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:12]
    return f"crawl-{digest}"
