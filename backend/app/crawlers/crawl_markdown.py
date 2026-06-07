from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html as html_lib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

import yaml


CRAWL_SCHEMA_VERSION = "crawl4ai-markdown-dump-v1"
DEFAULT_USER_AGENT = "KGU-SmartAssistant-Crawler/1.0"
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRAWLED_MARKDOWN_ROOT = BACKEND_ROOT / "data" / "crawled_markdown"
DEFAULT_EXCLUDE_PATTERNS = ("login", "logout", "signup", "search.do", "javascript:", "mailto:", "tel:")
PAGE_TYPE_LIST = "LIST_PAGE"
PAGE_TYPE_SINGLE = "SINGLE_PAGE"
_PAGINATION_HINTS = (
    "page",
    "paging",
    "pagination",
    "pager",
    "페이지",
    "페이징",
    "페이 지",
)
_PAGINATION_NEXT_HINTS = (
    "next",
    "다음",
    "›",
    ">",
    "»",
)
_PAGINATION_LAST_HINTS = (
    "last",
    "final",
    "마지막",
    "끝",
)
REQUIRED_CLASSIFICATION_COLUMNS = [
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
]


@dataclass(frozen=True)
class CrawlSource:
    name: str
    urls: list[str]
    domain: str = ""
    department: str = ""
    max_depth: int = 0
    max_pages: int = 0
    max_pagination_pages: int = 0
    same_host_only: bool = True
    collect_seed_pages: bool = True
    follow_patterns: tuple[str, ...] = ()
    collect_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    allowed_path_prefixes: tuple[str, ...] = ()
    allowed_query_param_filters: tuple[tuple[tuple[str, str], ...], ...] = ()
    recommended_action: str = "MANUAL_REVIEW"
    page_type: str = PAGE_TYPE_LIST


@dataclass(frozen=True)
class CrawlOptions:
    sources: list[CrawlSource]
    output_dir: Path
    force: bool = False
    user_agent: str = DEFAULT_USER_AGENT
    wait_until: str = "domcontentloaded"
    page_timeout_ms: int = 45000
    cache_mode: str = "bypass"
    max_pages: int | None = None


class Crawl4AIRunner(Protocol):
    async def arun(self, url: str, config: Any = None) -> Any:
        ...


def crawl_markdown(options: CrawlOptions, *, crawler: Crawl4AIRunner | None = None) -> dict[str, Any]:
    return asyncio.run(crawl_markdown_async(options, crawler=crawler))


async def crawl_markdown_async(
    options: CrawlOptions,
    *,
    crawler: Crawl4AIRunner | None = None,
) -> dict[str, Any]:
    _prepare_output_dir(options.output_dir, force=options.force)
    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    queued_count = 0
    fetched_count = 0
    list_pages_fetched = 0
    detail_pages_fetched = 0
    single_pages_fetched = 0

    if crawler is not None:
        for source in options.sources:
            result = await _crawl_source_for_type(crawler, source, options, documents, failures, run_config=None)
            queued_count += result["queued"]
            fetched_count += result["fetched"]
            list_pages_fetched += result["list_pages_fetched"]
            detail_pages_fetched += result["detail_pages_fetched"]
            single_pages_fetched += result["single_pages_fetched"]
    else:
        async with _create_crawl4ai_runner(options) as owned_crawler:
            run_config = _create_run_config(options)
            for source in options.sources:
                result = await _crawl_source_for_type(owned_crawler, source, options, documents, failures, run_config=run_config)
                queued_count += result["queued"]
                fetched_count += result["fetched"]
                list_pages_fetched += result["list_pages_fetched"]
                detail_pages_fetched += result["detail_pages_fetched"]
                single_pages_fetched += result["single_pages_fetched"]

    _write_seed_csv(options.output_dir / "manifest" / "crawl_classification_seed.csv", documents)
    _write_json(
        options.output_dir / "manifest" / "crawl_report.json",
        {
            "schema_version": CRAWL_SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "source_count": len(options.sources),
            "queued": queued_count,
            "fetched": fetched_count,
            "list_pages_fetched": list_pages_fetched,
            "detail_pages_fetched": detail_pages_fetched,
            "single_pages_fetched": single_pages_fetched,
            "written": len(documents),
            "failed": len(failures),
            "failures": failures,
        },
    )
    return {
        "queued": queued_count,
        "fetched": fetched_count,
        "list_pages_fetched": list_pages_fetched,
        "detail_pages_fetched": detail_pages_fetched,
        "single_pages_fetched": single_pages_fetched,
        "written": len(documents),
        "failed": len(failures),
        "documents": documents,
        "failures": failures,
    }


async def _crawl_source_for_type(
    crawler: Crawl4AIRunner,
    source: CrawlSource,
    options: CrawlOptions,
    documents: list[dict[str, Any]],
    failures: list[dict[str, str]],
    run_config: Any,
) -> dict[str, int]:
    if source.page_type == PAGE_TYPE_SINGLE:
        return await _crawl_single_page_source(crawler, source, options, documents, failures, run_config)
    return await _crawl_source(crawler, source, options, documents, failures, run_config)


async def _crawl_single_page_source(
    crawler: Crawl4AIRunner,
    source: CrawlSource,
    options: CrawlOptions,
    documents: list[dict[str, Any]],
    failures: list[dict[str, str]],
    run_config: Any,
) -> dict[str, int]:
    seen: set[str] = set()
    queued = 0
    fetched = 0
    single_pages_fetched = 0

    for url in (_canonical_fetch_url(seed_url) for seed_url in source.urls):
        normalized = _normalize_crawl_url(url)
        if normalized in seen or not _is_crawlable_url(url):
            continue
        if _single_page_limit_reached(single_pages_fetched, source, options):
            seen.add(normalized)
            continue
        seen.add(normalized)
        queued += 1
        fetched += 1
        single_pages_fetched += 1

        try:
            result = await crawler.arun(url, config=run_config)
        except Exception as exc:
            failures.append({"source_name": source.name, "source_url": url, "error": str(exc)})
            continue

        if not getattr(result, "success", True):
            failures.append(
                {
                    "source_name": source.name,
                    "source_url": url,
                    "error": str(getattr(result, "error_message", "") or "crawl4ai returned success=false"),
                }
            )
            continue

        markdown = _result_markdown(result)
        if not markdown:
            failures.append({"source_name": source.name, "source_url": url, "error": "empty markdown"})
            continue

        if _should_write_document(url, source, is_seed=True):
            body_markdown = _body_markdown_from_result(url, result, markdown, page_type=source.page_type)
            document = _write_markdown_document(options.output_dir, source, url, body_markdown, result)
            documents.append(document)

    return {
        "queued": queued,
        "fetched": fetched,
        "list_pages_fetched": 0,
        "detail_pages_fetched": 0,
        "single_pages_fetched": single_pages_fetched,
    }


async def _crawl_source(
    crawler: Crawl4AIRunner,
    source: CrawlSource,
    options: CrawlOptions,
    documents: list[dict[str, Any]],
    failures: list[dict[str, str]],
    run_config: Any,
) -> dict[str, int]:
    queue: list[tuple[str, int, bool]] = [(_canonical_fetch_url(url), 0, True) for url in source.urls]
    seen: set[str] = set()
    queued = 0
    fetched = 0
    list_pages_fetched = 0
    detail_pages_fetched = 0

    while queue:
        url, depth, is_seed = queue.pop(0)
        normalized = _normalize_crawl_url(url)
        if normalized in seen or not _is_crawlable_url(url):
            continue
        if _is_list_page_url(url) and _list_page_limit_reached(list_pages_fetched, source, options):
            seen.add(normalized)
            continue
        seen.add(normalized)
        queued += 1
        fetched += 1
        if _is_list_page_url(url):
            list_pages_fetched += 1
        else:
            detail_pages_fetched += 1

        try:
            result = await crawler.arun(url, config=run_config)
        except Exception as exc:
            failures.append({"source_name": source.name, "source_url": url, "error": str(exc)})
            continue

        if not getattr(result, "success", True):
            failures.append(
                {
                    "source_name": source.name,
                    "source_url": url,
                    "error": str(getattr(result, "error_message", "") or "crawl4ai returned success=false"),
                }
            )
            continue

        markdown = _result_markdown(result)
        if not markdown:
            failures.append({"source_name": source.name, "source_url": url, "error": "empty markdown"})
            continue

        if _should_write_document(url, source, is_seed=is_seed):
            body_markdown = _body_markdown_from_result(url, result, markdown, page_type=source.page_type)
            document = _write_markdown_document(options.output_dir, source, url, body_markdown, result)
            documents.append(document)

        if depth < source.max_depth and _is_list_page_url(url):
            next_urls = [
                next_url
                for next_url in (urljoin(url, href) for href in _extract_result_links(result))
                if _should_write_document(next_url, source, is_seed=False)
            ]
            next_urls.extend(_extract_pagination_urls(url, result, source))
            for next_url in sorted(set(next_urls), key=_html_url_priority):
                if _should_follow_link(url, next_url, source):
                    queue.append((_canonical_fetch_url(next_url), depth + 1, False))

    return {
        "queued": queued,
        "fetched": fetched,
        "list_pages_fetched": list_pages_fetched,
        "detail_pages_fetched": detail_pages_fetched,
        "single_pages_fetched": 0,
    }


def _create_crawl4ai_runner(options: CrawlOptions):
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
    except ImportError as exc:
        raise RuntimeError(
            "crawl4ai is not installed. Install backend requirements before running the crawler."
        ) from exc

    browser_config = BrowserConfig(headless=True, user_agent=options.user_agent)
    return AsyncWebCrawler(config=browser_config)


def _create_run_config(options: CrawlOptions) -> Any:
    try:
        from crawl4ai import CacheMode, CrawlerRunConfig
    except ImportError:
        return None

    cache_mode = CacheMode.BYPASS if options.cache_mode.casefold() == "bypass" else CacheMode.ENABLED
    return CrawlerRunConfig(
        cache_mode=cache_mode,
        page_timeout=options.page_timeout_ms,
        wait_until=options.wait_until,
    )


def _write_markdown_document(
    output_dir: Path,
    source: CrawlSource,
    url: str,
    markdown: str,
    result: Any,
) -> dict[str, Any]:
    title = _result_title(result, markdown, url)
    normalized_url = _normalize_crawl_url(url)
    attachment_urls = _extract_attachment_urls(url, markdown, result)
    doc_id = _document_id(source.name, normalized_url)
    source_dir = output_dir / "final" / _slug(source.name)
    source_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{doc_id}_{_slug(title)[:80]}.md"
    path = source_dir / filename
    metadata = {
        "schema_version": CRAWL_SCHEMA_VERSION,
        "doc_id": doc_id,
        "source_name": source.name,
        "source_url": url,
        "title": title,
        "domain": source.domain,
        "department": source.department,
        "source_type": "html",
        "page_type": source.page_type,
        "crawler": "crawl4ai",
        "crawled_at": _utc_now(),
        "attachment_urls": attachment_urls,
    }
    path.write_text(_serialize_markdown(metadata, markdown), encoding="utf-8", newline="\n")
    return {
        "recommended_action": source.recommended_action,
        "doc_id": doc_id,
        "source_name": source.name,
        "source_url": url,
        "final_md_path": path.relative_to(output_dir).as_posix(),
        "title": title,
        "domain": source.domain,
        "department": source.department,
        "filter_reason": "",
        "confidence": "0.50",
    }


def load_sources_config(path: Path) -> list[CrawlSource]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict):
        rows = list(raw.get("sources") or [])
        rows.extend(_generated_department_sources(raw))
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("crawler config must be a list or contain a 'sources' list")
    return [_source_from_config(row) for row in rows]


def _source_from_config(row: dict[str, Any]) -> CrawlSource:
    urls = row.get("urls") or row.get("seed_urls") or row.get("start_urls") or row.get("url")
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        raise ValueError(f"source {row.get('name') or '<unnamed>'} has no urls")
    restrict_to_seed_board_queries = bool(row.get("restrict_to_seed_board_queries", False))
    return CrawlSource(
        name=str(row["name"]),
        urls=[str(url) for url in urls],
        domain=str(row.get("domain") or ""),
        department=str(row.get("department") or ""),
        max_depth=int(row.get("max_depth") or 0),
        max_pages=int(row.get("max_pages") or 0),
        max_pagination_pages=int(row.get("max_pagination_pages") or 0),
        same_host_only=bool(row.get("same_host_only", True)),
        collect_seed_pages=bool(row.get("collect_seed_pages", True)),
        follow_patterns=_string_tuple(row.get("follow_patterns")),
        collect_patterns=_string_tuple(row.get("collect_patterns")),
        exclude_patterns=_string_tuple(row.get("exclude_patterns")) or DEFAULT_EXCLUDE_PATTERNS,
        allowed_path_prefixes=_string_tuple(row.get("allowed_path_prefixes")) or _derive_allowed_path_prefixes(
            [str(url) for url in urls]
        ),
        allowed_query_param_filters=(
            _query_param_filters([str(url) for url in urls]) if restrict_to_seed_board_queries else ()
        ),
        recommended_action=str(row.get("recommended_action") or "MANUAL_REVIEW"),
        page_type=_page_type(row.get("page_type")),
    )


def _generated_department_sources(raw: dict[str, Any]) -> list[dict[str, Any]]:
    department_sites = raw.get("department_sites") or []
    blueprints = raw.get("source_blueprints") or []
    generated: list[dict[str, Any]] = []
    if not isinstance(department_sites, list) or not isinstance(blueprints, list):
        return generated

    for site in department_sites:
        if not isinstance(site, dict):
            continue
        site_type = str(site.get("site_type") or "")
        site_slug = str(site.get("slug") or site.get("department") or "").strip()
        if not site_slug:
            continue
        overrides_by_suffix = site.get("source_overrides") or {}
        for blueprint in blueprints:
            if not isinstance(blueprint, dict):
                continue
            applies_to = blueprint.get("applies_to") or []
            if applies_to and site_type not in applies_to:
                continue
            suffix = str(blueprint.get("name_suffix") or "").strip()
            if not suffix:
                continue
            row = {key: value for key, value in blueprint.items() if key not in {"applies_to", "name_suffix"}}
            row["name"] = f"{site_slug}_{suffix}"
            row["department"] = str(site.get("department") or site_slug)
            if suffix == "notice" and row.get("domain") == "general_notice":
                row["domain"] = "department_notice"
            override = overrides_by_suffix.get(suffix) if isinstance(overrides_by_suffix, dict) else None
            if isinstance(override, dict):
                row.update(override)
            if not (row.get("urls") or row.get("seed_urls") or row.get("start_urls") or row.get("url")):
                template = row.pop("seed_url_template", None)
                if template:
                    row["seed_urls"] = [str(template).format(**site)]
            generated.append(row)
    return generated


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.casefold(),)
    return tuple(str(item).casefold() for item in value)


def _page_type(value: Any) -> str:
    normalized = str(value or PAGE_TYPE_LIST).strip().upper()
    if normalized in {"SINGLE", "SINGLE_PAGE", "STATIC_PAGE"}:
        return PAGE_TYPE_SINGLE
    if normalized in {"LIST", "LIST_PAGE", "BOARD"}:
        return PAGE_TYPE_LIST
    raise ValueError(f"unsupported crawler page_type: {value}")


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {path}")
        shutil.rmtree(path)
    (path / "manifest").mkdir(parents=True, exist_ok=True)
    (path / "final").mkdir(parents=True, exist_ok=True)


def _write_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_CLASSIFICATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REQUIRED_CLASSIFICATION_COLUMNS})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _result_markdown(result: Any) -> str:
    markdown = getattr(result, "markdown", "")
    if hasattr(markdown, "fit_markdown") and markdown.fit_markdown:
        return str(markdown.fit_markdown).strip()
    if hasattr(markdown, "raw_markdown") and markdown.raw_markdown:
        return str(markdown.raw_markdown).strip()
    return str(markdown or "").strip()


def _result_title(result: Any, markdown: str, url: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return _clean_title(stripped.lstrip("#").strip())
    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    if title:
        return _clean_title(str(title))
    parsed = urlparse(url)
    return _clean_title(Path(parsed.path).stem or parsed.netloc or "untitled")


def _body_markdown_from_result(url: str, result: Any, fallback_markdown: str, *, page_type: str = PAGE_TYPE_LIST) -> str:
    html = _result_html(result)
    if html:
        extracted = _extract_main_markdown_from_html(html, url)
        if extracted:
            return _clean_crawled_markdown(url, extracted, page_type=page_type)
    return _clean_crawled_markdown(url, fallback_markdown, page_type=page_type)


def _extract_attachment_urls(base_url: str, markdown: str, result: Any) -> list[str]:
    urls: list[str] = []
    for candidate in re.findall(r"\[[^\]]*\]\((https?://[^)\s]+)\)", markdown):
        if _is_attachment_url(candidate):
            urls.append(_normalize_attachment_url(candidate))
    links = getattr(result, "links", None) or {}
    if isinstance(links, dict):
        for group in ("internal", "external"):
            for item in links.get(group) or []:
                href = item.get("href") if isinstance(item, dict) else item
                if not href:
                    continue
                candidate = urljoin(base_url, str(href))
                if _is_attachment_url(candidate):
                    urls.append(_normalize_attachment_url(candidate))
    html = _result_html(result)
    if html:
        urls.extend(_extract_attachment_urls_from_html(base_url, html))
    return list(dict.fromkeys(urls))


def _extract_attachment_urls_from_html(base_url: str, html: str) -> list[str]:
    urls: list[str] = []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        soup = None
    else:
        soup = BeautifulSoup(html, "html.parser")

    if soup is not None:
        for tag in soup.find_all(True):
            for value in tag.attrs.values():
                values = value if isinstance(value, list) else [value]
                for raw_value in values:
                    urls.extend(_extract_attachment_urls_from_text(base_url, str(raw_value)))

    urls.extend(_extract_attachment_urls_from_text(base_url, html))
    return list(dict.fromkeys(urls))


def _extract_attachment_urls_from_text(base_url: str, text: str) -> list[str]:
    urls: list[str] = []
    decoded = html_lib.unescape(text)
    for match in re.findall(r"""(?:https?://[^\s"'<>)]*|(?:\.\.?/|/)?[^\s"'<>)]*)downloadBbsFile\.do\?[^\s"'<>)]*""", decoded, flags=re.IGNORECASE):
        candidate = urljoin(base_url, match)
        if _is_attachment_url(candidate):
            urls.append(_normalize_attachment_url(candidate))
    return urls


def _result_html(result: Any) -> str:
    for attr in ("cleaned_html", "html"):
        html = getattr(result, attr, None)
        if html:
            return str(html)
    return ""


def _extract_main_markdown_from_html(html: str, base_url: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    _prune_non_content_nodes(soup)
    node = _select_main_content_node(soup)
    if node is None:
        return ""
    preferred = _prefer_body_like_descendant(node)
    if preferred is not None:
        node = preferred
    _prune_noise_descendants(node)
    markdown = _html_node_to_markdown(node, base_url=base_url)
    title = _find_detail_title_for_body_node(node)
    if title and not markdown.lstrip().startswith("#"):
        markdown = f"# {title}\n\n{markdown}"
    return _normalize_markdown(markdown)


def _prune_non_content_nodes(soup: Any) -> None:
    for tag in soup.find_all(["script", "style", "noscript", "svg", "canvas", "iframe", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    _prune_noise_descendants(soup)


def _prune_noise_descendants(node: Any) -> None:
    for tag in list(node.find_all(True)):
        if _is_noise_dom_node(tag):
            tag.decompose()


def _is_noise_dom_node(node: Any) -> bool:
    if getattr(node, "attrs", None) is None:
        return False
    name = getattr(node, "name", "")
    if name in {"nav", "header", "footer", "aside", "form"}:
        return True
    joined = " ".join(_node_signal_tokens(node))
    if not joined:
        return False
    noise_keywords = (
        "breadcrumb",
        "location",
        "lnb",
        "snb",
        "sidebar",
        "side-menu",
        "sidemenu",
        "left-menu",
        "leftmenu",
        "sub-menu",
        "submenu",
        "share",
        "sns",
        "print",
        "page-util",
        "pageutil",
        "quick",
        "custom",
        "customize",
        "personal",
        "popup",
    )
    return any(keyword in joined for keyword in noise_keywords)


def _select_main_content_node(soup: Any) -> Any:
    candidates: list[Any] = list(soup.find_all(True))
    body = soup.find("body")
    if body is not None:
        candidates.append(body)

    best_node = None
    best_score = 0.0
    seen: set[int] = set()
    for node in candidates:
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        score = _content_node_score(node)
        if score > best_score:
            best_score = score
            best_node = node
    if best_node is None or best_score < 80:
        return None
    return _refine_main_content_node(best_node, best_score)


def _refine_main_content_node(node: Any, node_score: float) -> Any:
    try:
        from bs4 import Tag
    except ImportError:
        return node

    current = node
    current_score = node_score
    current_text_length = len(_visible_text(current))

    while True:
        children = [child for child in current.find_all(recursive=False) if isinstance(child, Tag)]
        best_child = None
        best_child_score = 0.0
        best_child_text_length = 0
        for child in children:
            score = _content_node_score(child)
            if score > best_child_score:
                best_child = child
                best_child_score = score
                best_child_text_length = len(_visible_text(child))
        if (
            best_child is None
            or best_child_score < current_score * 0.85
            or best_child_text_length < current_text_length * 0.6
        ):
            return current
        current = best_child
        current_score = best_child_score
        current_text_length = best_child_text_length


def _prefer_body_like_descendant(node: Any) -> Any | None:
    try:
        from bs4 import Tag
    except ImportError:
        return None

    best_node = None
    best_score = 0.0
    for child in node.find_all(True):
        if not isinstance(child, Tag):
            continue
        score = _content_node_score(child)
        if score <= 0:
            continue
        if _is_body_like_container(child):
            score += 120
        if score > best_score:
            best_node = child
            best_score = score
    if best_node is None:
        return None
    parent_text = len(_visible_text(node))
    child_text = len(_visible_text(best_node))
    if child_text >= max(80, parent_text * 0.35):
        return best_node
    return None


def _find_detail_title_for_body_node(node: Any) -> str:
    try:
        from bs4 import Tag
    except ImportError:
        return ""

    best_title = ""
    best_score = 0
    current = node
    for _ in range(5):
        parent = getattr(current, "parent", None)
        if not isinstance(parent, Tag):
            break
        for sibling in parent.find_all(recursive=False):
            if sibling is current:
                break
            if not isinstance(sibling, Tag):
                continue
            text = _normalize_inline_markdown(_visible_text(sibling))
            if not 5 <= len(text) <= 180:
                continue
            lowered = text.casefold()
            if any(keyword in text for keyword in ("작성자", "작성일", "조회수", "첨부파일")):
                continue
            if any(keyword in lowered for keyword in ("sns", "공유", "print", "목록")):
                continue
            tokens = _node_signal_tokens(sibling)
            joined = " ".join(tokens)
            score = len(text)
            if any(keyword in joined for keyword in ("subject", "title", "heading", "headline")):
                score += 300
            if score > best_score:
                best_title = text
                best_score = score
        current = parent
    return best_title


def _content_node_score(node: Any) -> float:
    text = _visible_text(node)
    text_length = len(text)
    if text_length < 60:
        return 0.0
    link_text_length = sum(len(_visible_text(link)) for link in node.find_all("a"))
    link_density = link_text_length / max(text_length, 1)
    punctuation_hits = len(re.findall(r"[.!?。！？]|[가-힣]{2,}", text))
    heading_hits = len(node.find_all(re.compile(r"^h[1-6]$")))
    block_hits = len(node.find_all(["p", "li", "table", "tr", "blockquote", "section", "article", "div"], recursive=True))
    depth_penalty = _node_depth(node) * 10
    tag_penalty = 600 if getattr(node, "name", "") == "body" else 0
    class_bonus, class_penalty = _node_class_signal_score(node)
    return (
        text_length * (1.0 - min(link_density, 0.9))
        + min(punctuation_hits, 80) * 6
        + min(block_hits, 40) * 12
        + heading_hits * 25
        + class_bonus
        - class_penalty
        - depth_penalty
        - tag_penalty
    )


def _is_body_like_container(node: Any) -> bool:
    tokens = _node_signal_tokens(node)
    if not tokens:
        return False
    joined = " ".join(tokens)
    return any(keyword in joined for keyword in ("boardbody", "body", "content", "article", "view", "detail"))


def _node_class_signal_score(node: Any) -> tuple[float, float]:
    tokens = _node_signal_tokens(node)

    if not tokens:
        return 0.0, 0.0

    joined = " ".join(tokens)
    bonus = 0.0
    penalty = 0.0
    bonus_keywords = {
        "content": 180,
        "body": 220,
        "article": 180,
        "view": 150,
        "detail": 130,
        "board": 90,
        "notice": 80,
        "post": 70,
        "bbs": 70,
        "articleview": 150,
        "boardview": 180,
        "boardbody": 240,
        "contentbody": 240,
        "viewcontent": 520,
        "viewcontentbox": 540,
        "contenttext": 520,
        "bbsviewbox": 340,
    }
    penalty_keywords = {
        "menu": 260,
        "gnb": 240,
        "nav": 220,
        "breadcrumb": 180,
        "location": 120,
        "footer": 260,
        "header": 260,
        "aside": 180,
        "sidebar": 180,
        "search": 180,
        "login": 180,
        "share": 120,
        "sns": 120,
        "paging": 150,
        "pagination": 150,
        "list": 90,
        "table": 40,
    }
    for keyword, weight in bonus_keywords.items():
        if keyword in joined:
            bonus += weight
    for keyword, weight in penalty_keywords.items():
        if keyword in joined:
            penalty += weight
    return bonus, penalty


def _node_signal_tokens(node: Any) -> list[str]:
    tokens: list[str] = []
    for attr in ("id", "class"):
        value = getattr(node, "get", lambda *_: None)(attr)
        if not value:
            continue
        if isinstance(value, str):
            tokens.extend(_split_signal_tokens(value))
        else:
            for item in value:
                tokens.extend(_split_signal_tokens(str(item)))
    return tokens


def _split_signal_tokens(value: str) -> list[str]:
    return [part.casefold() for part in re.split(r"[^0-9A-Za-z가-힣]+", value) if part]


def _visible_text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _node_depth(node: Any) -> int:
    depth = 0
    parent = getattr(node, "parent", None)
    while parent is not None:
        depth += 1
        parent = getattr(parent, "parent", None)
    return depth


def _html_node_to_markdown(node: Any, *, base_url: str, list_depth: int = 0) -> str:
    try:
        from bs4 import Comment, NavigableString, Tag
    except ImportError:
        return ""

    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""

    name = node.name.casefold()
    if name in {"script", "style", "noscript", "svg", "canvas", "iframe"}:
        return ""
    if name in {"br"}:
        return "\n"
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        return f"\n{'#' * level} {_children_markdown(node, base_url=base_url, list_depth=list_depth).strip()}\n\n"
    if name == "a":
        label = _children_markdown(node, base_url=base_url, list_depth=list_depth).strip() or node.get("href", "").strip()
        href = node.get("href", "").strip()
        if not href or href.startswith("#"):
            return label
        return f"[{label}]({urljoin(base_url, href)})"
    if name in {"ul", "ol"}:
        return "\n".join(
            _html_node_to_markdown(child, base_url=base_url, list_depth=list_depth + 1)
            for child in node.find_all("li", recursive=False)
        ) + "\n\n"
    if name == "li":
        prefix = "  " * max(list_depth - 1, 0) + "- "
        return prefix + _children_markdown(node, base_url=base_url, list_depth=list_depth).strip() + "\n"
    if name == "table":
        return _table_to_markdown(node, base_url=base_url)
    if name in {"p", "div", "section", "article", "main", "blockquote", "tr"}:
        return f"\n{_children_markdown(node, base_url=base_url, list_depth=list_depth).strip()}\n"
    if name in {"td", "th"}:
        return _children_markdown(node, base_url=base_url, list_depth=list_depth).strip()
    return _children_markdown(node, base_url=base_url, list_depth=list_depth)


def _children_markdown(node: Any, *, base_url: str, list_depth: int) -> str:
    return "".join(_html_node_to_markdown(child, base_url=base_url, list_depth=list_depth) for child in node.children)


def _table_to_markdown(node: Any, *, base_url: str) -> str:
    rows: list[list[str]] = []
    for tr in node.find_all("tr"):
        cells = [
            _normalize_inline_markdown(_html_node_to_markdown(cell, base_url=base_url).strip())
            for cell in tr.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    separator = ["---"] * width
    body = rows[1:]
    markdown_rows = [f"| {' | '.join(header)} |", f"| {' | '.join(separator)} |"]
    markdown_rows.extend(f"| {' | '.join(row)} |" for row in body)
    return "\n" + "\n".join(markdown_rows) + "\n\n"


def _normalize_markdown(markdown: str) -> str:
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    lines = [re.sub(r"\s+", " ", line).strip() for line in markdown.splitlines()]
    lines = [line for line in lines if not re.fullmatch(r"//[A-Za-z0-9_-]+", line)]
    return "\n".join(line for line in lines).strip()


def _normalize_inline_markdown(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().replace("|", "\\|")


def _is_attachment_url(url: str) -> bool:
    lowered = url.casefold()
    if "downloadbbsfile.do" in lowered:
        return True
    path = urlparse(url).path.casefold()
    return path.endswith(
        (
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
    )


def _clean_crawled_markdown(url: str, markdown: str, *, page_type: str = PAGE_TYPE_LIST) -> str:
    if page_type == PAGE_TYPE_SINGLE or "contents.do" in urlparse(url).path.casefold():
        return _clean_content_page_markdown(markdown)
    if "selectbbsnttview.do" in urlparse(url).path.casefold():
        return _clean_detail_markdown(markdown)
    return markdown.strip()


def _clean_content_page_markdown(markdown: str) -> str:
    lines = _compact_markdown_lines(markdown)
    start = _find_content_page_body_start(lines)
    if start is not None:
        lines = lines[start:]
    lines = _trim_trailing_content_page_noise(lines)
    lines = _promote_first_heading_to_h1(lines)
    return "\n".join(lines).strip() or markdown.strip()


def _compact_markdown_lines(markdown: str) -> list[str]:
    cleaned: list[str] = []
    previous_blank = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        previous_blank = False
        cleaned.append(stripped)
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def _find_content_page_body_start(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        normalized = re.sub(r"\s+", " ", line).strip()
        if re.match(r"^#{3,}\s+", normalized):
            return index
        if re.match(r"^#{2,}\s*\d+[.)]\s+", normalized):
            return index
    for index, line in enumerate(lines):
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized.startswith("|") and normalized.endswith("|"):
            return index
        if index > 0 and _is_body_signal_line(normalized) and not _is_markdown_link_only_line(normalized):
            return index
    return 0 if lines else None


def _promote_first_heading_to_h1(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if not line:
            continue
        match = re.match(r"^(#{2,6})\s+(.+)$", line)
        if match:
            return [*lines[:index], f"# {match.group(2).strip()}", *lines[index + 1 :]]
        if line.startswith("# "):
            return lines
        return [f"# {line}", *lines[index + 1 :]]
    return lines


def _is_markdown_link_only_line(line: str) -> bool:
    return bool(re.fullmatch(r"-?\s*\[[^\]]+\]\([^)]+\)", line.strip()))


def _trim_trailing_content_page_noise(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if index == 0:
            continue
        if _is_content_page_trailing_noise_start(line):
            return lines[:index]
    return lines


def _is_content_page_trailing_noise_start(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip()
    lowered = normalized.casefold()
    if normalized.startswith("- [HOME]("):
        return True
    if normalized in {"맞춤", "설정", "닫기", "초기화", "맞춤설정", "맞춤설정이란?"}:
        return True
    if normalized.startswith("## 맞춤정보"):
        return True
    if "어떤 정보를" in normalized or "찾고계시나요" in normalized:
        return True
    if "맞춤설정 저장하기" in normalized or "#custom" in lowered:
        return True
    return False


def _clean_detail_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        previous_blank = False
        if cleaned and _is_related_detail_link_line(stripped):
            break
        if _is_detail_noise_line(stripped):
            continue
        cleaned.append(stripped)
    body_start = _find_body_start_index(cleaned)
    if body_start is not None:
        cleaned = cleaned[body_start:]
    cleaned = _trim_trailing_detail_navigation(cleaned)
    return "\n".join(cleaned).strip() or markdown.strip()


def _find_body_start_index(lines: list[str]) -> int | None:
    if lines and lines[0].lstrip().startswith("#"):
        return 0
    for index, line in enumerate(lines):
        if _is_body_signal_line(line):
            return index
    return None


def _is_detail_noise_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip()
    if not normalized:
        return True
    if normalized in {"닫기", "* 인쇄", "인쇄", "SNS공유"}:
        return True
    lowered = normalized.casefold()
    if "주메뉴 열기" in normalized or lowered.startswith("site header") or lowered.startswith("site menu") or lowered.startswith("full crawl markdown"):
        return True
    if normalized.startswith(("[목록]", "이전글", "다음글", "게시물삭제")):
        return True
    if _is_related_detail_link_line(normalized):
        return True
    if " / " in normalized and normalized.count("/") >= 2 and len(normalized) <= 120 and "http" not in lowered:
        return True
    if re.fullmatch(r"-?\s*\[[^\]]+\]\([^)]+\)", normalized):
        return True
    if normalized.lower() in {"login", "logout", "search", "menu"}:
        return True
    return False


def _trim_trailing_detail_navigation(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if _is_related_detail_link_line(line):
            return lines[:index]
    return lines


def _is_related_detail_link_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip().casefold()
    return bool(
        "selectbbsnttview.do" in normalized
        and ("pageunit=" in normalized or "searchcnd=" in normalized or normalized.startswith("["))
    )


def _is_body_signal_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip()
    if not normalized:
        return False
    if normalized.startswith(("1.", "1)", "2.", "2)", "3.", "3)", "4.", "4)", "5.", "5)", "6.", "6)", "7.", "7)")):
        return True
    if normalized.startswith(("-", "•", "※", "◎", "▶", "☞")):
        return True
    if re.search(r"[가-힣].*[.!?。！？]$", normalized) and len(normalized) >= 15:
        return True
    return False


def _extract_result_links(result: Any) -> list[str]:
    links = getattr(result, "links", None) or {}
    hrefs: list[str] = []
    if isinstance(links, dict):
        for item in links.get("internal") or []:
            if isinstance(item, dict) and item.get("href"):
                hrefs.append(str(item["href"]))
            elif isinstance(item, str):
                hrefs.append(item)
    return hrefs


def _should_follow_link(source_url: str, next_url: str, source: CrawlSource) -> bool:
    if not _is_crawlable_url(next_url):
        return False
    if source.same_host_only and urlparse(source_url).netloc.casefold() != urlparse(next_url).netloc.casefold():
        return False
    lowered = _normalize_crawl_url(next_url).casefold()
    if any(pattern in lowered for pattern in source.exclude_patterns):
        return False
    if source.allowed_query_param_filters and not _matches_allowed_query_filters(next_url, source.allowed_query_param_filters):
        return False
    if source.allowed_path_prefixes and not _matches_allowed_path_prefix(next_url, source.allowed_path_prefixes):
        return False
    return not source.follow_patterns or any(pattern in lowered for pattern in source.follow_patterns)


def _should_write_document(url: str, source: CrawlSource, *, is_seed: bool) -> bool:
    if is_seed and not source.collect_seed_pages:
        return False
    lowered = _normalize_crawl_url(url).casefold()
    if any(pattern in lowered for pattern in source.exclude_patterns):
        return False
    if source.allowed_query_param_filters and not _matches_allowed_query_filters(url, source.allowed_query_param_filters):
        return False
    if source.collect_patterns:
        return any(pattern in lowered for pattern in source.collect_patterns)
    return "selectbbsnttlist.do" not in urlparse(url).path.casefold()


def _is_crawlable_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme.casefold() not in {"http", "https"}:
        return False
    lowered = url.strip().casefold()
    return not lowered.startswith(("javascript:", "mailto:", "tel:", "#"))


def _extract_pagination_urls(url: str, result: Any, source: CrawlSource) -> list[str]:
    if "selectbbsnttlist.do" not in urlparse(url).path.casefold():
        return []
    last_page = max(_pagination_last_page(result), _pagination_last_page_from_dom(url, result))
    if last_page <= 1:
        return []
    if source.max_pagination_pages > 0:
        last_page = min(last_page, source.max_pagination_pages)
    current_page = _current_page_index(url)
    urls = [_with_query_param(url, "pageIndex", str(page)) for page in range(2, last_page + 1) if page != current_page]
    return [next_url for next_url in urls if _should_follow_link(url, next_url, source)]


def _pagination_last_page(result: Any) -> int:
    markdown = _result_markdown(result)
    match = re.search(r"페이지\s*:\s*_?\d+_?\s*/\s*_?(\d+)_?", markdown)
    last_page = int(match.group(1)) if match else 1
    links = getattr(result, "links", None) or {}
    if isinstance(links, dict):
        for group in ("internal", "external"):
            for item in links.get(group) or []:
                href = item.get("href") if isinstance(item, dict) else item
                if not href:
                    continue
                query = dict(_casefold_query(parse_qsl(urlparse(str(href)).query, keep_blank_values=True)))
                page_index = query.get("pageindex")
                if page_index and page_index.isdigit():
                    last_page = max(last_page, int(page_index))
    return last_page


def _pagination_last_page_from_dom(url: str, result: Any) -> int:
    html = _result_html(result)
    if not html:
        return 1
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return 1

    soup = BeautifulSoup(html, "html.parser")
    current_page = _current_page_index(url)
    list_path = urlparse(url).path.casefold()
    page_indexes: set[int] = set()

    for element in soup.find_all(["a", "button"]):
        page_index = _pagination_page_index_from_element(
            element,
            current_page=current_page,
            list_path=list_path,
            source_url=url,
        )
        if page_index is not None:
            page_indexes.add(page_index)

    return max(page_indexes) if page_indexes else 1


def _pagination_page_index_from_element(
    element: Any,
    *,
    current_page: int,
    list_path: str,
    source_url: str,
) -> int | None:
    text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
    attrs = " ".join(
        str(value)
        for value in (
            element.get("aria-label"),
            element.get("title"),
            element.get("data-page"),
            element.get("data-page-index"),
            element.get("data-pageno"),
            element.get("data-index"),
            element.get("class"),
            element.get("id"),
        )
        if value
    )
    parent = getattr(element, "parent", None)
    parent_attrs = ""
    if parent is not None and getattr(parent, "attrs", None):
        parent_attrs = " ".join(
            str(value)
            for value in (
                parent.get("aria-label"),
                parent.get("title"),
                parent.get("class"),
                parent.get("id"),
            )
            if value
        )
    hint = f"{text} {attrs} {parent_attrs}".casefold()
    href = str(element.get("href") or element.get("data-href") or element.get("data-url") or "")
    resolved_href = urljoin(source_url, href) if href else ""
    href_query = dict(_casefold_query(parse_qsl(urlparse(resolved_href).query, keep_blank_values=True))) if resolved_href else {}

    for key in ("pageindex", "page", "pageno", "p", "idx", "page_no"):
        page_value = href_query.get(key)
        if page_value and page_value.isdigit():
            return int(page_value)

    if text.isdigit():
        page_index = int(text)
        if page_index != current_page and (
            _has_pagination_hint(hint) or urlparse(resolved_href).path.casefold() == list_path
        ):
            return page_index

    if _is_next_pagination_hint(hint):
        return current_page + 1

    if _is_last_pagination_hint(hint):
        max_visible = _max_visible_page_from_hint(hint)
        if max_visible is not None and max_visible > current_page:
            return max_visible
        if text.isdigit():
            page_index = int(text)
            if page_index > current_page:
                return page_index
        return current_page + 1

    return None


def _has_pagination_hint(hint: str) -> bool:
    return any(keyword in hint for keyword in _PAGINATION_HINTS)


def _is_next_pagination_hint(hint: str) -> bool:
    return any(keyword in hint for keyword in _PAGINATION_NEXT_HINTS)


def _is_last_pagination_hint(hint: str) -> bool:
    return any(keyword in hint for keyword in _PAGINATION_LAST_HINTS)


def _max_visible_page_from_hint(hint: str) -> int | None:
    matches = [int(match) for match in re.findall(r"\b(\d+)\b", hint)]
    return max(matches) if matches else None


def _is_list_page_url(url: str) -> bool:
    path = urlparse(url).path.casefold()
    if "selectbbsnttlist.do" in path or path.endswith("/list.do"):
        return True
    query = dict(_casefold_query(parse_qsl(urlparse(url).query, keep_blank_values=True)))
    return "pageindex" in query and ("bbsno" in query or "key" in query)


def _list_page_limit_reached(list_pages_fetched: int, source: CrawlSource, options: CrawlOptions) -> bool:
    if options.max_pages is not None:
        return list_pages_fetched >= options.max_pages
    return source.max_pages > 0 and list_pages_fetched >= source.max_pages


def _single_page_limit_reached(single_pages_fetched: int, source: CrawlSource, options: CrawlOptions) -> bool:
    if options.max_pages is not None:
        return single_pages_fetched >= options.max_pages
    return source.max_pages > 0 and single_pages_fetched >= source.max_pages


def _with_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query = [(name, item_value) for name, item_value in parse_qsl(parsed.query, keep_blank_values=True)]
    lowered_key = key.casefold()
    replaced = False
    next_query: list[tuple[str, str]] = []
    for name, item_value in query:
        if name.casefold() == lowered_key:
            next_query.append((name, value))
            replaced = True
        else:
            next_query.append((name, item_value))
    if not replaced:
        next_query.append((key, value))
    return urlunparse(parsed._replace(query=urlencode(next_query, doseq=True), fragment=""))


def _canonical_fetch_url(url: str) -> str:
    url, _fragment = urldefrag(url.strip())
    parsed = urlparse(url)
    path = _strip_path_params(parsed.path).rstrip("/") or parsed.path
    path_lower = path.casefold()
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query_by_lower: dict[str, str] = {}
    for key, value in query:
        query_by_lower.setdefault(key.casefold(), value)

    if "selectbbsnttview.do" in path_lower and all(key in query_by_lower for key in ("bbsno", "key", "nttno")):
        canonical_query = [
            ("bbsNo", query_by_lower["bbsno"]),
            ("key", query_by_lower["key"]),
            ("nttNo", query_by_lower["nttno"]),
        ]
    elif "selectbbsnttlist.do" in path_lower:
        canonical_query = [
            (name, query_by_lower[key])
            for name, key in (("bbsNo", "bbsno"), ("dc", "dc"), ("key", "key"), ("pageIndex", "pageindex"))
            if key in query_by_lower
        ]
    else:
        canonical_query = query

    return urlunparse(
        parsed._replace(
            scheme=parsed.scheme.casefold(),
            netloc=parsed.netloc.casefold(),
            path=path,
            params="",
            query=urlencode(canonical_query, doseq=True),
            fragment="",
        )
    )


def _current_page_index(url: str) -> int:
    query = dict(_casefold_query(parse_qsl(urlparse(url).query, keep_blank_values=True)))
    page_index = query.get("pageindex")
    return int(page_index) if page_index and page_index.isdigit() else 1


def _html_url_priority(url: str) -> tuple[int, str]:
    path = urlparse(url).path.casefold()
    if "selectbbsnttview.do" in path:
        return (0, url)
    if "selectbbsnttlist.do" in path:
        return (1, url)
    if "contents.do" in path:
        return (2, url)
    return (3, url)


def _query_param_filters(urls: list[str]) -> tuple[tuple[tuple[str, str], ...], ...]:
    filters: list[tuple[tuple[str, str], ...]] = []
    for url in urls:
        parsed = urlparse(url)
        if "selectbbsnttlist.do" not in parsed.path.casefold():
            continue
        query = dict(_casefold_query(parse_qsl(parsed.query, keep_blank_values=True)))
        params = tuple(
            sorted(
                (key, query[key])
                for key in ("bbsno", "key")
                if query.get(key)
            )
        )
        if len(params) == 2:
            filters.append(params)
    return tuple(dict.fromkeys(filters))


def _matches_allowed_query_filters(
    url: str,
    filters: tuple[tuple[tuple[str, str], ...], ...],
) -> bool:
    params = dict(_casefold_query(parse_qsl(urlparse(url).query, keep_blank_values=True)))
    for required_params in filters:
        if all(params.get(key) == value for key, value in required_params):
            return True
    return False


def _derive_allowed_path_prefixes(urls: list[str]) -> tuple[str, ...]:
    prefixes: list[str] = []
    for url in urls:
        path = urlparse(url).path or "/"
        if path == "/":
            prefix = "/"
        else:
            first_segment = path.strip("/").split("/", 1)[0]
            prefix = f"/{first_segment}/" if first_segment else "/"
        if prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def _matches_allowed_path_prefix(url: str, prefixes: tuple[str, ...]) -> bool:
    path = urlparse(url).path
    normalized_path = path if path.endswith("/") else f"{path}/"
    return any(normalized_path.startswith(prefix) for prefix in prefixes)


def _casefold_query(query: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(key.casefold(), value.casefold()) for key, value in query]


def _normalize_crawl_url(url: str) -> str:
    url, _fragment = urldefrag(url.strip())
    parsed = urlparse(url)
    path = _strip_path_params(parsed.path).rstrip("/") or parsed.path
    query_pairs = _normalized_query_pairs(parsed.path, parsed.query)
    return urlunparse(
        parsed._replace(
            scheme=parsed.scheme.casefold(),
            netloc=parsed.netloc.casefold(),
            path=path,
            params="",
            query=urlencode(query_pairs, doseq=True),
            fragment="",
        )
    )


def _normalize_attachment_url(url: str) -> str:
    url, _fragment = urldefrag(url.strip())
    parsed = urlparse(url)
    path = _strip_path_params(parsed.path).rstrip("/") or parsed.path
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    return urlunparse(
        parsed._replace(
            scheme=parsed.scheme.casefold(),
            netloc=parsed.netloc.casefold(),
            path=path,
            params="",
            query=urlencode(query_pairs, doseq=True),
            fragment="",
        )
    )


def _strip_path_params(path: str) -> str:
    return re.sub(r";jsessionid=[^/?#]+", "", path, flags=re.IGNORECASE)


def _normalized_query_pairs(path: str, query: str) -> list[tuple[str, str]]:
    pairs = _casefold_query(parse_qsl(query, keep_blank_values=True))
    path_lower = path.casefold()
    if "selectbbsnttview.do" in path_lower:
        keep = {"bbsno", "key", "nttno", "selfat"}
    elif "selectbbsnttlist.do" in path_lower:
        keep = {"bbsno", "dc", "key", "pageindex", "selfat", "sf.dc", "sf.dcof", "sf.of1", "sf.of2"}
    else:
        keep = None
    if keep is not None:
        pairs = [(key, value) for key, value in pairs if key in keep]
    return sorted(pairs)


def _serialize_markdown(metadata: dict[str, Any], markdown: str) -> str:
    frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
    return f"---\n{frontmatter}\n---\n\n{markdown.strip()}\n"


def _document_id(source_name: str, normalized_url: str) -> str:
    return "crawl-" + hashlib.sha256(f"{source_name}:{normalized_url}".encode("utf-8")).hexdigest()[:24]


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "untitled"


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:300] or "untitled"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_args() -> CrawlOptions:
    parser = argparse.ArgumentParser(description="Crawl web pages with Crawl4AI and write prepare_markdown input files.")
    parser.add_argument("--config", type=Path, required=True, help="YAML file with a sources list.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to backend/data/crawled_markdown/<run>.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--source", dest="source_names", action="append", help="Source name to run. Can be repeated.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--run-id", help="Directory name to use under backend/data/crawled_markdown when --output-dir is omitted.")
    args = parser.parse_args()
    sources = load_sources_config(args.config)
    if args.source_names:
        allowed = {name.casefold() for name in args.source_names}
        sources = [source for source in sources if source.name.casefold() in allowed]
    return CrawlOptions(
        sources=sources,
        output_dir=args.output_dir or _default_output_dir(args.run_id),
        force=args.force,
        max_pages=args.max_pages,
        user_agent=args.user_agent,
    )


def _default_output_dir(run_id: str | None = None) -> Path:
    if run_id:
        return DEFAULT_CRAWLED_MARKDOWN_ROOT / _safe_path_name(run_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_CRAWLED_MARKDOWN_ROOT / f"crawl4ai-{timestamp}"


def _safe_path_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in value.strip())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:120] or "run"


if __name__ == "__main__":
    print(json.dumps(crawl_markdown(_parse_args()), ensure_ascii=False, indent=2))
