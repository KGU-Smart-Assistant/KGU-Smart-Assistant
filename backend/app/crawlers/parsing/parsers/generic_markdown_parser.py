import re
from urllib.parse import urljoin

from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.crawlers.parsing.parsers.base import BaseParser
from app.crawlers.parsing.schemas import ParseContext, ParsedDocument


class GenericMarkdownParser(BaseParser):
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

        if "rule.kyonggi.ac.kr" in context.url.casefold():
            rule_title_match = re.search(r"\*\*([^*\n]+)\*\*", content)
            if rule_title_match:
                return ParsedDocument(
                    title=rule_title_match.group(1).strip()[:300],
                    content=content,
                    attachment_urls=_extract_attachment_urls(context.url, result),
                )

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("![") or stripped.startswith("["):
                continue

            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                return ParsedDocument(
                    title=title[:300],
                    content=content,
                    attachment_urls=_extract_attachment_urls(context.url, result),
                )

        return None


def _extract_attachment_urls(base_url: str, result) -> list[str]:
    attachment_urls: list[str] = []
    seen: set[str] = set()
    for link_group in (getattr(result, "links", {}) or {}).values():
        for link in link_group or []:
            href = (link or {}).get("href") or ""
            normalized = urljoin(base_url, href)
            if not _looks_like_attachment_url(normalized):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            attachment_urls.append(normalized)
    return attachment_urls


def _looks_like_attachment_url(url: str) -> bool:
    lowered = url.casefold()
    return (
        "downloadcontentsfile.do" in lowered
        or "downloadbbsfile.do" in lowered
        or lowered.endswith((".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx"))
    )
