import re

from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.crawlers.parsing.parsers.base import BaseParser
from app.crawlers.parsing.schemas import ParseContext, ParsedDocument


class ScheduleParser(BaseParser):
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

        title = self._extract_title(result, content)
        if not title:
            return None

        return ParsedDocument(title=title, content=content)

    def _extract_title(self, result, content: str) -> str | None:
        metadata = getattr(result, "metadata", None) or {}
        for key in ("title", "og:title"):
            value = metadata.get(key)
            if value:
                normalized = re.sub(r"\s+", " ", str(value)).strip()
                if normalized:
                    return normalized[:300]

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("![") or stripped.startswith("["):
                continue
            return re.sub(r"^#+\s*", "", stripped).strip()[:300]

        return None
