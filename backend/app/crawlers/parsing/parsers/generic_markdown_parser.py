import re

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

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("![") or stripped.startswith("["):
                continue

            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                return ParsedDocument(title=title[:300], content=content)

        return None
