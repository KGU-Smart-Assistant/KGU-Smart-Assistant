from app.crawlers.parsing.content_cleaner import clean_crawled_markdown
from app.crawlers.parsing.parsers.base import BaseParser
from app.crawlers.parsing.schemas import ParseContext, ParsedDocument


class FaqParser(BaseParser):
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

        normalized = " ".join(content.split())
        if (
            "총게시물" in normalized
            and "_0_" in normalized
            and "검색결과가 없습니다" in normalized
        ):
            return None
        if "총게시물 : 0" in normalized or "총게시물: 0" in normalized:
            return None

        title = self._extract_title(result)
        cleaned_lines = self._extract_faq_lines(content)
        if not cleaned_lines:
            return None

        return ParsedDocument(
            title=(title or cleaned_lines[0])[:300],
            content="\n".join(cleaned_lines),
        )

    def _extract_title(self, result) -> str | None:
        metadata = getattr(result, "metadata", None) or {}
        for key in ("title", "og:title"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return None

    def _extract_faq_lines(self, content: str) -> list[str]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        cleaned_lines: list[str] = []
        in_faq_section = False

        for line in lines:
            normalized = " ".join(line.split()).casefold()
            if normalized in {"faq", "## faq"}:
                in_faq_section = True
                continue

            if not in_faq_section:
                continue

            if self._should_skip_line(line):
                continue

            cleaned_lines.append(line)

        if cleaned_lines:
            return cleaned_lines

        for line in lines:
            if self._should_skip_line(line):
                continue
            cleaned_lines.append(line)

        return cleaned_lines

    def _should_skip_line(self, line: str) -> bool:
        if not line:
            return True

        stripped = line.strip()
        if (
            stripped.startswith("![")
            or stripped.startswith("[")
            or stripped.startswith("* [")
            or stripped.startswith("*")
            or stripped.startswith("1.")
            or stripped.startswith("2.")
            or stripped.startswith("3.")
        ):
            return True

        normalized = " ".join(stripped.split()).casefold()
        boilerplate_tokens = (
            "faq",
            "질문 답변 검색",
            "question answer search",
            "본문 바로가기",
            "열린광장",
            "페이스북",
            "트위터",
            "네이버블로그",
            "카카오톡",
        )
        return any(token in normalized for token in boilerplate_tokens)
