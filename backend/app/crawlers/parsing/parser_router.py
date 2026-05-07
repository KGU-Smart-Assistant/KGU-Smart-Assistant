from app.crawlers.parsing.parser_registry import REGISTRY
from app.crawlers.parsing.parsers.faq_parser import FaqParser
from app.crawlers.parsing.parsers.generic_markdown_parser import GenericMarkdownParser
from app.crawlers.parsing.parsers.notice_detail_parser import NoticeDetailParser
from app.crawlers.parsing.parsers.schedule_parser import ScheduleParser
from app.crawlers.parsing.schemas import ParseContext, ParsedDocument


class ParserRouter:
    def __init__(self) -> None:
        self.parsers = {
            "faq": FaqParser(),
            "notice_detail": NoticeDetailParser(),
            "generic_markdown": GenericMarkdownParser(),
            "schedule": ScheduleParser(),
        }

    def parse(self, result, context: ParseContext) -> ParsedDocument | None:
        url = context.url.lower()
        category = (context.category or "").lower()
        matched_specific_parser = False

        for entry in REGISTRY:
            url_patterns = entry.get("url_patterns", ())
            categories = entry.get("categories", ())
            url_matches = not url_patterns or any(pattern in url for pattern in url_patterns)
            category_matches = not categories or category in categories
            if not (url_matches and category_matches):
                continue

            parser_name = entry["parser"]
            if entry["name"] != "generic_markdown":
                matched_specific_parser = True

            parser_context = ParseContext(
                url=context.url,
                category=context.category,
                department=context.department,
                allowed_keyword_filters=context.allowed_keyword_filters,
                blocked_keyword_filters=context.blocked_keyword_filters,
                allowed_author_department_filters=context.allowed_author_department_filters,
                blocked_author_department_filters=context.blocked_author_department_filters,
                parser_options=dict(entry.get("options", {})),
            )
            parsed = self.parsers[parser_name].parse(result=result, context=parser_context)
            if parsed is not None:
                return parsed
            if matched_specific_parser:
                return None

        return self.parsers["generic_markdown"].parse(result=result, context=context)
