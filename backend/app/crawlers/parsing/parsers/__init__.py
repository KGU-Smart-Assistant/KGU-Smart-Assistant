from app.crawlers.parsing.parsers.base import BaseParser
from app.crawlers.parsing.parsers.faq_parser import FaqParser
from app.crawlers.parsing.parsers.generic_markdown_parser import GenericMarkdownParser
from app.crawlers.parsing.parsers.notice_detail_parser import NoticeDetailParser
from app.crawlers.parsing.parsers.schedule_parser import ScheduleParser

__all__ = [
    "BaseParser",
    "FaqParser",
    "GenericMarkdownParser",
    "NoticeDetailParser",
    "ScheduleParser",
]
