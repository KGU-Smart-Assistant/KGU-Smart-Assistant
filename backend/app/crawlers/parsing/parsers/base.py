from abc import ABC, abstractmethod

from app.crawlers.parsing.schemas import ParseContext, ParsedDocument


class BaseParser(ABC):
    @abstractmethod
    def parse(self, result, context: ParseContext) -> ParsedDocument | None:
        raise NotImplementedError
