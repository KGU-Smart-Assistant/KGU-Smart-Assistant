from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParseContext:
    url: str
    category: Optional[str]
    department: Optional[str]
    allowed_keyword_filters: Optional[Tuple[str, ...]] = None
    blocked_keyword_filters: Optional[Tuple[str, ...]] = None
    allowed_author_department_filters: Optional[Tuple[str, ...]] = None
    blocked_author_department_filters: Optional[Tuple[str, ...]] = None
    parser_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    title: str
    content: str
    published_at: Optional[datetime] = None
    author_department: Optional[str] = None
    attachment_urls: List[str] = field(default_factory=list)
