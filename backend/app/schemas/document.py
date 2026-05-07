from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


SourceType = Literal[
    "notice",
    "pdf",
    "docx",
    "hwp",
    "hwpx",
    "zip",
    "image",
    "html",
    "markdown",
    "file",
]


class DocumentBase(BaseModel):
    source_type: SourceType = "notice"
    source_url: str
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1)
    category: Optional[str] = None
    department: Optional[str] = None
    author_department: Optional[str] = None
    published_at: Optional[datetime] = None
    attachment_urls: List[str] = Field(default_factory=list)


class Document(DocumentBase):
    doc_id: str
    collected_at: datetime
