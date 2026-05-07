from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    source_url: str
    source_type: str = "html"
    published_at: Optional[datetime] = None
