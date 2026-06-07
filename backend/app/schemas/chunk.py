from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    source_url: str
    source_type: str = "html"
    domain: Optional[str] = None
    department: Optional[str] = None
    published_at: Optional[datetime] = None
    content: Optional[str] = None
    embedding_text: Optional[str] = None
    chunk_text_hash: Optional[str] = None
    section_title: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)
    section_kind: Optional[str] = None
    source_name: Optional[str] = None
    embedding_eligibility: Optional[str] = None
    chunk_quality_status: Optional[str] = None
    document_quality_status: Optional[str] = None
    content_token_count: Optional[int] = None
    embedding_token_count: Optional[int] = None
    valid_until: Optional[datetime] = None
    chunk_valid_until: Optional[datetime] = None
    prepared_body_hash: Optional[str] = None
    prepared_artifact_hash: Optional[str] = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
