from typing import List

from pydantic import Field

from app.schemas.chunk import DocumentChunk


class EmbeddedChunk(DocumentChunk):
    embedding: List[float] = Field(min_length=1)
    embedding_model: str
