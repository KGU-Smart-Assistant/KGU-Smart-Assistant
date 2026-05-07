from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: Optional[
        Literal[
            "notice",
            "academic",
            "scholarship",
            "faq",
            "materials",
            "academic_schedule",
        ]
    ] = None


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    title: str
    source_url: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
