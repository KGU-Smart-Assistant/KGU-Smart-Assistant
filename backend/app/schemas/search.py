from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: Optional[
        Literal[
            "notice",
            "academic",
            "scholarship",
            "support",
            "faq",
            "materials",
            "academic_schedule",
            "graduation",
            "career",
            "student_life",
        ]
    ] = None


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    title: str
    source_url: str
    category: Optional[str] = None
    department: Optional[str] = None
    published_at: Optional[str] = None
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
