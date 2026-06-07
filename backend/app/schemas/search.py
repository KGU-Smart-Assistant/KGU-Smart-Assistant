from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    domain: Optional[
        Literal[
            "scholarship",
            "course_registration",
            "academic_calendar",
            "academic_status",
            "major_change",
            "multi_major",
            "admission_transfer",
            "teaching_certification",
            "graduation",
            "tuition",
            "document_materials",
            "student_life",
            "career_support",
            "international_exchange",
            "department_notice",
            "general_notice",
            "faq",
            "unknown",
        ]
    ] = None
    category: Optional[str] = Field(default=None, description="Deprecated. Use domain.")
    detail: Optional[
        Literal[
            "period",
            "eligibility",
            "procedure",
            "required_documents",
            "benefit",
            "announcement_lookup",
            "summary",
            "unknown",
        ]
    ] = None


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    title: str
    source_url: str
    domain: Optional[str] = None
    category: Optional[str] = Field(default=None, description="Deprecated. Mirrors domain for old clients.")
    department: Optional[str] = None
    source_name: Optional[str] = None
    section_title: Optional[str] = None
    section_kind: Optional[str] = None
    vector_point_id: Optional[str] = None
    published_at: Optional[str] = None
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
