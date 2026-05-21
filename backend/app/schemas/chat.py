# 스키마 정의 : 질문과 응답의 형식
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str


class ChatSource(BaseModel):
    type: str
    title: str
    source_url: str | None = None
    score: float | None = None


class RagIntentScore(BaseModel):
    domain: str
    score: float
    matched_keywords: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    intent: str
    route: Literal["llm", "relational_db", "rag", "weather", "multi"] = "llm"
    sources: list[ChatSource] = Field(default_factory=list)
    rag_domain: str | None = None
    rag_domains: list[str] = Field(default_factory=list)
    rag_detail: str | None = None
    source_scope: str | None = None
    rag_confidence: float | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    intent_scores: list[RagIntentScore] = Field(default_factory=list)
    answer_status: Literal["answered", "partial", "insufficient"] = "answered"
    unverified: list[str] = Field(default_factory=list)
