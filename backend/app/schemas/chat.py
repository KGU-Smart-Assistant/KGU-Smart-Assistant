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


class ChatResponse(BaseModel):
    reply: str
    intent: str
    route: Literal["llm", "relational_db", "rag", "weather"] = "llm"
    sources: list[ChatSource] = Field(default_factory=list)
    rag_domain: str | None = None
    rag_detail: str | None = None
