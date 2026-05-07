from typing import Literal

from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=30_000)
    target_language: str = Field(..., min_length=2, max_length=10)
    source_language: str | None = Field(default=None, min_length=2, max_length=10)
    format: Literal["text", "html"] = "text"


class TranslationResponse(BaseModel):
    translated_text: str
    target_language: str
    source_language: str | None = None
    provider: str
