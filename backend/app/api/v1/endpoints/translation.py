from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.translation import TranslationRequest, TranslationResponse
from app.services.translation_service import TranslationServiceError, translate_text

router = APIRouter()


@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest) -> TranslationResponse:
    if settings.translation_provider != "google":
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported translation provider: {settings.translation_provider}",
        )
    if not settings.translation_api_key:
        raise HTTPException(status_code=500, detail="TRANSLATION_API_KEY is not configured")

    try:
        result = await translate_text(
            api_key=settings.translation_api_key,
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language,
            text_format=request.format,
            api_url=settings.google_translation_api_url,
        )
    except TranslationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TranslationResponse(**result)
