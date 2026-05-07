from html import unescape
from typing import Any

import httpx


class TranslationServiceError(Exception):
    pass


async def translate_text(
    *,
    api_key: str,
    text: str,
    target_language: str,
    source_language: str | None = None,
    text_format: str = "text",
    api_url: str = "https://translation.googleapis.com/language/translate/v2",
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "key": api_key,
        "q": text,
        "target": target_language,
        "format": text_format,
    }
    if source_language:
        params["source"] = source_language

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(api_url, params=params)

    if response.status_code >= 400:
        raise TranslationServiceError(
            f"Translation API failed with status {response.status_code}"
        )

    data = response.json()
    translations = data.get("data", {}).get("translations", [])
    if not translations:
        raise TranslationServiceError("Translation API returned no translations")

    translated = translations[0]
    return {
        "translated_text": unescape(translated["translatedText"]),
        "target_language": target_language,
        "source_language": translated.get("detectedSourceLanguage", source_language),
        "provider": "google",
    }
