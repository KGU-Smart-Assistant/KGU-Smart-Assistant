import httpx
import pytest

from app.services.translation_service import translate_text


@pytest.mark.anyio
async def test_translate_text_calls_google_translation_api(monkeypatch) -> None:
    seen_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "data": {
                    "translations": [
                        {
                            "translatedText": "Hello",
                            "detectedSourceLanguage": "ko",
                        }
                    ]
                }
            },
        )

    original_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)

    result = await translate_text(
        api_key="test-key",
        text="안녕하세요",
        target_language="en",
        api_url="https://translation.googleapis.com/language/translate/v2",
    )

    assert seen_request is not None
    assert seen_request.url.params["key"] == "test-key"
    assert seen_request.url.params["q"] == "안녕하세요"
    assert seen_request.url.params["target"] == "en"
    assert seen_request.url.params["format"] == "text"
    assert seen_request.url.path == "/language/translate/v2"
    assert result == {
        "translated_text": "Hello",
        "target_language": "en",
        "source_language": "ko",
        "provider": "google",
    }
