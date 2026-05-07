from fastapi.testclient import TestClient

from app.api.v1.endpoints import translation
from app.main import app


def test_translate_endpoint_returns_translated_text(monkeypatch) -> None:
    async def fake_translate_text(**kwargs):
        return {
            "translated_text": "Hello",
            "target_language": kwargs["target_language"],
            "source_language": "ko",
            "provider": "google",
        }

    monkeypatch.setattr(translation.settings, "translation_api_key", "test-key")
    monkeypatch.setattr(translation.settings, "translation_provider", "google")
    monkeypatch.setattr(translation, "translate_text", fake_translate_text)

    response = TestClient(app).post(
        "/api/v1/translation/translate",
        json={"text": "안녕하세요", "target_language": "en"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "translated_text": "Hello",
        "target_language": "en",
        "source_language": "ko",
        "provider": "google",
    }
