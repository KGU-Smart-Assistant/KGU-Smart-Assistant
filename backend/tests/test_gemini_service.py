import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.services import gemini_service


def test_gemini_call_includes_runtime_date_context(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        text = "ok"

    class FakeModels:
        def generate_content(self, *, model: str, contents: str):
            captured["model"] = model
            captured["contents"] = contents
            return FakeResponse()

    class FakeClient:
        def __init__(self, *, api_key: str):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(gemini_service.genai, "Client", FakeClient)

    reply = gemini_service.get_gemini_response("2026년 날씨 예측해줘")

    assert reply == "ok"
    assert captured["model"] == gemini_service.settings.gemini_model
    assert "Current date:" in captured["contents"]
    assert "Timezone: Asia/Seoul" in captured["contents"]
    assert "Do not use the model training cutoff as today's date." in captured["contents"]
    assert "real-time weather data is required" in captured["contents"]
    assert "2026년 날씨 예측해줘" in captured["contents"]
