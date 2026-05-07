import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from app.services import weather_service


def test_weather_service_fetches_open_meteo_and_builds_context(monkeypatch) -> None:
    captured = {}
    forecast_date = (
        weather_service.datetime.now(weather_service._TIMEZONE).date()
        + weather_service.timedelta(days=1)
    )
    forecast_date_text = forecast_date.isoformat()

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "daily": {
                    "time": [forecast_date_text],
                    "weather_code": [61],
                    "temperature_2m_max": [18.0],
                    "temperature_2m_min": [9.0],
                    "precipitation_probability_max": [80],
                    "precipitation_sum": [4.2],
                    "wind_speed_10m_max": [18.0],
                }
            }

    def fake_get(url: str, *, params: dict, timeout: int):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    def fake_answer(user_input: str, context: str) -> str:
        captured["user_input"] = user_input
        captured["context"] = context
        return "내일 수원은 약한 비가 예상됩니다."

    monkeypatch.setattr(weather_service.requests, "get", fake_get)
    monkeypatch.setattr(weather_service, "get_gemini_response_with_context", fake_answer)
    monkeypatch.setattr(weather_service, "_extract_forecast_window", lambda _: (forecast_date, 1))

    report = weather_service.get_weather_response("내일 수원 날씨 알려줘")

    assert report.reply == "내일 수원은 약한 비가 예상됩니다."
    assert report.location_name == "수원"
    assert captured["url"] == "https://api.open-meteo.com/v1/forecast"
    assert captured["params"]["timezone"] == "Asia/Seoul"
    assert captured["params"]["latitude"] == 37.2636
    assert forecast_date_text in captured["context"]
    assert "약한 비" in captured["context"]
    assert "precipitation probability 80%" in captured["context"]
