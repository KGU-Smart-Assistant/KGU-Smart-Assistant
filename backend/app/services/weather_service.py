from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from zoneinfo import ZoneInfo

import requests

from app.services.gemini_service import get_gemini_response_with_context


_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEZONE = ZoneInfo("Asia/Seoul")
_MAX_FORECAST_DAYS = 16

_LOCATIONS: dict[str, tuple[str, float, float]] = {
    "경기대": ("경기대학교 수원캠퍼스", 37.3005, 127.0358),
    "경기대학교": ("경기대학교 수원캠퍼스", 37.3005, 127.0358),
    "수원": ("수원", 37.2636, 127.0286),
    "서울": ("서울", 37.5665, 126.9780),
    "인천": ("인천", 37.4563, 126.7052),
    "부산": ("부산", 35.1796, 129.0756),
    "대전": ("대전", 36.3504, 127.3845),
    "대구": ("대구", 35.8714, 128.6014),
    "광주": ("광주", 35.1595, 126.8526),
}
_DEFAULT_LOCATION = _LOCATIONS["경기대"]

_WEATHER_CODE_LABELS = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "서리 안개",
    51: "약한 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    95: "뇌우",
    96: "우박을 동반한 뇌우",
    99: "강한 우박을 동반한 뇌우",
}


@dataclass(frozen=True)
class WeatherReport:
    reply: str
    location_name: str
    source_url: str = _FORECAST_URL


def get_weather_response(user_input: str) -> WeatherReport:
    location_name, latitude, longitude = _extract_location(user_input)
    start_date, forecast_days = _extract_forecast_window(user_input)
    today = datetime.now(_TIMEZONE).date()
    days_from_today = (start_date - today).days

    if days_from_today < 0:
        return WeatherReport(
            reply="과거 날씨는 예보 API가 아니라 과거 관측/기록 API가 필요합니다. 오늘 이후 날짜로 질문해 주세요.",
            location_name=location_name,
        )
    if days_from_today >= _MAX_FORECAST_DAYS:
        return WeatherReport(
            reply=f"Open-Meteo 예보는 최대 {_MAX_FORECAST_DAYS}일 범위까지만 조회할 수 있습니다. 더 가까운 날짜로 질문해 주세요.",
            location_name=location_name,
        )

    forecast_days = min(max(forecast_days, 1), _MAX_FORECAST_DAYS - days_from_today)
    payload = _fetch_forecast(latitude=latitude, longitude=longitude, forecast_days=_MAX_FORECAST_DAYS)
    daily_rows = _select_daily_rows(payload, start_date=start_date, forecast_days=forecast_days)

    if not daily_rows:
        return WeatherReport(
            reply="해당 날짜의 날씨 예보를 찾지 못했습니다. 날짜를 다시 확인해 주세요.",
            location_name=location_name,
        )

    context = _format_weather_context(location_name, daily_rows)
    reply = get_gemini_response_with_context(user_input=user_input, context=context)
    return WeatherReport(reply=reply, location_name=location_name)


def _fetch_forecast(latitude: float, longitude: float, forecast_days: int) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]
        ),
        "forecast_days": forecast_days,
        "timezone": "Asia/Seoul",
    }
    response = requests.get(_FORECAST_URL, params=params, timeout=5)
    response.raise_for_status()
    return response.json()


def _extract_location(user_input: str) -> tuple[str, float, float]:
    for keyword, location in _LOCATIONS.items():
        if keyword in user_input:
            return location
    return _DEFAULT_LOCATION


def _extract_forecast_window(user_input: str) -> tuple[date, int]:
    today = datetime.now(_TIMEZONE).date()

    iso_date = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", user_input)
    if iso_date:
        return date.fromisoformat(iso_date.group(0)), 1

    if "이번 주" in user_input or "주간" in user_input or "일주일" in user_input:
        return today, 7
    if "모레" in user_input:
        return today + timedelta(days=2), 1
    if "내일" in user_input:
        return today + timedelta(days=1), 1
    return today, 1


def _select_daily_rows(payload: dict, start_date: date, forecast_days: int) -> list[dict]:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    selected = []

    for index, value in enumerate(times):
        row_date = date.fromisoformat(value)
        if start_date <= row_date < start_date + timedelta(days=forecast_days):
            selected.append(
                {
                    "date": value,
                    "weather": _weather_label(_safe_list_get(daily, "weather_code", index)),
                    "temperature_max": _safe_list_get(daily, "temperature_2m_max", index),
                    "temperature_min": _safe_list_get(daily, "temperature_2m_min", index),
                    "precipitation_probability": _safe_list_get(
                        daily,
                        "precipitation_probability_max",
                        index,
                    ),
                    "precipitation_sum": _safe_list_get(daily, "precipitation_sum", index),
                    "wind_speed_max": _safe_list_get(daily, "wind_speed_10m_max", index),
                }
            )

    return selected


def _safe_list_get(payload: dict, key: str, index: int):
    values = payload.get(key) or []
    if index >= len(values):
        return None
    return values[index]


def _weather_label(code) -> str:
    if code is None:
        return "정보 없음"
    return _WEATHER_CODE_LABELS.get(int(code), f"날씨 코드 {code}")


def _format_weather_context(location_name: str, daily_rows: list[dict]) -> str:
    lines = [
        "Weather forecast data from Open-Meteo.",
        f"Location: {location_name}",
        "Timezone: Asia/Seoul",
        "Use only this forecast data. Do not add unsupported weather details.",
    ]

    for row in daily_rows:
        lines.append(
            (
                f"- {row['date']}: {row['weather']}, "
                f"min {row['temperature_min']}°C, max {row['temperature_max']}°C, "
                f"precipitation probability {row['precipitation_probability']}%, "
                f"precipitation {row['precipitation_sum']}mm, "
                f"max wind {row['wind_speed_max']}km/h"
            )
        )

    return "\n".join(lines)
