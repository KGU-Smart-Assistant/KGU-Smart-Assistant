import httpx
import pytest

from app.services.kakao_map_service import (
    KAKAO_DIRECTIONS_URL,
    KAKAO_LOCAL_KEYWORD_URL,
    KakaoMapConfigurationError,
    KakaoMapService,
    get_driving_directions,
    search_place,
)


@pytest.mark.anyio
async def test_search_place_calls_kakao_keyword_api() -> None:
    seen_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "id": "1",
                        "place_name": "KGU",
                        "address_name": "Suwon",
                        "x": "127.036",
                        "y": "37.300",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        place = await search_place(client, api_key="test-key", query="KGU")

    assert seen_request is not None
    assert str(seen_request.url).startswith(KAKAO_LOCAL_KEYWORD_URL)
    assert seen_request.headers["Authorization"] == "KakaoAK test-key"
    assert place["place_name"] == "KGU"


@pytest.mark.anyio
async def test_get_driving_directions_calls_kakao_navigation_api() -> None:
    seen_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json={"routes": [{"summary": {"distance": 1200}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        directions = await get_driving_directions(
            client,
            api_key="test-key",
            origin_x="127.036",
            origin_y="37.300",
            destination_x="127.040",
            destination_y="37.305",
        )

    assert seen_request is not None
    assert str(seen_request.url).startswith(KAKAO_DIRECTIONS_URL)
    assert seen_request.headers["Authorization"] == "KakaoAK test-key"
    assert seen_request.url.params["origin"] == "127.036,37.300"
    assert seen_request.url.params["destination"] == "127.040,37.305"
    assert directions["routes"][0]["summary"]["distance"] == 1200


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_search_keyword_calls_kakao_local_api(monkeypatch):
    captured = {}

    def fake_get(url, *, headers, params, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse({"meta": {"total_count": 0}, "documents": []})

    monkeypatch.setattr("app.services.kakao_map_service.httpx.get", fake_get)

    service = KakaoMapService(
        api_key="test-key",
        base_url="https://dapi.kakao.com",
        timeout=3.0,
    )
    result = service.search_keyword(
        "library",
        x=127.03645,
        y=37.30125,
        radius=1000,
        page=2,
        size=10,
    )

    assert result == {"meta": {"total_count": 0}, "documents": []}
    assert captured["url"] == "https://dapi.kakao.com/v2/local/search/keyword.json"
    assert captured["headers"] == {"Authorization": "KakaoAK test-key"}
    assert captured["params"] == {
        "query": "library",
        "x": 127.03645,
        "y": 37.30125,
        "radius": 1000,
        "page": 2,
        "size": 10,
    }
    assert captured["timeout"] == 3.0


def test_search_address_requires_api_key(monkeypatch):
    monkeypatch.setattr("app.services.kakao_map_service.settings.kakao_rest_api_key", None)
    monkeypatch.setattr("app.services.kakao_map_service.settings.kakao_map_api_key", None)

    service = KakaoMapService(api_key=None)

    with pytest.raises(KakaoMapConfigurationError):
        service.search_address("Suwon")
