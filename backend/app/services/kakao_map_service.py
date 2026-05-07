from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings

KAKAO_LOCAL_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"


class KakaoMapServiceError(Exception):
    """Raised when Kakao APIs return an error or cannot be reached."""


class KakaoMapConfigurationError(KakaoMapServiceError):
    """Raised when Kakao API configuration is missing."""


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"KakaoAK {api_key}",
        "Content-Type": "application/json",
    }


async def search_place(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    query: str,
    x: str | None = None,
    y: str | None = None,
    radius: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query, "size": 1}
    if x is not None and y is not None:
        params["x"] = x
        params["y"] = y
    if radius is not None:
        params["radius"] = radius

    response = await client.get(
        KAKAO_LOCAL_KEYWORD_URL,
        headers=_auth_headers(api_key),
        params=params,
    )
    if response.status_code >= 400:
        raise KakaoMapServiceError(
            f"Kakao place search failed with status {response.status_code}"
        )

    payload = response.json()
    documents = payload.get("documents", [])
    if not documents:
        raise KakaoMapServiceError(f"No Kakao place found for query: {query}")

    return documents[0]


async def get_driving_directions(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    origin_x: str,
    origin_y: str,
    destination_x: str,
    destination_y: str,
) -> dict[str, Any]:
    response = await client.get(
        KAKAO_DIRECTIONS_URL,
        headers=_auth_headers(api_key),
        params={
            "origin": f"{origin_x},{origin_y}",
            "destination": f"{destination_x},{destination_y}",
            "priority": "RECOMMEND",
        },
    )
    if response.status_code >= 400:
        raise KakaoMapServiceError(
            f"Kakao directions failed with status {response.status_code}"
        )

    return response.json()


async def get_navigation_route(
    *,
    api_key: str,
    origin: str,
    destination: str,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        origin_place = await search_place(client, api_key=api_key, query=origin)
        destination_place = await search_place(
            client,
            api_key=api_key,
            query=destination,
            x=origin_place.get("x"),
            y=origin_place.get("y"),
            radius=20000,
        )
        directions = await get_driving_directions(
            client,
            api_key=api_key,
            origin_x=origin_place["x"],
            origin_y=origin_place["y"],
            destination_x=destination_place["x"],
            destination_y=destination_place["y"],
        )

    return {
        "origin": origin_place,
        "destination": destination_place,
        "directions": directions,
    }


class KakaoMapService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = settings.kakao_local_base_url,
        timeout: float = 5.0,
    ) -> None:
        self.api_key = (
            api_key
            or settings.kakao_rest_api_key
            or settings.kakao_map_api_key
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search_keyword(
        self,
        query: str,
        *,
        x: float | None = None,
        y: float | None = None,
        radius: int | None = None,
        page: int = 1,
        size: int = 15,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query,
            "page": page,
            "size": size,
        }
        if x is not None:
            params["x"] = x
        if y is not None:
            params["y"] = y
        if radius is not None:
            params["radius"] = radius

        return self._get("/v2/local/search/keyword.json", params=params)

    def search_address(
        self,
        query: str,
        *,
        page: int = 1,
        size: int = 10,
    ) -> dict[str, Any]:
        return self._get(
            "/v2/local/search/address.json",
            params={"query": query, "page": page, "size": size},
        )

    def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise KakaoMapConfigurationError(
                "KAKAO_REST_API_KEY or KAKAO_MAP_API_KEY is not configured."
            )

        try:
            response = httpx.get(
                f"{self.base_url}{path}",
                headers={"Authorization": f"KakaoAK {self.api_key}"},
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise KakaoMapServiceError(
                f"Kakao Local API returned {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise KakaoMapServiceError("Kakao Local API request failed.") from exc

        data = response.json()
        if not isinstance(data, dict):
            raise KakaoMapServiceError("Kakao Local API returned an invalid response.")
        return data
