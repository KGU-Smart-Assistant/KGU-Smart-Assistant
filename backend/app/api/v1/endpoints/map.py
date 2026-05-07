from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.map import KakaoAddressSearchResponse, KakaoKeywordSearchResponse
from app.services.kakao_map_service import (
    KakaoMapConfigurationError,
    KakaoMapService,
    KakaoMapServiceError,
)

router = APIRouter()


@router.get("/keyword", response_model=KakaoKeywordSearchResponse)
def search_places_by_keyword(
    query: str = Query(..., min_length=1),
    x: float | None = Query(default=None, description="Longitude"),
    y: float | None = Query(default=None, description="Latitude"),
    radius: int | None = Query(default=None, ge=0, le=20000),
    page: int = Query(default=1, ge=1, le=45),
    size: int = Query(default=15, ge=1, le=15),
) -> dict:
    service = KakaoMapService()
    try:
        return service.search_keyword(
            query=query,
            x=x,
            y=y,
            radius=radius,
            page=page,
            size=size,
        )
    except KakaoMapConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except KakaoMapServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.get("/address", response_model=KakaoAddressSearchResponse)
def search_address(
    query: str = Query(..., min_length=1),
    page: int = Query(default=1, ge=1, le=45),
    size: int = Query(default=10, ge=1, le=30),
) -> dict:
    service = KakaoMapService()
    try:
        return service.search_address(query=query, page=page, size=size)
    except KakaoMapConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except KakaoMapServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
