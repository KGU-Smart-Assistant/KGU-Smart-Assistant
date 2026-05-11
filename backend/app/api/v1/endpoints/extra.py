from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import KguPlace
from app.schemas.contact import DepartmentContact, DepartmentContactListResponse
from app.schemas.map import CampusPlace, CampusPlaceListResponse, MapNavigationResponse
from app.services.contact_service import get_department_contact, list_department_contacts
from app.services.kakao_map_service import KakaoMapServiceError, get_navigation_route

router = APIRouter()


@router.get("/map/navigation", response_model=MapNavigationResponse)
async def get_campus_navigation(
    origin: str = Query(..., min_length=1, description="Start place keyword"),
    destination: str = Query(..., min_length=1, description="End place keyword"),
):
    if not settings.kakao_map_api_key:
        raise HTTPException(status_code=500, detail="KAKAO_MAP_API_KEY is not configured")

    try:
        return await get_navigation_route(
            api_key=settings.kakao_map_api_key,
            origin=origin,
            destination=destination,
        )
    except KakaoMapServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/contacts",
    response_model=DepartmentContactListResponse,
    response_model_exclude_none=True,
)
async def get_department_contacts(db: Session = Depends(get_db)) -> DepartmentContactListResponse:
    return DepartmentContactListResponse(contacts=list_department_contacts(db))


@router.get(
    "/contacts/{department_id}",
    response_model=DepartmentContact,
    response_model_exclude_none=True,
)
async def get_department_contact_by_id(
    department_id: str,
    db: Session = Depends(get_db),
) -> DepartmentContact:
    contact = get_department_contact(db, department_id)
    if contact is None:
        raise HTTPException(status_code=404, detail="Department contact not found")
    return DepartmentContact(**contact)


@router.get("/places", response_model=CampusPlaceListResponse)
async def get_campus_places(db: Session = Depends(get_db)) -> CampusPlaceListResponse:
    places = db.execute(select(KguPlace).order_by(KguPlace.id)).scalars().all()
    return CampusPlaceListResponse(
        places=[
            CampusPlace(
                id=str(place.id),
                name=place.name,
                description=place.description,
                latitude=place.latitude,
                longitude=place.longitude,
            )
            for place in places
        ],
    )


@router.get("/language-settings")
async def get_language_settings():
    return {"supported_languages": ["ko", "en", "ja"]}
