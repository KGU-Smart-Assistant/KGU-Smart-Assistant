"""앱 기동 시 JSON 기반 장소 데이터를 DB에 채웁니다(서버에 빈 DB만 있어도 동작)."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from app.models import KguContact, KguPlace

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PLACE_JSON_FILES = (
    "kgu_suwon_core_places.json",
    "kgu_suwon_lecture_halls.json",
)
_CONTACT_JSON_FILES = ("kgu_contacts.json",)


def seed_places_from_json(db) -> int:
    """
    app/data/*.json의 장소를 name 기준 upsert.
    반환: 처리한 레코드 수
    """
    count = 0
    for fname in _PLACE_JSON_FILES:
        path = DATA_DIR / fname
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        for row in raw:
            name = row["name"]
            desc = row.get("description")
            lat = float(row["latitude"])
            lng = float(row["longitude"])
            existing = db.execute(select(KguPlace).where(KguPlace.name == name)).scalar_one_or_none()
            if existing is None:
                db.add(KguPlace(name=name, description=desc, latitude=lat, longitude=lng))
            else:
                existing.description = desc
                existing.latitude = lat
                existing.longitude = lng
            count += 1
        db.commit()
    return count


def seed_contacts_from_json(db) -> int:
    """
    app/data/kgu_contacts.json 등 연락처를 name 기준 upsert.
    반환: 처리한 레코드 수
    """
    count = 0
    for fname in _CONTACT_JSON_FILES:
        path = DATA_DIR / fname
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        for row in raw:
            name = row["name"]
            phone = str(row["phone"]).strip()
            desc = row.get("description")
            existing = db.execute(
                select(KguContact).where(
                    KguContact.name == name,
                    KguContact.phone == phone,
                )
            ).scalar_one_or_none()
            if existing is None:
                db.add(KguContact(name=name, phone=phone, description=desc))
            else:
                existing.phone = phone
                existing.description = desc
            count += 1
        db.commit()
    return count
