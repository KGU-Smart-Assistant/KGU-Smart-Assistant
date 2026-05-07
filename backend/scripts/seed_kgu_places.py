from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable

import requests
from sqlalchemy import select

from app.db.session import SessionLocal, init_db
from app.models import KguPlace


@dataclass(frozen=True)
class SeedPlace:
    name: str
    description: str | None = None


KGU_SUWON_SEED: list[SeedPlace] = [
    # 강의동(통칭/별칭은 학교 안내/커뮤니티에서 자주 쓰이는 형태로 함께 표기 권장)
    SeedPlace("경기대학교 수원캠퍼스 제1강의동(진리관)"),
    SeedPlace("경기대학교 수원캠퍼스 제2강의동(성신관)"),
    SeedPlace("경기대학교 수원캠퍼스 제3강의동(애경관)"),
    SeedPlace("경기대학교 수원캠퍼스 제4강의동(예지관)"),
    SeedPlace("경기대학교 수원캠퍼스 제5강의동(덕문관)"),
    SeedPlace("경기대학교 수원캠퍼스 제6강의동(광교관)"),
    SeedPlace("경기대학교 수원캠퍼스 제7강의동(집현관)"),
    SeedPlace("경기대학교 수원캠퍼스 제8강의동(육영관)"),
    SeedPlace("경기대학교 수원캠퍼스 제9강의동(호연관)"),
    # 주요 시설(대표 키워드 위주)
    SeedPlace("경기대학교 수원캠퍼스 중앙도서관", "중앙도서관"),
    SeedPlace("경기대학교 수원캠퍼스 학생회관", "학생회관"),
    SeedPlace("경기대학교 수원캠퍼스 복지관", "복지관"),
    SeedPlace("경기대학교 수원캠퍼스 텔레컨벤션센터", "Tel-Convention Center"),
    SeedPlace("경기대학교 수원캠퍼스 박물관", "박물관"),
    SeedPlace("경기대학교 수원캠퍼스 산학협력단", "산학협력단"),
    SeedPlace("경기대학교 수원캠퍼스 종합강의동", "종합강의동"),
    SeedPlace("경기대학교 수원캠퍼스 공학관", "공학관"),
    SeedPlace("경기대학교 수원캠퍼스 운동장", "운동장"),
    # 출입구/정류장(사용자 질문에 자주 등장)
    SeedPlace("경기대학교 수원캠퍼스 정문", "정문"),
    SeedPlace("경기대학교 수원캠퍼스 후문", "후문"),
]


def geocode_google(query: str, api_key: str, *, timeout_s: float = 15.0) -> tuple[float, float] | None:
    """
    Google Geocoding API로 query를 좌표로 변환.
    반환: (lat, lng) 또는 None
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    resp = requests.get(url, params={"address": query, "key": api_key, "language": "ko"}, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        return None
    results = data.get("results") or []
    if not results:
        return None
    loc = results[0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])


def upsert_places(
    seeds: Iterable[SeedPlace],
    *,
    api_key: str,
    sleep_s: float = 0.1,
) -> list[dict]:
    """
    seeds를 Google로 지오코딩해 SQLite(app.db)에 upsert.
    반환: 저장된 레코드 요약 리스트
    """
    init_db()
    saved: list[dict] = []

    with SessionLocal() as db:
        for seed in seeds:
            coords = geocode_google(seed.name, api_key)
            if coords is None:
                saved.append({"name": seed.name, "ok": False, "reason": "geocode_failed"})
                continue

            lat, lng = coords
            existing = db.execute(select(KguPlace).where(KguPlace.name == seed.name)).scalar_one_or_none()
            if existing is None:
                db.add(
                    KguPlace(
                        name=seed.name,
                        description=seed.description,
                        latitude=lat,
                        longitude=lng,
                    )
                )
            else:
                existing.description = seed.description
                existing.latitude = lat
                existing.longitude = lng

            db.commit()
            saved.append({"name": seed.name, "ok": True, "latitude": lat, "longitude": lng})

            if sleep_s:
                time.sleep(sleep_s)

    return saved


def main() -> None:
    """
    사용법:
      $env:GOOGLE_MAPS_API_KEY="..."
      python scripts/seed_kgu_places.py
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_MAPS_API_KEY (or GOOGLE_API_KEY) env var.")

    result = upsert_places(KGU_SUWON_SEED, api_key=api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

