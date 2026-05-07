"""
수원캠퍼스 강의동(제1~제9) JSON을 읽어 SQLite kgu_places에 upsert합니다.

사용 (Backend 루트에서):
  python scripts/load_kgu_lecture_halls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models import KguPlace  # noqa: E402

DATA_FILE = ROOT / "app" / "data" / "kgu_suwon_lecture_halls.json"


def main() -> None:
    init_db()
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("JSON must be a list of objects")

    with SessionLocal() as db:
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
        db.commit()

    print(f"OK: {len(raw)} lecture halls upserted from {DATA_FILE}")


if __name__ == "__main__":
    main()
