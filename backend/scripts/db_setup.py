"""
PostgreSQL(DB)에 테이블 생성 + JSON 시드 데이터를 채웁니다.

기본 동작:
  - app/db/session.py의 Base.metadata.create_all()
  - app/db/seed.py의 seed_places_from_json / seed_contacts_from_json()

용도:
  - docker-compose로 DB를 올린 뒤, 한 번 수동으로 초기화할 때 사용
  - 또는 API 서버 시작 전에 데이터가 필요할 때 사용
"""

from __future__ import annotations

import argparse

from app.db.seed import seed_contacts_from_json, seed_places_from_json
from app.db.session import SessionLocal, init_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-places", action="store_true", help="kgu_places 시드 제외")
    parser.add_argument("--no-contacts", action="store_true", help="kgu_contacts 시드 제외")
    args = parser.parse_args()

    init_db()

    with SessionLocal() as db:
        places_count = 0
        contacts_count = 0

        if not args.no_places:
            places_count = seed_places_from_json(db)
        if not args.no_contacts:
            contacts_count = seed_contacts_from_json(db)

    print(f"OK: kgu_places={places_count}, kgu_contacts={contacts_count}")


if __name__ == "__main__":
    main()

