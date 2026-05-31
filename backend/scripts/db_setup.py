"""Create PostgreSQL tables and seed JSON-backed application data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.seed import seed_contacts_from_json, seed_info_links_from_json, seed_places_from_json
from app.db.session import SessionLocal, init_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-places", action="store_true", help="Skip kgu_places seed")
    parser.add_argument("--no-contacts", action="store_true", help="Skip kgu_contacts seed")
    parser.add_argument("--no-info-links", action="store_true", help="Skip kgu_info_links seed")
    args = parser.parse_args()

    init_db()

    with SessionLocal() as db:
        places_count = 0
        contacts_count = 0
        info_links_count = 0

        if not args.no_places:
            places_count = seed_places_from_json(db)
        if not args.no_contacts:
            contacts_count = seed_contacts_from_json(db)
        if not args.no_info_links:
            info_links_count = seed_info_links_from_json(db)

    print(
        f"OK: kgu_places={places_count}, "
        f"kgu_contacts={contacts_count}, "
        f"kgu_info_links={info_links_count}"
    )


if __name__ == "__main__":
    main()
