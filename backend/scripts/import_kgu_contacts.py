from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.seed import seed_contacts_from_json
from app.db.session import SessionLocal
from app.models import KguContact


DEFAULT_URL = "https://www.kyonggi.ac.kr/www/searchTnTelnoListU.do"
DEFAULT_KEY = "8816"
DEFAULT_OUTPUT = Path("app/data/kgu_contacts.json")
PHONE_PATTERN = re.compile(r"\d{2,4}-\d{3,4}-\d{4}|\d{4}-\d{4}")


@dataclass(frozen=True)
class ImportedContact:
    name: str
    phone: str
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Kyonggi University phone directory into kgu_contacts seed data."
    )
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-db", action="store_true", help="Only write JSON; do not seed DB.")
    parser.add_argument(
        "--sync-db",
        action="store_true",
        help="Delete kgu_contacts rows that are not present in the imported official directory.",
    )
    args = parser.parse_args()

    contacts = fetch_contacts(
        base_url=args.base_url,
        key=args.key,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps([asdict(contact) for contact in contacts], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    if not args.no_db:
        with SessionLocal() as db:
            seed_contacts_from_json(db)
            if args.sync_db:
                delete_stale_contacts(db, contacts)

    print(f"Imported contacts: {len(contacts)}")
    print(f"Wrote: {output_path}")


def fetch_contacts(
    base_url: str,
    key: str,
    max_pages: int,
    timeout: int,
) -> list[ImportedContact]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        }
    )

    contacts: list[ImportedContact] = []
    seen: set[tuple[str, str]] = set()
    previous_first_id: str | None = None

    for page in range(1, max_pages + 1):
        response = session.get(
            base_url,
            params={"key": key, "cpn": page},
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        page_contacts, first_id = parse_contacts_page(response.text)
        if not page_contacts:
            break
        if page > 1 and first_id is not None and first_id == previous_first_id:
            break

        previous_first_id = first_id
        for contact in page_contacts:
            key_pair = (contact.name, contact.phone)
            if key_pair in seen:
                continue
            seen.add(key_pair)
            contacts.append(contact)

    return contacts


def delete_stale_contacts(db, contacts: list[ImportedContact]) -> int:
    official_pairs = {(contact.name, contact.phone) for contact in contacts}
    deleted = 0
    for contact in db.query(KguContact).all():
        if (contact.name, contact.phone) in official_pairs:
            continue
        db.delete(contact)
        deleted += 1
    db.commit()
    return deleted


def parse_contacts_page(html: str) -> tuple[list[ImportedContact], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return [], None

    contacts: list[ImportedContact] = []
    first_id: str | None = None
    for row in table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) < 4:
            continue

        official_id, campus, label, phone = cells[:4]
        phone = _normalize_phone(phone)
        if not phone:
            continue
        if first_id is None:
            first_id = official_id

        label = _normalize_spaces(label)
        campus = _normalize_spaces(campus)
        name = _normalize_spaces(f"경기대학교 {campus} {label}")
        description = _normalize_spaces(
            f"경기대학교 공식 전화번호검색, 공식번호 {official_id}, 캠퍼스 {campus}"
        )
        contacts.append(ImportedContact(name=name, phone=phone, description=description))

    return contacts, first_id


def _normalize_phone(value: str) -> str:
    value = _normalize_spaces(value)
    if value in {"", "-"}:
        return ""
    match = PHONE_PATTERN.search(value)
    return match.group(0) if match else ""


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


if __name__ == "__main__":
    main()
