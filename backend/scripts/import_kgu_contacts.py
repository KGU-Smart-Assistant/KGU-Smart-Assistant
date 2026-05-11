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


DEFAULT_URL = "https://www.kyonggi.ac.kr/www/selectTnTelnoListU.do"
DEFAULT_KEY = "5155"
DEFAULT_CAMPUS = "수원"
DEFAULT_OUTPUT = Path("app/data/kgu_contacts.json")
PHONE_PATTERN = re.compile(r"\d{2,4}-\d{3,4}-\d{4}|\d{4}-\d{4}")


@dataclass(frozen=True)
class ImportedContact:
    name: str
    phone: str
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Kyonggi University Suwon campus phone directory into kgu_contacts."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--campus", default=DEFAULT_CAMPUS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-db", action="store_true", help="Only write JSON; do not seed DB.")
    parser.add_argument(
        "--sync-db",
        action="store_true",
        help="Delete kgu_contacts rows that are not present in the imported Suwon directory.",
    )
    args = parser.parse_args()

    contacts = fetch_suwon_contacts(
        url=args.url,
        key=args.key,
        campus=args.campus,
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
                deleted = delete_stale_contacts(db, contacts)
                print(f"Deleted stale contacts: {deleted}")

    print(f"Imported Suwon contacts: {len(contacts)}")
    print(f"Wrote: {output_path}")


def fetch_suwon_contacts(
    url: str,
    key: str,
    campus: str,
    timeout: int,
) -> list[ImportedContact]:
    response = requests.get(
        url,
        params={"key": key, "sc1": campus},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_suwon_contacts_page(response.text, campus=campus)


def parse_suwon_contacts_page(html: str, campus: str = DEFAULT_CAMPUS) -> list[ImportedContact]:
    soup = BeautifulSoup(html, "html.parser")
    contacts: list[ImportedContact] = []
    seen: set[tuple[str, str]] = set()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        header_cells = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"])]
        if len(header_cells) < 2 or "연락처" not in header_cells[-1]:
            continue

        section = _table_section(table)
        group = _normalize_spaces(header_cells[0])
        for row in rows[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) < 2:
                continue

            role = _normalize_spaces(cells[0])
            phone = _normalize_phone(cells[-1])
            if not role or not phone:
                continue

            name = _normalize_spaces(f"경기대학교 {campus}캠퍼스 {section} {group} {role}")
            description = _normalize_spaces(
                f"경기대학교 {campus}캠퍼스 공식 전화번호, 섹션 {section}, 구분 {group}"
            )
            key_pair = (name, phone)
            if key_pair in seen:
                continue
            seen.add(key_pair)
            contacts.append(ImportedContact(name=name, phone=phone, description=description))

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


def _table_section(table) -> str:
    heading = table.find_previous(["h2", "h3", "h4"])
    if heading is None:
        return "수원캠퍼스 전화번호"
    return _normalize_spaces(heading.get_text(" ", strip=True))


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
