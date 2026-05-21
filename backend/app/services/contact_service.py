from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import KguContact

_FALLBACK_CONTACTS: tuple[dict[str, str | None], ...] = (
    {
        "department_id": "academic_affairs",
        "department_name": "학사지원팀",
        "phone_number": "031-249-0000",
        "description": None,
    },
    {
        "department_id": "student_support",
        "department_name": "학생지원팀",
        "phone_number": "031-249-1111",
        "description": None,
    },
)


def _tel_uri(phone_number: str) -> str:
    digits = "".join(char for char in phone_number if char.isdigit())
    return f"tel:{digits}"


def _contact_to_dict(contact: KguContact) -> dict[str, str | None]:
    department_id = str(contact.id)
    return {
        "department_id": department_id,
        "department_name": contact.name,
        "phone_number": contact.phone,
        "description": contact.description,
        "tel_uri": _tel_uri(contact.phone),
    }


def _fallback_contact_to_dict(contact: dict[str, str | None]) -> dict[str, str | None]:
    phone_number = str(contact["phone_number"])
    return {
        **contact,
        "tel_uri": _tel_uri(phone_number),
    }


def _fallback_contacts() -> list[dict[str, str | None]]:
    return [_fallback_contact_to_dict(contact) for contact in _FALLBACK_CONTACTS]


def list_department_contacts(db: Session) -> list[dict[str, str | None]]:
    try:
        contacts = db.execute(select(KguContact).order_by(KguContact.id)).scalars().all()
    except SQLAlchemyError:
        return _fallback_contacts()
    return [_contact_to_dict(contact) for contact in contacts]


def get_department_contact(db: Session, department_id: str) -> dict[str, str | None] | None:
    fallback = {
        str(contact["department_id"]): _fallback_contact_to_dict(contact)
        for contact in _FALLBACK_CONTACTS
    }
    if department_id in fallback:
        return fallback[department_id]

    try:
        contact_id = int(department_id)
    except ValueError:
        return None

    try:
        contact = db.get(KguContact, contact_id)
    except SQLAlchemyError:
        return None
    if contact is None:
        return None

    return _contact_to_dict(contact)
