from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KguContact


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


def list_department_contacts(db: Session) -> list[dict[str, str | None]]:
    contacts = db.execute(select(KguContact).order_by(KguContact.id)).scalars().all()
    return [_contact_to_dict(contact) for contact in contacts]


def get_department_contact(db: Session, department_id: str) -> dict[str, str | None] | None:
    try:
        contact_id = int(department_id)
    except ValueError:
        return None

    contact = db.get(KguContact, contact_id)
    if contact is None:
        return None

    return _contact_to_dict(contact)
