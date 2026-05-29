from __future__ import annotations

from collections import OrderedDict

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import KguInfoLink


def list_info_link_groups(db: Session) -> list[dict]:
    try:
        rows = (
            db.execute(
                select(KguInfoLink)
                .where(KguInfoLink.is_active.is_(True))
                .order_by(KguInfoLink.group_order, KguInfoLink.link_order, KguInfoLink.id)
            )
            .scalars()
            .all()
        )
    except SQLAlchemyError:
        return []

    groups: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        group = groups.setdefault(
            row.group_id,
            {
                "id": row.group_id,
                "title": row.group_title,
                "links": [],
            },
        )
        group["links"].append({"label": row.label, "url": row.url})

    return list(groups.values())
