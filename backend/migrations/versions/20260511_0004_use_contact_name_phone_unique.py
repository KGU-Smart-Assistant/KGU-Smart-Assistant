"""use name and phone unique key for contacts

Revision ID: 20260511_0004
Revises: 20260511_0003
Create Date: 2026-05-11 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "20260511_0004"
down_revision = "20260511_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_kgu_contacts_name",
        "kgu_contacts",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_kgu_contacts_name_phone",
        "kgu_contacts",
        ["name", "phone"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_kgu_contacts_name_phone",
        "kgu_contacts",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_kgu_contacts_name",
        "kgu_contacts",
        ["name"],
    )
