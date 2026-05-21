"""add unique constraint for contact names

Revision ID: 20260511_0003
Revises: 20260506_0002
Create Date: 2026-05-11 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "20260511_0003"
down_revision = "20260506_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_kgu_contacts_name",
        "kgu_contacts",
        ["name"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_kgu_contacts_name",
        "kgu_contacts",
        type_="unique",
    )
