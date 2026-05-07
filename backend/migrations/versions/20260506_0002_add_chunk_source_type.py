"""add chunk source type

Revision ID: 20260506_0002
Revises: 20260505_0001
Create Date: 2026-05-06 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260506_0002"
down_revision = "20260505_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "crawler_document_chunks",
        sa.Column(
            "source_type",
            sa.String(length=32),
            nullable=False,
            server_default="html",
        ),
    )
    op.alter_column("crawler_document_chunks", "source_type", server_default=None)


def downgrade() -> None:
    op.drop_column("crawler_document_chunks", "source_type")
