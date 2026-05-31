"""add kgu info links

Revision ID: 20260529_0006
Revises: 20260521_0005
Create Date: 2026-05-29 00:06:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260529_0006"
down_revision = "20260521_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kgu_info_links",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("group_id", sa.String(length=64), nullable=False),
        sa.Column("group_title", sa.String(length=255), nullable=False),
        sa.Column("group_order", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("link_order", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("group_id", "label", "url", name="uq_kgu_info_links_group_label_url"),
    )
    op.create_index(op.f("ix_kgu_info_links_group_id"), "kgu_info_links", ["group_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_kgu_info_links_group_id"), table_name="kgu_info_links")
    op.drop_table("kgu_info_links")
