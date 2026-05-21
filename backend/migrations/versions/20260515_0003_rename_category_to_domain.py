"""rename crawler category fields to domain

Revision ID: 20260515_0003
Revises: 20260506_0002
Create Date: 2026-05-15 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "20260515_0003"
down_revision = "20260506_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("crawler_sources", "category", new_column_name="domain")
    op.drop_index("ix_crawler_documents_category", table_name="crawler_documents")
    op.alter_column("crawler_documents", "category", new_column_name="domain")
    op.create_index("ix_crawler_documents_domain", "crawler_documents", ["domain"], unique=False)
    _normalize_domains()


def downgrade() -> None:
    op.drop_index("ix_crawler_documents_domain", table_name="crawler_documents")
    op.alter_column("crawler_documents", "domain", new_column_name="category")
    op.create_index("ix_crawler_documents_category", "crawler_documents", ["category"], unique=False)
    op.alter_column("crawler_sources", "domain", new_column_name="category")


def _normalize_domains() -> None:
    mapping = {
        "academic": "academic_calendar",
        "academic_schedule": "academic_calendar",
        "support": "scholarship",
        "materials": "document_materials",
        "career": "career_support",
        "notice": "general_notice",
    }
    for old, new in mapping.items():
        op.execute(f"UPDATE crawler_sources SET domain = '{new}' WHERE domain = '{old}'")
        op.execute(f"UPDATE crawler_documents SET domain = '{new}' WHERE domain = '{old}'")
