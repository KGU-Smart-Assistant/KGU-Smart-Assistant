"""expand crawler chunk metadata

Revision ID: 20260606_0008
Revises: 20260604_0007
Create Date: 2026-06-06 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260606_0008"
down_revision = "20260604_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("crawler_document_chunks", sa.Column("content", sa.Text(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("embedding_text", sa.Text(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("chunk_text_hash", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("section_title", sa.Text(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("section_path", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("section_kind", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("embedding_eligibility", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("chunk_quality_status", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("document_quality_status", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("content_token_count", sa.Integer(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("embedding_token_count", sa.Integer(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("chunk_valid_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("domain", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("department", sa.String(length=255), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("source_name", sa.String(length=255), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("prepared_body_hash", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("prepared_artifact_hash", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("vector_point_id", sa.String(length=255), nullable=True))
    op.add_column(
        "crawler_document_chunks",
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    op.create_index("ix_crawler_document_chunks_chunk_text_hash", "crawler_document_chunks", ["chunk_text_hash"])
    op.create_index("ix_crawler_document_chunks_section_kind", "crawler_document_chunks", ["section_kind"])
    op.create_index("ix_crawler_document_chunks_embedding_eligibility", "crawler_document_chunks", ["embedding_eligibility"])
    op.create_index("ix_crawler_document_chunks_chunk_quality_status", "crawler_document_chunks", ["chunk_quality_status"])
    op.create_index("ix_crawler_document_chunks_document_quality_status", "crawler_document_chunks", ["document_quality_status"])
    op.create_index("ix_crawler_document_chunks_valid_until", "crawler_document_chunks", ["valid_until"])
    op.create_index("ix_crawler_document_chunks_chunk_valid_until", "crawler_document_chunks", ["chunk_valid_until"])
    op.create_index("ix_crawler_document_chunks_domain", "crawler_document_chunks", ["domain"])
    op.create_index("ix_crawler_document_chunks_department", "crawler_document_chunks", ["department"])
    op.create_index("ix_crawler_document_chunks_source_name", "crawler_document_chunks", ["source_name"])
    op.create_index("ix_crawler_document_chunks_vector_point_id", "crawler_document_chunks", ["vector_point_id"])


def downgrade() -> None:
    op.drop_index("ix_crawler_document_chunks_vector_point_id", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_source_name", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_department", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_domain", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_chunk_valid_until", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_valid_until", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_document_quality_status", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_chunk_quality_status", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_embedding_eligibility", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_section_kind", table_name="crawler_document_chunks")
    op.drop_index("ix_crawler_document_chunks_chunk_text_hash", table_name="crawler_document_chunks")

    op.drop_column("crawler_document_chunks", "metadata_json")
    op.drop_column("crawler_document_chunks", "vector_point_id")
    op.drop_column("crawler_document_chunks", "prepared_artifact_hash")
    op.drop_column("crawler_document_chunks", "prepared_body_hash")
    op.drop_column("crawler_document_chunks", "source_name")
    op.drop_column("crawler_document_chunks", "department")
    op.drop_column("crawler_document_chunks", "domain")
    op.drop_column("crawler_document_chunks", "chunk_valid_until")
    op.drop_column("crawler_document_chunks", "valid_until")
    op.drop_column("crawler_document_chunks", "embedding_token_count")
    op.drop_column("crawler_document_chunks", "content_token_count")
    op.drop_column("crawler_document_chunks", "document_quality_status")
    op.drop_column("crawler_document_chunks", "chunk_quality_status")
    op.drop_column("crawler_document_chunks", "embedding_eligibility")
    op.drop_column("crawler_document_chunks", "section_kind")
    op.drop_column("crawler_document_chunks", "section_path")
    op.drop_column("crawler_document_chunks", "section_title")
    op.drop_column("crawler_document_chunks", "chunk_text_hash")
    op.drop_column("crawler_document_chunks", "embedding_text")
    op.drop_column("crawler_document_chunks", "content")
