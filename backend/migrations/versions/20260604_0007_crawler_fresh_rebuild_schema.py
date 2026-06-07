"""add crawler fresh rebuild schema

Revision ID: 20260604_0007
Revises: 20260529_0006
Create Date: 2026-06-04 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260604_0007"
down_revision = "20260529_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("crawler_documents", sa.Column("canonical_doc_key", sa.String(length=255), nullable=True))
    op.add_column("crawler_documents", sa.Column("canonical_source_url", sa.Text(), nullable=True))
    op.add_column(
        "crawler_documents",
        sa.Column("validity_status", sa.String(length=32), nullable=False, server_default="unknown"),
    )
    op.add_column(
        "crawler_documents",
        sa.Column("index_status", sa.String(length=32), nullable=False, server_default="pending"),
    )
    op.add_column(
        "crawler_documents",
        sa.Column("vector_metadata_status", sa.String(length=32), nullable=False, server_default="pending"),
    )
    op.add_column("crawler_documents", sa.Column("skip_reason", sa.Text(), nullable=True))
    op.add_column("crawler_documents", sa.Column("metadata_hash", sa.String(length=64), nullable=True))
    op.add_column("crawler_documents", sa.Column("attachment_hash", sa.String(length=64), nullable=True))
    op.add_column("crawler_documents", sa.Column("current_index_fingerprint", sa.String(length=64), nullable=True))
    op.add_column("crawler_documents", sa.Column("target_index_fingerprint", sa.String(length=64), nullable=True))
    op.add_column("crawler_documents", sa.Column("rebuild_id", sa.String(length=64), nullable=True))
    op.create_index("ix_crawler_documents_canonical_doc_key", "crawler_documents", ["canonical_doc_key"])
    op.create_index("ix_crawler_documents_rebuild_id", "crawler_documents", ["rebuild_id"])

    op.create_table(
        "crawler_document_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=64), nullable=True),
        sa.Column("department", sa.String(length=255), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_run_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("rebuild_id", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["doc_id"], ["crawler_documents.doc_id"]),
        sa.ForeignKeyConstraint(["source_name"], ["crawler_sources.name"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id", "source_name", "source_url", name="uq_crawler_document_sources_doc_source_url"),
    )
    op.create_index("ix_crawler_document_sources_doc_id", "crawler_document_sources", ["doc_id"])
    op.create_index("ix_crawler_document_sources_source_name", "crawler_document_sources", ["source_name"])
    op.create_index("ix_crawler_document_sources_domain", "crawler_document_sources", ["domain"])
    op.create_index("ix_crawler_document_sources_department", "crawler_document_sources", ["department"])
    op.create_index("ix_crawler_document_sources_last_seen_run_id", "crawler_document_sources", ["last_seen_run_id"])
    op.create_index("ix_crawler_document_sources_rebuild_id", "crawler_document_sources", ["rebuild_id"])

    op.add_column("crawler_document_chunks", sa.Column("error_reason", sa.Text(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("embedded_at", sa.DateTime(), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("embedding_version", sa.String(length=64), nullable=True))
    op.add_column("crawler_document_chunks", sa.Column("index_fingerprint", sa.String(length=64), nullable=True))
    op.create_index("ix_crawler_document_chunks_index_fingerprint", "crawler_document_chunks", ["index_fingerprint"])

    # Fresh rebuild truncates old crawler data before these uniqueness constraints are enforced.
    op.create_unique_constraint(
        "uq_crawler_documents_canonical_doc_key",
        "crawler_documents",
        ["canonical_doc_key"],
    )
    op.create_unique_constraint(
        "uq_crawler_document_chunks_doc_fingerprint_index",
        "crawler_document_chunks",
        ["doc_id", "index_fingerprint", "chunk_index"],
    )

    op.add_column(
        "crawler_ingest_runs",
        sa.Column("run_type", sa.String(length=32), nullable=False, server_default="incremental"),
    )
    op.add_column("crawler_ingest_runs", sa.Column("rebuild_id", sa.String(length=64), nullable=True))
    op.create_index("ix_crawler_ingest_runs_rebuild_id", "crawler_ingest_runs", ["rebuild_id"])


def downgrade() -> None:
    op.drop_index("ix_crawler_ingest_runs_rebuild_id", table_name="crawler_ingest_runs")
    op.drop_column("crawler_ingest_runs", "rebuild_id")
    op.drop_column("crawler_ingest_runs", "run_type")

    op.drop_constraint("uq_crawler_document_chunks_doc_fingerprint_index", "crawler_document_chunks", type_="unique")
    op.drop_constraint("uq_crawler_documents_canonical_doc_key", "crawler_documents", type_="unique")
    op.drop_index("ix_crawler_document_chunks_index_fingerprint", table_name="crawler_document_chunks")
    op.drop_column("crawler_document_chunks", "index_fingerprint")
    op.drop_column("crawler_document_chunks", "embedding_version")
    op.drop_column("crawler_document_chunks", "embedding_model")
    op.drop_column("crawler_document_chunks", "embedded_at")
    op.drop_column("crawler_document_chunks", "error_reason")

    op.drop_index("ix_crawler_document_sources_rebuild_id", table_name="crawler_document_sources")
    op.drop_index("ix_crawler_document_sources_last_seen_run_id", table_name="crawler_document_sources")
    op.drop_index("ix_crawler_document_sources_department", table_name="crawler_document_sources")
    op.drop_index("ix_crawler_document_sources_domain", table_name="crawler_document_sources")
    op.drop_index("ix_crawler_document_sources_source_name", table_name="crawler_document_sources")
    op.drop_index("ix_crawler_document_sources_doc_id", table_name="crawler_document_sources")
    op.drop_table("crawler_document_sources")

    op.drop_index("ix_crawler_documents_rebuild_id", table_name="crawler_documents")
    op.drop_index("ix_crawler_documents_canonical_doc_key", table_name="crawler_documents")
    op.drop_column("crawler_documents", "rebuild_id")
    op.drop_column("crawler_documents", "target_index_fingerprint")
    op.drop_column("crawler_documents", "current_index_fingerprint")
    op.drop_column("crawler_documents", "attachment_hash")
    op.drop_column("crawler_documents", "metadata_hash")
    op.drop_column("crawler_documents", "skip_reason")
    op.drop_column("crawler_documents", "vector_metadata_status")
    op.drop_column("crawler_documents", "index_status")
    op.drop_column("crawler_documents", "validity_status")
    op.drop_column("crawler_documents", "canonical_source_url")
    op.drop_column("crawler_documents", "canonical_doc_key")
