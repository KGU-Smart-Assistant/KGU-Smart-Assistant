"""initial schema

Revision ID: 20260505_0001
Revises:
Create Date: 2026-05-05 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260505_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kgu_contacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "kgu_places",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "crawler_sources",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("department", sa.String(length=255), nullable=True),
        sa.Column("seed_urls_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_crawled_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "crawler_documents",
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("doc_type", sa.String(length=32), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("department", sa.String(length=255), nullable=True),
        sa.Column("author_department", sa.String(length=255), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("collected_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["source_name"], ["crawler_sources.name"]),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index(
        op.f("ix_crawler_documents_category"),
        "crawler_documents",
        ["category"],
        unique=False,
    )
    op.create_index(
        op.f("ix_crawler_documents_content_hash"),
        "crawler_documents",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_crawler_documents_department"),
        "crawler_documents",
        ["department"],
        unique=False,
    )
    op.create_index(
        op.f("ix_crawler_documents_doc_type"),
        "crawler_documents",
        ["doc_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_crawler_documents_source_name"),
        "crawler_documents",
        ["source_name"],
        unique=False,
    )
    op.create_table(
        "crawler_attachments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("attachment_url", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(length=300), nullable=True),
        sa.Column("file_type", sa.String(length=32), nullable=True),
        sa.Column("extraction_status", sa.String(length=32), nullable=False),
        sa.Column("error_reason", sa.Text(), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["doc_id"], ["crawler_documents.doc_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "doc_id",
            "attachment_url",
            name="uq_crawler_attachment_doc_url",
        ),
    )
    op.create_index(
        op.f("ix_crawler_attachments_doc_id"),
        "crawler_attachments",
        ["doc_id"],
        unique=False,
    )
    op.create_table(
        "crawler_document_chunks",
        sa.Column("chunk_id", sa.String(length=255), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["doc_id"], ["crawler_documents.doc_id"]),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index(
        op.f("ix_crawler_document_chunks_content_hash"),
        "crawler_document_chunks",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_crawler_document_chunks_doc_id"),
        "crawler_document_chunks",
        ["doc_id"],
        unique=False,
    )
    op.create_table(
        "crawler_ingest_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("status_reason", sa.Text(), nullable=False),
        sa.Column("raw_documents", sa.Integer(), nullable=False),
        sa.Column("documents", sa.Integer(), nullable=False),
        sa.Column("exact_duplicates_removed", sa.Integer(), nullable=False),
        sa.Column("version_duplicates_removed", sa.Integer(), nullable=False),
        sa.Column("chunks", sa.Integer(), nullable=False),
        sa.Column("embedded_chunks", sa.Integer(), nullable=False),
        sa.Column("stored_chunks", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["source_name"], ["crawler_sources.name"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_crawler_ingest_runs_run_id"),
        "crawler_ingest_runs",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_crawler_ingest_runs_source_name"),
        "crawler_ingest_runs",
        ["source_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_crawler_ingest_runs_source_name"), table_name="crawler_ingest_runs")
    op.drop_index(op.f("ix_crawler_ingest_runs_run_id"), table_name="crawler_ingest_runs")
    op.drop_table("crawler_ingest_runs")
    op.drop_index(op.f("ix_crawler_document_chunks_doc_id"), table_name="crawler_document_chunks")
    op.drop_index(
        op.f("ix_crawler_document_chunks_content_hash"),
        table_name="crawler_document_chunks",
    )
    op.drop_table("crawler_document_chunks")
    op.drop_index(op.f("ix_crawler_attachments_doc_id"), table_name="crawler_attachments")
    op.drop_table("crawler_attachments")
    op.drop_index(op.f("ix_crawler_documents_source_name"), table_name="crawler_documents")
    op.drop_index(op.f("ix_crawler_documents_doc_type"), table_name="crawler_documents")
    op.drop_index(op.f("ix_crawler_documents_department"), table_name="crawler_documents")
    op.drop_index(op.f("ix_crawler_documents_content_hash"), table_name="crawler_documents")
    op.drop_index(op.f("ix_crawler_documents_category"), table_name="crawler_documents")
    op.drop_table("crawler_documents")
    op.drop_table("crawler_sources")
    op.drop_table("kgu_places")
    op.drop_table("kgu_contacts")
