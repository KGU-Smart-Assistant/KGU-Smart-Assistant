from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KguPlace(Base):
    __tablename__ = "kgu_places"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)


class KguContact(Base):
    __tablename__ = "kgu_contacts"
    __table_args__ = (UniqueConstraint("name", name="uq_kgu_contacts_name"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class CrawlerSource(Base):
    __tablename__ = "crawler_sources"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    seed_urls_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class CrawlerDocument(Base):
    __tablename__ = "crawler_documents"

    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_name: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("crawler_sources.name"),
        nullable=False,
        index=True,
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    author_department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class CrawlerDocumentChunk(Base):
    __tablename__ = "crawler_document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("crawler_documents.doc_id"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="html")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class CrawlerAttachment(Base):
    __tablename__ = "crawler_attachments"
    __table_args__ = (
        UniqueConstraint("doc_id", "attachment_url", name="uq_crawler_attachment_doc_url"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("crawler_documents.doc_id"),
        nullable=False,
        index=True,
    )
    attachment_url: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str | None] = mapped_column(String(300), nullable=True)
    file_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    extraction_status: Mapped[str] = mapped_column(String(32), nullable=False, default="discovered")
    error_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class CrawlerIngestRun(Base):
    __tablename__ = "crawler_ingest_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("crawler_sources.name"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    status_reason: Mapped[str] = mapped_column(Text, nullable=False)
    raw_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exact_duplicates_removed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version_duplicates_removed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedded_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stored_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

