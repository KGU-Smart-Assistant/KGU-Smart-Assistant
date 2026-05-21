from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import text

from app.db.session import SessionLocal

DEFAULT_CRAWLER_LOCK_NAME = "kgu-smart-assistant:crawler-ingest"


@contextmanager
def postgres_advisory_lock(lock_name: str = DEFAULT_CRAWLER_LOCK_NAME) -> Iterator[bool]:
    """Acquire a PostgreSQL session-level advisory lock for crawler jobs."""
    db = SessionLocal()
    acquired = False
    try:
        acquired = bool(
            db.execute(
                text("SELECT pg_try_advisory_lock(hashtext(:lock_name))"),
                {"lock_name": lock_name},
            ).scalar()
        )
        yield acquired
    finally:
        if acquired:
            db.execute(
                text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
                {"lock_name": lock_name},
            )
        db.close()
