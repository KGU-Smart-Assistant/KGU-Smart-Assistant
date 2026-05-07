# Database Migrations

Alembic owns schema changes for PostgreSQL.

Common commands:

```bash
alembic upgrade head
alembic revision --autogenerate -m "describe change"
alembic downgrade -1
```

`migrations/env.py` reads `DATABASE_URL` from the environment or local `.env`.
It imports only SQLAlchemy model metadata and does not require Gemini or other API keys.
