from app.crawlers import run_once


def test_default_ingest_args_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("CRAWLER_INGEST_ARGS", "--store-db --limit 1")

    assert run_once.default_ingest_args() == ["--store-db", "--limit", "1"]


def test_run_once_skips_when_lock_is_not_acquired(monkeypatch) -> None:
    calls = []

    class FakeLock:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(run_once, "postgres_advisory_lock", lambda lock_name: FakeLock())
    monkeypatch.setattr(run_once.run_ingest, "main", lambda args: calls.append(args))

    assert run_once.run_once(lock_name="lock", ingest_args=["--store-db"]) is False
    assert calls == []


def test_run_once_executes_ingest_when_lock_is_acquired(monkeypatch) -> None:
    calls = []

    class FakeLock:
        def __enter__(self):
            return True

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(run_once, "postgres_advisory_lock", lambda lock_name: FakeLock())
    monkeypatch.setattr(run_once.run_ingest, "main", lambda args: calls.append(args))

    assert run_once.run_once(lock_name="lock", ingest_args=["--store-db"]) is True
    assert calls == [["--store-db"]]


def test_main_uses_explicit_args(monkeypatch) -> None:
    calls = []

    monkeypatch.setenv("CRAWLER_LOCK_NAME", "test-lock")
    monkeypatch.setattr(
        run_once,
        "run_once",
        lambda *, lock_name, ingest_args: calls.append((lock_name, list(ingest_args))) or True,
    )

    run_once.main(["--source", "alpha_notice"])

    assert calls == [("test-lock", ["--source", "alpha_notice"])]


def test_main_does_not_keep_alive_by_default(monkeypatch) -> None:
    calls = []

    monkeypatch.delenv("CRAWLER_KEEP_ALIVE", raising=False)
    monkeypatch.setattr(
        run_once,
        "run_once",
        lambda *, lock_name, ingest_args: calls.append((lock_name, list(ingest_args))) or True,
    )

    run_once.main(["--store-db"])

    assert calls == [("kgu-smart-assistant:crawler-ingest", ["--store-db"])]
