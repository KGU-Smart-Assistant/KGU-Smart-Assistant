from types import SimpleNamespace

from app.crawlers import embedding_pipeline


def test_embed_text_retries_retryable_errors(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeModels:
        def embed_content(self, *, model, contents):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("429 rate limit")
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[0.1, 0.2, 0.3]),
                ]
            )

    monkeypatch.setattr(
        embedding_pipeline,
        "_create_client",
        lambda: SimpleNamespace(models=FakeModels()),
    )
    monkeypatch.setattr(embedding_pipeline.time, "sleep", lambda seconds: None)

    assert embedding_pipeline.embed_text("hello", max_retries=1) == [0.1, 0.2, 0.3]
    assert calls["count"] == 2


def test_embed_text_does_not_retry_non_retryable_errors(monkeypatch) -> None:
    class FakeModels:
        def embed_content(self, *, model, contents):
            raise RuntimeError("invalid request")

    monkeypatch.setattr(
        embedding_pipeline,
        "_create_client",
        lambda: SimpleNamespace(models=FakeModels()),
    )

    try:
        embedding_pipeline.embed_text("hello", max_retries=3)
    except RuntimeError as exc:
        assert str(exc) == "invalid request"
    else:
        raise AssertionError("Expected non-retryable embedding error.")
