import json
from datetime import datetime, timedelta
from pathlib import Path

from app.crawlers import run_ingest
from app.schemas import Document, DocumentChunk


def _build_document(
    *,
    doc_id: str,
    source_url: str,
    title: str,
    content: str,
    collected_at: datetime,
) -> Document:
    return Document(
        doc_id=doc_id,
        source_type="html",
        source_url=source_url,
        title=title,
        content=content,
        category="academic",
        department="academic_affairs",
        published_at=None,
        collected_at=collected_at,
    )


def test_run_ingest_executes_end_to_end_pipeline(monkeypatch, capsys) -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    duplicate_content = "duplicate-notice-" * 80
    long_content = "long-notice-" * 120

    source_config = [
        {
            "name": "academic_notices",
            "seed_urls": ["https://example.com/notices"],
            "category": "academic",
            "department": "academic_affairs",
            "max_pages": 5,
            "max_depth": 1,
            "embed": True,
            "embedding_limit": 2,
        }
    ]

    documents = [
        _build_document(
            doc_id="dup-old",
            source_url="https://example.com/notices/1",
            title="Academic Notice",
            content=duplicate_content,
            collected_at=now,
        ),
        _build_document(
            doc_id="dup-new",
            source_url="https://example.com/notices/2",
            title="Academic Notice",
            content=duplicate_content,
            collected_at=now + timedelta(minutes=1),
        ),
        _build_document(
            doc_id="unique",
            source_url="https://example.com/notices/3",
            title="Second Notice",
            content=long_content,
            collected_at=now,
        ),
    ]

    monkeypatch.setattr(run_ingest, "load_sources_config", lambda _: source_config)
    monkeypatch.setattr(
        run_ingest,
        "collect_documents_with_crawl4ai",
        lambda _: documents,
    )
    monkeypatch.setattr(run_ingest, "embed_chunks", lambda chunks: [object() for _ in chunks])
    monkeypatch.setattr(run_ingest, "upsert_embedded_chunks", lambda chunks, **_: len(chunks))
    monkeypatch.setattr(run_ingest, "default_report_output_dir", lambda: Path(".tmp/test-reports"))
    monkeypatch.setattr(
        run_ingest,
        "write_ingest_report",
        lambda report, output_dir: output_dir / "ingest-report-test.json",
    )

    run_ingest.main()

    captured = capsys.readouterr()
    assert "[academic_notices] raw_documents=3 documents=2" in captured.out
    assert "exact_duplicates_removed=1" in captured.out
    assert "version_duplicates_removed=0" in captured.out
    assert "chunks=4 embedded_chunks=2 stored_chunks=0 db_documents=0 db_chunks=0" in captured.out
    assert "status=ok" in captured.out
    assert "total_documents=2" in captured.out
    assert "total_chunks=4" in captured.out
    assert "total_embedded_chunks=2" in captured.out
    assert "total_stored_chunks=0" in captured.out
    assert "total_db_documents=0" in captured.out
    assert "total_db_chunks=0" in captured.out
    normalized_output = captured.out.replace("\\", "/")
    assert "ingest_report=.tmp/test-reports/ingest-report-test.json" in normalized_output


def test_load_sources_config_expands_generated_sources() -> None:
    config_path = Path(".tmp/generated-sources-test.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(
            json.dumps(
                {
                    "sources": [{"name": "explicit_source", "seed_urls": ["https://example.com/a"]}],
                    "department_sites": [
                        {
                            "slug": "example_department",
                            "department": "example_department",
                            "base_url": "https://example.com/index.do",
                            "site_type": "portal",
                        }
                    ],
                    "source_blueprints": [
                        {
                            "name_suffix": "notice",
                            "applies_to": ["portal"],
                            "category": "notice",
                            "seed_url_template": "{base_url}/notice",
                            "follow_patterns": ["selectbbsnttlist.do", "selectbbsnttview.do"],
                            "collect_patterns": ["selectbbsnttview.do"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        sources = run_ingest.load_sources_config(config_path)

        assert [source["name"] for source in sources] == [
            "explicit_source",
            "example_department_notice",
        ]
        assert sources[1]["department"] == "example_department"
        assert sources[1]["seed_urls"] == ["https://example.com/index.do/notice"]
    finally:
        config_path.unlink(missing_ok=True)


def test_load_sources_config_applies_site_source_overrides() -> None:
    config_path = Path(".tmp/generated-sources-override-test.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(
            json.dumps(
                {
                    "department_sites": [
                        {
                            "slug": "example_department",
                            "department": "example_department",
                            "base_url": "https://example.com/index.do",
                            "site_type": "portal",
                            "source_overrides": {
                                "faq": {
                                    "seed_urls": ["https://example.com/faq"],
                                    "collect_seed_pages": True,
                                    "collect_patterns": ["faq"],
                                }
                            },
                        }
                    ],
                    "source_blueprints": [
                        {
                            "name_suffix": "faq",
                            "applies_to": ["portal"],
                            "category": "faq",
                            "seed_url_template": "{base_url}/faq-default",
                            "collect_seed_pages": False,
                            "collect_patterns": ["selectbbsnttview.do"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        sources = run_ingest.load_sources_config(config_path)

        assert [source["name"] for source in sources] == ["example_department_faq"]
        assert sources[0]["seed_urls"] == ["https://example.com/faq"]
        assert sources[0]["collect_seed_pages"] is True
        assert sources[0]["collect_patterns"] == ["faq"]
    finally:
        config_path.unlink(missing_ok=True)


def test_build_crawler_config_derives_allowed_path_prefixes_and_skip_images() -> None:
    config = run_ingest.build_crawler_config(
        {
            "seed_urls": ["https://www.kyonggi.ac.kr/open_major_Seoul/selectBbsNttList.do?key=1"],
            "category": "notice",
            "department": "open_major_seoul",
            "min_published_year": 2018,
            "skip_images": True,
            "allowed_author_department_filters": ["?먯쑀?꾧났"],
        }
    )

    assert config.allowed_path_prefixes == ("/open_major_Seoul/",)
    assert config.min_published_at == datetime(2018, 1, 1)
    assert config.docling_config.skip_images is True
    assert config.allowed_author_department_filters == ("?먯쑀?꾧났",)


def test_classify_source_report_statuses() -> None:
    assert run_ingest.classify_source_report(
        category="notice",
        raw_documents=0,
        documents=0,
        exact_duplicates_removed=0,
        version_duplicates_removed=0,
    )["status"] == "no_content_discovered"

    assert run_ingest.classify_source_report(
        category="faq",
        raw_documents=2,
        documents=0,
        exact_duplicates_removed=0,
        version_duplicates_removed=0,
    )["status"] == "empty_board_or_filtered"

    assert run_ingest.classify_source_report(
        category="materials",
        raw_documents=3,
        documents=0,
        exact_duplicates_removed=1,
        version_duplicates_removed=1,
    )["status"] == "fully_deduplicated"

    assert run_ingest.classify_source_report(
        category="notice",
        raw_documents=3,
        documents=2,
        exact_duplicates_removed=0,
        version_duplicates_removed=0,
    )["status"] == "ok"


def test_select_sources_applies_name_prefix_and_limit_filters() -> None:
    sources = [
        {"name": "alpha_notice"},
        {"name": "alpha_materials"},
        {"name": "beta_notice"},
    ]

    selected = run_ingest.select_sources(
        sources,
        names=["alpha_notice", "beta_notice"],
        prefixes=["alpha_", "beta_"],
        exclude_names=["beta_notice"],
        limit=1,
    )

    assert [source["name"] for source in selected] == ["alpha_notice"]


def test_apply_runtime_overrides_updates_optional_settings() -> None:
    source = {
        "name": "alpha_notice",
        "embed": True,
        "max_pages": 20,
        "max_pagination_pages": 200,
    }

    overridden = run_ingest.apply_runtime_overrides(
        source,
        skip_embed=True,
        max_pages=5,
        max_pagination_pages=2,
    )

    assert overridden["embed"] is False
    assert overridden["max_pages"] == 5
    assert overridden["max_pagination_pages"] == 2
    assert source["embed"] is True


def test_select_embedding_chunks_includes_attachment_quota() -> None:
    chunks = [
        DocumentChunk(
            chunk_id=f"main-{index}",
            doc_id="main-doc",
            chunk_index=index,
            text=f"main chunk {index}",
            title="Main notice",
            source_url="https://example.com/notice",
            source_type="html",
        )
        for index in range(3)
    ]
    chunks.extend(
        [
            DocumentChunk(
                chunk_id=f"attachment-{index}",
                doc_id="attachment-doc",
                chunk_index=index,
                text=f"attachment chunk {index}",
                title="Attachment",
                source_url=f"https://example.com/downloadBbsFile.do?atchmnflNo={index}",
                source_type="pdf",
            )
            for index in range(3)
        ]
    )

    selected = run_ingest.select_embedding_chunks(
        chunks,
        embedding_limit=2,
        attachment_embedding_limit=2,
    )

    assert [chunk.chunk_id for chunk in selected] == [
        "main-0",
        "main-1",
        "attachment-0",
        "attachment-1",
    ]


def test_run_ingest_main_applies_cli_filters_and_runtime_overrides(monkeypatch, capsys) -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    source_config = [
        {
            "name": "alpha_notice",
            "seed_urls": ["https://example.com/notices"],
            "category": "academic",
            "department": "academic_affairs",
            "max_pages": 20,
            "max_pagination_pages": 200,
            "embed": True,
        },
        {
            "name": "beta_notice",
            "seed_urls": ["https://example.com/other"],
            "category": "academic",
            "department": "academic_affairs",
            "max_pages": 20,
            "max_pagination_pages": 200,
            "embed": True,
        },
    ]
    documents = [
        _build_document(
            doc_id="unique",
            source_url="https://example.com/notices/1",
            title="Academic Notice",
            content="long-notice-" * 120,
            collected_at=now,
        )
    ]
    built_configs = []

    monkeypatch.setattr(run_ingest, "load_sources_config", lambda _: source_config)

    def _build_config(source):
        built_configs.append(source)

        class DummyConfig:
            pass

        return DummyConfig()

    monkeypatch.setattr(run_ingest, "build_crawler_config", _build_config)
    monkeypatch.setattr(run_ingest, "collect_documents_with_crawl4ai", lambda _: documents)
    monkeypatch.setattr(
        run_ingest,
        "embed_chunks",
        lambda _: (_ for _ in ()).throw(AssertionError("embed should be skipped")),
    )
    monkeypatch.setattr(run_ingest, "default_report_output_dir", lambda: Path(".tmp/test-reports"))
    monkeypatch.setattr(
        run_ingest,
        "write_ingest_report",
        lambda report, output_dir: output_dir / "ingest-report-test.json",
    )

    run_ingest.main(
        [
            "--source-prefix",
            "alpha_",
            "--limit",
            "1",
            "--skip-embed",
            "--max-pages",
            "5",
            "--max-pagination-pages",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert len(built_configs) == 1
    assert built_configs[0]["name"] == "alpha_notice"
    assert built_configs[0]["embed"] is False
    assert built_configs[0]["max_pages"] == 5
    assert built_configs[0]["max_pagination_pages"] == 2
    assert "source_count=1" not in captured.out
    assert "[alpha_notice] raw_documents=1 documents=1" in captured.out


def test_run_ingest_main_reports_no_matching_sources(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_ingest,
        "load_sources_config",
        lambda _: [{"name": "alpha_notice", "seed_urls": ["https://example.com/notices"]}],
    )

    run_ingest.main(["--source", "missing_source"])

    captured = capsys.readouterr()
    assert "No sources matched the requested filters." in captured.out


def test_run_ingest_main_stores_embedded_chunks_when_requested(monkeypatch, capsys) -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    source_config = [
        {
            "name": "alpha_notice",
            "seed_urls": ["https://example.com/notices"],
            "category": "academic",
            "department": "academic_affairs",
            "max_pages": 20,
            "max_pagination_pages": 200,
            "embed": True,
        }
    ]
    documents = [
        _build_document(
            doc_id="unique",
            source_url="https://example.com/notices/1",
            title="Academic Notice",
            content="long-notice-" * 120,
            collected_at=now,
        )
    ]

    monkeypatch.setattr(run_ingest, "load_sources_config", lambda _: source_config)
    monkeypatch.setattr(run_ingest, "collect_documents_with_crawl4ai", lambda _: documents)
    monkeypatch.setattr(run_ingest, "embed_chunks", lambda chunks: [object() for _ in chunks])
    deleted_documents = []
    monkeypatch.setattr(
        run_ingest,
        "delete_embedded_chunks_for_documents",
        lambda docs: deleted_documents.extend(docs),
    )
    monkeypatch.setattr(run_ingest, "upsert_embedded_chunks", lambda chunks, **_: len(chunks))
    monkeypatch.setattr(run_ingest, "default_report_output_dir", lambda: Path(".tmp/test-reports"))
    monkeypatch.setattr(
        run_ingest,
        "write_ingest_report",
        lambda report, output_dir: output_dir / "ingest-report-test.json",
    )

    run_ingest.main(["--source", "alpha_notice", "--store-vectors"])

    captured = capsys.readouterr()
    assert deleted_documents == documents
    assert "stored_chunks=2" in captured.out
    assert "total_stored_chunks=2" in captured.out


def test_run_ingest_main_stores_documents_in_db_when_requested(monkeypatch, capsys) -> None:
    now = datetime(2026, 4, 10, 12, 0, 0)
    source_config = [
        {
            "name": "alpha_notice",
            "seed_urls": ["https://example.com/notices"],
            "category": "notice",
            "department": "academic_affairs",
            "max_pages": 20,
            "max_pagination_pages": 200,
            "embed": False,
        }
    ]
    documents = [
        _build_document(
            doc_id="unique",
            source_url="https://example.com/notices/1",
            title="Academic Notice",
            content="long-notice-" * 120,
            collected_at=now,
        )
    ]
    captured_store = {}

    monkeypatch.setattr(run_ingest, "load_sources_config", lambda _: source_config)
    monkeypatch.setattr(run_ingest, "collect_documents_with_crawl4ai", lambda _: documents)
    monkeypatch.setattr(run_ingest, "default_report_output_dir", lambda: Path(".tmp/test-reports"))
    monkeypatch.setattr(
        run_ingest,
        "write_ingest_report",
        lambda report, output_dir: output_dir / "ingest-report-test.json",
    )

    def _store_result(**kwargs):
        captured_store.update(kwargs)
        return {"documents": len(kwargs["documents"]), "chunks": len(kwargs["chunks"])}

    monkeypatch.setattr(run_ingest, "store_ingest_source_result", _store_result)

    run_ingest.main(["--source", "alpha_notice", "--store-db"])

    captured = capsys.readouterr()
    assert captured_store["source"]["name"] == "alpha_notice"
    assert captured_store["source_report"]["status"] == "ok"
    assert captured_store["source_report"]["db_documents"] == 1
    assert captured_store["source_report"]["db_chunks"] == 2
    assert captured_store["run_id"]
    assert "db_documents=1 db_chunks=2" in captured.out
    assert "total_db_documents=1" in captured.out
    assert "total_db_chunks=2" in captured.out


def test_run_ingest_main_rejects_store_vectors_with_skip_embed() -> None:
    try:
        run_ingest.main(["--skip-embed", "--store-vectors"])
    except ValueError as exc:
        assert str(exc) == "--store-vectors cannot be used together with --skip-embed."
    else:
        raise AssertionError("Expected ValueError for incompatible CLI options.")
