from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.crawlers.run_pipeline import (
    DEFAULT_CRAWLED_MARKDOWN_ROOT,
    DEFAULT_PREPARED_MARKDOWN_ROOT,
    CrawlPreparePipelineOptions,
    _resolve_output_dirs,
    run_crawl_prepare_pipeline,
)


def test_run_crawl_prepare_pipeline_generates_chunks(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
        sources:
          - name: university_notices
            urls:
              - https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=7520&nttNo=3
            domain: general_notice
            department: university
            collect_seed_pages: true
            collect_patterns:
              - selectbbsnttview.do
            recommended_action: KEEP
        """,
        encoding="utf-8",
    )

    result = run_crawl_prepare_pipeline(
        CrawlPreparePipelineOptions(
            config_path=config,
            crawl_output_dir=tmp_path / "crawl",
            prepared_output_dir=tmp_path / "prepared",
            force=True,
        ),
        crawler=FakeCrawl4AI(
            {
                "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=7520&nttNo=3": (
                    "# Sample Notice\n\n"
                    "작성자: 학사혁신팀\n\n"
                    "## 신청 방법\n\n"
                    "학생은 KUTIS에서 신청서를 제출해야 합니다. "
                    "신청 기간은 2026.06.01부터 2026.06.30까지입니다. "
                    "문의는 sample@kgu.ac.kr 또는 031-249-9000으로 연락합니다.\n"
                )
            }
        ),
    )

    assert result["crawl"]["written"] == 1
    assert result["prepare"]["published_chunks"] >= 1
    assert (tmp_path / "prepared" / "chunks" / "chunks.jsonl").exists()
    assert (tmp_path / "prepared" / "manifest" / "pipeline_report.json").exists()


def test_run_crawl_prepare_pipeline_defaults_outputs_under_backend_data(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    options = CrawlPreparePipelineOptions(
        config_path=config,
        source_names=("finance_notices",),
        run_id="finance notices 2026/06/06",
    )

    crawl_output_dir, prepared_output_dir = _resolve_output_dirs(
        options,
        started_at=datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert crawl_output_dir == DEFAULT_CRAWLED_MARKDOWN_ROOT / "finance-notices-2026-06-06"
    assert prepared_output_dir == DEFAULT_PREPARED_MARKDOWN_ROOT / "finance-notices-2026-06-06"


def test_run_crawl_prepare_pipeline_optionally_stores_prepared_rows(tmp_path: Path) -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.orm import sessionmaker

    from app.models import Base, CrawlerAttachment, CrawlerDocument, CrawlerDocumentChunk

    config = tmp_path / "sources.yaml"
    config.write_text(
        """
        sources:
          - name: finance_notices
            urls:
              - https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=8004&nttNo=497328
            domain: tuition
            department: finance_accounting
            collect_seed_pages: true
            collect_patterns:
              - selectbbsnttview.do
            recommended_action: KEEP
        """,
        encoding="utf-8",
    )
    db = _session(sqlalchemy.create_engine, sqlalchemy.event, sessionmaker, Base)

    result = run_crawl_prepare_pipeline(
        CrawlPreparePipelineOptions(
            config_path=config,
            crawl_output_dir=tmp_path / "crawl",
            prepared_output_dir=tmp_path / "prepared",
            force=True,
            store_db=True,
            run_id="test-run",
        ),
        crawler=FakeCrawl4AI(
            {
                "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=8004&nttNo=497328": (
                    "# 연말정산 안내\n\n"
                    "신청 기간은 2026.01.01부터 2026.01.31까지입니다. "
                    "KUTIS에서 자료를 등록하고 재무회계팀에 제출합니다. "
                    "문의는 031-249-9000으로 연락합니다.\n\n"
                    "[서식 다운로드](https://www.kyonggi.ac.kr/www/downloadBbsFile.do?atchmnflNo=981574)\n"
                )
            }
        ),
        db_session=db,
    )

    assert result["db_store"]["documents"] == 1
    assert result["db_store"]["chunks"] >= 1
    assert db.query(CrawlerDocument).count() == 1
    assert db.query(CrawlerDocumentChunk).count() >= 1
    chunk = db.query(CrawlerDocumentChunk).one()
    assert chunk.chunk_id
    assert chunk.vector_point_id
    assert chunk.chunk_id != chunk.vector_point_id
    assert chunk.content
    assert chunk.embedding_text == chunk.text
    assert chunk.chunk_text_hash
    assert chunk.section_title
    assert chunk.section_path
    assert chunk.section_kind
    assert chunk.embedding_eligibility == "ELIGIBLE"
    assert chunk.chunk_quality_status == "PASSED"
    assert chunk.document_quality_status == "PASSED"
    assert chunk.content_token_count is not None
    assert chunk.embedding_token_count is not None
    assert chunk.domain == "tuition"
    assert chunk.department == "finance_accounting"
    assert chunk.source_name == "finance_notices"
    assert chunk.prepared_body_hash
    assert chunk.prepared_artifact_hash
    assert chunk.metadata_json["page_type"] == "LIST_PAGE"
    attachment = db.query(CrawlerAttachment).one()
    assert attachment.doc_id
    assert attachment.attachment_url.endswith("atchmnflNo=981574")
    assert attachment.extraction_status == "discovered"


def _session(create_engine, event, sessionmaker, base):
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


class FakeCrawl4AI:
    def __init__(self, markdown_by_url: dict[str, str]) -> None:
        self.markdown_by_url = markdown_by_url

    async def arun(self, url: str, config=None):
        normalized_url = url.split("?utm_source=", 1)[0]
        markdown = self.markdown_by_url[normalized_url]
        title = markdown.splitlines()[0].lstrip("# ").strip()
        return SimpleNamespace(
            success=True,
            url=url,
            markdown=markdown,
            metadata={"title": title},
            links={"internal": []},
            error_message="",
        )
