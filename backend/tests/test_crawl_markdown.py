from pathlib import Path
from types import SimpleNamespace

from app.crawlers.crawl_markdown import (
    CrawlOptions,
    CrawlSource,
    DEFAULT_CRAWLED_MARKDOWN_ROOT,
    _default_output_dir,
    crawl_markdown,
    load_sources_config,
)


def test_default_crawl_output_dir_uses_backend_data() -> None:
    assert _default_output_dir("finance notices 2026/06/06") == (
        DEFAULT_CRAWLED_MARKDOWN_ROOT / "finance-notices-2026-06-06"
    )


def test_crawl_markdown_writes_prepare_markdown_input_and_seed_csv(tmp_path: Path) -> None:
    result_by_url = {
        "https://www.kyonggi.ac.kr/notice": """
        # Scholarship Notice

        Apply by June 30.

        - Bring student ID.
        """,
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="university_notices",
                    urls=["https://www.kyonggi.ac.kr/notice?utm_source=test"],
                    domain="university",
                    department="student",
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url),
    )

    assert result["fetched"] == 1
    assert result["written"] == 1

    seed_csv = tmp_path / "crawl" / "manifest" / "crawl_classification_seed.csv"
    seed = seed_csv.read_text(encoding="utf-8")
    assert "recommended_action,doc_id,source_name,source_url,final_md_path" in seed
    assert "MANUAL_REVIEW" in seed
    assert "university_notices" in seed

    markdown_path = next((tmp_path / "crawl" / "final" / "university_notices").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "source_name: university_notices" in markdown
    assert "source_url: https://www.kyonggi.ac.kr/notice?utm_source=test" in markdown
    assert "# Scholarship Notice" in markdown
    assert "Apply by June 30." in markdown


def test_crawl_markdown_single_page_writes_seed_only_and_does_not_follow_links(tmp_path: Path) -> None:
    seed_url = "https://www.kyonggi.ac.kr/www/contents.do?key=8418"
    linked_url = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    result_by_url = {
        seed_url: """
        # Graduation Requirements

        Complete required credits.
        """,
    }
    links_by_url = {
        seed_url: {
            "internal": [
                {"href": linked_url},
                {"href": "https://www.kyonggi.ac.kr/www/contents.do?key=8418&pageIndex=2"},
            ]
        }
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="graduation_requirements",
                    urls=[seed_url],
                    page_type="SINGLE_PAGE",
                    max_depth=2,
                    collect_seed_pages=True,
                    follow_patterns=("contents.do", "selectbbsnttview.do"),
                    collect_patterns=("contents.do",),
                    recommended_action="KEEP",
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["fetched"] == 1
    assert result["written"] == 1
    assert result["single_pages_fetched"] == 1
    assert result["list_pages_fetched"] == 0
    assert result["detail_pages_fetched"] == 0
    assert result["documents"][0]["source_url"] == seed_url

    seed_csv = (tmp_path / "crawl" / "manifest" / "crawl_classification_seed.csv").read_text(encoding="utf-8")
    assert "KEEP" in seed_csv
    markdown_path = next((tmp_path / "crawl" / "final" / "graduation_requirements").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "page_type: SINGLE_PAGE" in markdown
    assert "# Graduation Requirements" in markdown
    assert linked_url not in markdown


def test_crawl_markdown_single_page_extracts_contents_body_only(tmp_path: Path) -> None:
    seed_url = "https://www.kyonggi.ac.kr/www/contents.do?key=8418"
    result_by_url = {
        seed_url: """
        # Graduation Requirements

        University Office
        Share
        Academic Affairs

        ### [Academic Affairs: Graduation Requirements]

        #### 1. Graduation Guide

        | Requirement | Detail |
        | --- | --- |
        | Credits | Complete 130 credits. |

        - [HOME](https://www.kyonggi.ac.kr/www/index.do)
        Custom settings
        """,
    }
    html_by_url = {
        seed_url: """
        <html>
          <body>
            <div class="breadcrumb">Home / University Office / Academic Affairs</div>
            <div class="share sns">Share Facebook Print</div>
            <div class="page-layout">
              <div class="left-menu">
                <a href="/www/contents.do?key=8411">Academic records</a>
                <a href="/www/contents.do?key=8418">Graduation Requirements</a>
              </div>
              <main class="contents">
                <h3>[Academic Affairs: Graduation Requirements]</h3>
                <h4>1. Graduation Guide</h4>
                <table>
                  <tr><th>Requirement</th><th>Detail</th></tr>
                  <tr><td>Credits</td><td>Complete 130 credits.</td></tr>
                </table>
              </main>
            </div>
            <div class="custom popup">Custom settings</div>
          </body>
        </html>
        """,
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="graduation_requirements",
                    urls=[seed_url],
                    page_type="SINGLE_PAGE",
                    collect_patterns=("contents.do",),
                    recommended_action="KEEP",
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, html_by_url=html_by_url),
    )

    assert result["written"] == 1
    markdown_path = next((tmp_path / "crawl" / "final" / "graduation_requirements").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    body = markdown.split("---", 2)[2]
    assert "# [Academic Affairs: Graduation Requirements]" in body
    assert "Complete 130 credits." in body
    assert "Home / University Office" not in body
    assert "Share Facebook Print" not in body
    assert "Academic records" not in body
    assert "Custom settings" not in body
    assert "[HOME]" not in body


def test_crawl_markdown_follows_same_host_links_and_blocks_script_links(tmp_path: Path) -> None:
    result_by_url = {
        "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073": "# List",
        "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1": """
        # Site Header
        주메뉴 열기

        Detail title
        _작성자_ admin
        _작성일_ 2026년 06월 04일
        Useful body.
        [목록](https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073)
        _이전글_
        """,
    }
    links_by_url = {
        "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073": {
            "internal": [
                {"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"},
                {"href": "javascript:downloadBbsFile()"},
            ],
            "external": [{"href": "https://external.example/detail"}],
        }
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=["https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073"],
                    max_depth=1,
                    collect_seed_pages=False,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["fetched"] == 2
    assert result["written"] == 1
    assert result["queued"] == 2
    assert result["documents"][0]["source_url"] == "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    assert not any("javascript" in row["source_url"] for row in result["documents"])
    assert not any("external.example" in row["source_url"] for row in result["documents"])

    markdown_path = next((tmp_path / "crawl" / "final" / "notice").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Detail title" in markdown
    assert "Useful body." in markdown
    assert "주메뉴 열기" not in markdown
    assert "[목록]" not in markdown
    assert "# List" not in markdown


def test_crawl_markdown_extracts_main_body_from_dom_before_markdown_cleanup(tmp_path: Path) -> None:
    url = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    result_by_url = {
        url: """
        # Full Crawl Markdown
        주메뉴 열기
        Real notice title
        1. Real body text.
        2. Required document.
        Footer text
        """,
    }
    html_by_url = {
        url: """
        <html>
          <body>
            <div class="site-menu">주메뉴 열기 <a href="/login">login</a></div>
            <div class="page-shell">
              <div class="breadcrumb">대학본부 / 교육혁신처 / 학사혁신팀 / 공지사항</div>
              <div class="board-area">
                <div class="board-title">Real notice title</div>
                <div class="board-meta">
                  <span>작성자</span>
                  <span>학사혁신팀</span>
                  <span>작성일</span>
                  <span>2026년 04월 24일 16시 18분 05초</span>
                </div>
                    <div class="board-body">
                      <p>1. Real body text.</p>
                      <ul><li>2. Required document.</li></ul>
                      <p><a href="/www/downloadBbsFile.do?atchmnflNo=1">Attachment</a></p>
                    </div>
                <div class="board-nav">[목록] 이전글 다음글</div>
              </div>
            </div>
            <footer>Footer text</footer>
          </body>
        </html>
        """,
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[url],
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, html_by_url=html_by_url),
    )

    assert result["written"] == 1
    markdown_path = next((tmp_path / "crawl" / "final" / "notice").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "attachment_urls:" in markdown
    assert "downloadBbsFile.do?atchmnflNo=1" in markdown
    assert "1. Real body text." in markdown
    assert "2. Required document." in markdown
    assert "주메뉴 열기" not in markdown
    assert "대학본부" not in markdown
    assert "board-nav" not in markdown
    assert "[목록]" not in markdown
    assert "이전글" not in markdown
    assert "다음글" not in markdown
    assert "Footer text" not in markdown


def test_crawl_markdown_extracts_attachment_urls_from_html_attributes_and_scripts(tmp_path: Path) -> None:
    url = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"
    result_by_url = {
        url: """
        # Detail

        Body only.
        """,
    }
    html_by_url = {
        url: """
        <html>
          <body>
            <div class="board-title">Detail</div>
            <div class="board-body">
              <p>Body only.</p>
              <button onclick="location.href='./downloadBbsFile.do?atchmnflNo=123&amp;foo=bar'">Download</button>
              <span data-url="/www/downloadBbsFile.do?atchmnflNo=456">Download</span>
              <script>var file = "/www/downloadBbsFile.do?atchmnflNo=789";</script>
            </div>
          </body>
        </html>
        """,
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[url],
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, html_by_url=html_by_url),
    )

    assert result["written"] == 1
    markdown_path = next((tmp_path / "crawl" / "final" / "notice").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "downloadBbsFile.do?atchmnflNo=123&foo=bar" in markdown
    assert "downloadBbsFile.do?atchmnflNo=456" in markdown
    assert "downloadBbsFile.do?atchmnflNo=789" in markdown
    assert "atchmnflno=" not in markdown


def test_crawl_markdown_trims_detail_metadata_and_keeps_body_block(tmp_path: Path) -> None:
    url = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=615831"
    result_by_url = {
        url: """
        # [입학에서 취업까지] 2025학년도 1학기 등록금 분할납부 안내
        작성자 재무회계팀
        작성일 2025년 02월 03일 11시 08분 25초
        1. 분할납부 신청 대상
        2. 분할납부 신청기간
        ※ 신청은 반드시 기간 내에 진행해야 합니다.
        3. 납부방법 및 등록금 납부 일정
        6. 유의사항
        [[도시·교통] 2021-2학기 BARUN 또래강사 모집 안내](https://www.kyonggi.ac.kr/www/selectBbsNttView.do?key=8004&bbsNo=1073&pageIndex=1&pageUnit=10&searchCnd=all&nttNo=491911)
        2021-08-31
        [목록]
        """,
    }
    html_by_url = {
        url: """
        <html>
          <body>
            <div class="site-menu">대학본부 / 학생지원 / 등록금</div>
            <div class="page-shell">
              <div class="breadcrumb">대학본부 / 학생지원 / 재무회계팀 / 공지사항</div>
              <div class="board-title">[입학에서 취업까지] 2025학년도 1학기 등록금 분할납부 안내</div>
              <div class="board-meta">
                <span>작성자</span><span>재무회계팀</span>
                <span>작성일</span><span>2025년 02월 03일 11시 08분 25초</span>
              </div>
              <div class="board-body">
                <p>1. 분할납부 신청 대상</p>
                <p>2. 분할납부 신청기간</p>
                <p>※ 신청은 반드시 기간 내에 진행해야 합니다.</p>
                <p>3. 납부방법 및 등록금 납부 일정</p>
                <p>6. 유의사항</p>
                <p><a href="/www/selectBbsNttView.do?key=8004&amp;bbsNo=1073&amp;pageIndex=1&amp;pageUnit=10&amp;searchCnd=all&amp;nttNo=491911">[도시·교통] 2021-2학기 BARUN 또래강사 모집 안내</a></p>
                <p>2021-08-31</p>
                <p>[목록]</p>
              </div>
              <div class="board-nav">이전글 다음글</div>
            </div>
          </body>
        </html>
        """,
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[url],
                    max_depth=1,
                    max_pagination_pages=1,
                    collect_seed_pages=True,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, html_by_url=html_by_url),
    )

    assert result["written"] == 1
    markdown_path = next((tmp_path / "crawl" / "final" / "notice").glob("*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    body = markdown.split("---", 2)[2]
    assert "# [입학에서 취업까지] 2025학년도 1학기 등록금 분할납부 안내" in body
    assert "분할납부 신청 대상" in markdown
    assert "분할납부 신청기간" in markdown
    assert "납부방법 및 등록금 납부 일정" in markdown
    assert "유의사항" in markdown
    assert "1. 분할납부 신청 대상" in markdown
    assert "작성자" not in body
    assert "재무회계팀" not in body
    assert "작성일" not in body
    assert "대학본부" not in body
    assert "이전글" not in body
    assert "다음글" not in body
    assert "[목록]" not in body
    assert "selectBbsNttView.do" not in body
    assert "2021-08-31" not in body


def test_crawl_markdown_generates_pagination_and_writes_detail_pages_only(tmp_path: Path) -> None:
    page_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073"
    page_2 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&pageIndex=2"
    detail_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    detail_2 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"
    result_by_url = {
        page_1: "# List page 1\n\n페이지 : _1_ / 3",
        page_2: "# List page 2\n\n페이지 : _2_ / 3",
        detail_1: "# Detail 1\n\nUseful body 1.",
        detail_2: "# Detail 2\n\nUseful body 2.",
    }
    links_by_url = {
        page_1: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"}]},
        page_2: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"}]},
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[page_1],
                    max_depth=2,
                    max_pagination_pages=2,
                    collect_seed_pages=False,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["fetched"] == 4
    assert result["written"] == 2
    assert {row["source_url"] for row in result["documents"]} == {detail_1, detail_2}

    markdown = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "crawl" / "final" / "notice").glob("*.md"))
    assert "# Detail 1" in markdown
    assert "# Detail 2" in markdown
    assert "# List page" not in markdown


def test_crawl_markdown_does_not_follow_detail_page_links(tmp_path: Path) -> None:
    page_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073"
    detail_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    detail_3 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=3"
    result_by_url = {
        page_1: "# List page 1\n\n?섏씠吏 : _1_ / 1",
        detail_1: "# Detail 1\n\nUseful body 1.",
        detail_3: "# Detail 3\n\nShould not be fetched.",
    }
    links_by_url = {
        page_1: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"}]},
        detail_1: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=3"}]},
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[page_1],
                    max_depth=2,
                    max_pagination_pages=2,
                    collect_seed_pages=False,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["fetched"] == 2
    assert result["written"] == 1
    assert {row["source_url"] for row in result["documents"]} == {detail_1}


def test_crawl_markdown_does_not_follow_secondary_list_urls_from_list_pages(tmp_path: Path) -> None:
    page_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&key=5258"
    stray_list = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&key=5258&sf.bf1=K0505&sf.dc=11R11&sf.pnos=888"
    detail_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    result_by_url = {
        page_1: "# List page 1\n\n?섏씠吏 : _1_ / 1",
        stray_list: "# Stray list\n\n?섏씠吏 : _1_ / 1",
        detail_1: "# Detail 1\n\nUseful body 1.",
    }
    links_by_url = {
        page_1: {
            "internal": [
                {"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"},
                {"href": "/www/selectBbsNttList.do?bbsNo=1073&key=5258&sf.bf1=K0505&sf.dc=11R11&sf.pnos=888"},
            ]
        },
        stray_list: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=3"}]},
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[page_1],
                    max_depth=1,
                    collect_seed_pages=False,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["list_pages_fetched"] == 1
    assert result["detail_pages_fetched"] == 1
    assert result["fetched"] == 2
    assert result["written"] == 1
    assert {row["source_url"] for row in result["documents"]} == {detail_1}


def test_crawl_markdown_canonicalizes_detail_urls_before_fetch(tmp_path: Path) -> None:
    page_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&dc=11J30&key=8004"
    detail = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&key=8004&nttNo=615831"
    noisy_detail = (
        "https://www.kyonggi.ac.kr/www/selectBbsNttView.do;jsessionid=ABC"
        "?bbsNo=1073&key=8004&nttNo=615831&pageUnit=10&searchCnd=all"
        "&sf.bf1=K0505&sf.dc=11J30&sf.pnos=1073&sf.pnos=888"
    )
    result_by_url = {
        page_1: "# List page 1\n\n페이지 : _1_ / 1",
        detail: "# Detail\n\n1. Useful finance body.",
    }
    links_by_url = {
        page_1: {"internal": [{"href": noisy_detail}]},
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="finance_notices",
                    urls=[page_1],
                    max_depth=2,
                    max_pagination_pages=1,
                    collect_seed_pages=False,
                    follow_patterns=("selectbbsnttlist.do", "selectbbsnttview.do"),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["fetched"] == 2
    assert result["written"] == 1
    assert result["documents"][0]["source_url"] == detail


def test_crawl_markdown_max_pages_limits_list_pages_not_detail_pages(tmp_path: Path) -> None:
    page_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073"
    page_2 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&pageIndex=2"
    detail_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    detail_2 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"
    result_by_url = {
        page_1: "# List page 1\n\n페이지 : _1_ / 2",
        page_2: "# List page 2\n\n페이지 : _2_ / 2",
        detail_1: "# Detail 1\n\nUseful body 1.",
        detail_2: "# Detail 2\n\nUseful body 2.",
    }
    links_by_url = {
        page_1: {
            "internal": [
                {"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"},
                {"href": "/www/selectBbsNttList.do?bbsNo=1073&pageIndex=2"},
            ]
        },
        page_2: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"}]},
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[page_1],
                    max_depth=2,
                    collect_seed_pages=False,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
            max_pages=1,
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url),
    )

    assert result["list_pages_fetched"] == 1
    assert result["detail_pages_fetched"] == 1
    assert result["fetched"] == 2
    assert result["written"] == 1
    assert {row["source_url"] for row in result["documents"]} == {detail_1}


def test_crawl_markdown_uses_dom_pagination_fallback_without_pageindex_or_total_label(tmp_path: Path) -> None:
    page_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073"
    page_2 = "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&pageIndex=2"
    detail_1 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"
    detail_2 = "https://www.kyonggi.ac.kr/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"
    result_by_url = {
        page_1: "# List page 1\n\nNo explicit pagination label.",
        page_2: "# List page 2\n\nNo explicit pagination label.",
        detail_1: "# Detail 1\n\nUseful body 1.",
        detail_2: "# Detail 2\n\nUseful body 2.",
    }
    links_by_url = {
        page_1: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=1"}]},
        page_2: {"internal": [{"href": "/www/selectBbsNttView.do?bbsNo=1073&nttNo=2"}]},
    }
    html_by_url = {
        page_1: """
        <html>
          <body>
            <div class="board-list">
              <nav class="pagination">
                <a href="/www/selectBbsNttList.do?bbsNo=1073">다음</a>
              </nav>
            </div>
          </body>
        </html>
        """,
        page_2: """
        <html>
          <body>
            <div class="board-list">
              <nav class="pagination">
                <span>끝</span>
              </nav>
            </div>
          </body>
        </html>
        """,
    }

    result = crawl_markdown(
        CrawlOptions(
            sources=[
                CrawlSource(
                    name="notice",
                    urls=[page_1],
                    max_depth=2,
                    collect_seed_pages=False,
                    follow_patterns=("bbsno=1073",),
                    collect_patterns=("selectbbsnttview.do",),
                )
            ],
            output_dir=tmp_path / "crawl",
        ),
        crawler=FakeCrawl4AI(result_by_url, links_by_url, html_by_url),
    )

    assert result["list_pages_fetched"] == 2
    assert result["detail_pages_fetched"] == 2
    assert result["fetched"] == 4
    assert result["written"] == 2
    assert {row["source_url"] for row in result["documents"]} == {detail_1, detail_2}


def test_load_sources_config_accepts_existing_seed_urls_format(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
        sources:
          - name: university_notices
            seed_urls:
              - https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073
            domain: general_notice
            department: university
            max_depth: 2
            max_pages: 0
            max_pagination_pages: 200
            restrict_to_seed_board_queries: true
            collect_seed_pages: false
            follow_patterns:
              - "bbsno=1073"
            collect_patterns:
              - "selectBbsNttView.do"
            page_type: LIST_PAGE
        """,
        encoding="utf-8",
    )

    sources = load_sources_config(config)

    assert sources == [
        CrawlSource(
            name="university_notices",
            urls=["https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073"],
            domain="general_notice",
            department="university",
            max_depth=2,
            max_pages=0,
            max_pagination_pages=200,
            collect_seed_pages=False,
            follow_patterns=("bbsno=1073",),
            collect_patterns=("selectbbsnttview.do",),
            allowed_path_prefixes=("/www/",),
            page_type="LIST_PAGE",
        )
    ]


def test_load_sources_config_accepts_single_page_type(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
        sources:
          - name: graduation_requirements
            seed_urls:
              - https://www.kyonggi.ac.kr/www/contents.do?key=8418
            domain: graduation
            department: university
            collect_seed_pages: true
            collect_patterns:
              - contents.do
            page_type: SINGLE_PAGE
            recommended_action: KEEP
        """,
        encoding="utf-8",
    )

    sources = load_sources_config(config)

    assert sources[0].page_type == "SINGLE_PAGE"
    assert sources[0].recommended_action == "KEEP"


def test_load_sources_config_expands_department_blueprints(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
        sources: []
        department_sites:
          - slug: computer
            department: computer_engineering
            site_type: portal
            base_url: https://example.edu/computer/index.do
            source_overrides:
              notice:
                seed_urls:
                  - https://example.edu/computer/selectBbsNttList.do?bbsNo=1073
        source_blueprints:
          - name_suffix: notice
            applies_to:
              - portal
            domain: general_notice
            seed_url_template: "{base_url}"
            max_depth: 2
            collect_seed_pages: false
        """,
        encoding="utf-8",
    )

    sources = load_sources_config(config)

    assert sources == [
        CrawlSource(
            name="computer_notice",
            urls=["https://example.edu/computer/selectBbsNttList.do?bbsNo=1073"],
            domain="department_notice",
            department="computer_engineering",
            max_depth=2,
            collect_seed_pages=False,
            allowed_path_prefixes=("/computer/",),
        )
    ]


class FakeCrawl4AI:
    def __init__(
        self,
        markdown_by_url: dict[str, str],
        links_by_url: dict[str, dict] | None = None,
        html_by_url: dict[str, str] | None = None,
    ) -> None:
        self.markdown_by_url = markdown_by_url
        self.links_by_url = links_by_url or {}
        self.html_by_url = html_by_url or {}

    async def arun(self, url: str, config=None):
        normalized_url = url.split("?utm_source=", 1)[0]
        markdown = self.markdown_by_url[normalized_url]
        title = markdown.splitlines()[0].lstrip("# ").strip()
        return SimpleNamespace(
            success=True,
            url=url,
            markdown=markdown,
            cleaned_html=self.html_by_url.get(normalized_url, ""),
            metadata={"title": title},
            links=self.links_by_url.get(normalized_url, {}),
            error_message="",
        )
