from app.crawlers.parsing.content_cleaner import clean_crawled_markdown


def test_clean_crawled_markdown_trims_ee_site_navigation() -> None:
    content = (
        "# [![](https://ee.kgu.ac.kr/images/common/logo_1.png)](https://ee.kgu.ac.kr/) "
        "[![](https://ee.kgu.ac.kr/images/main/login.png)](https://ee.kgu.ac.kr/index.php?hCode=LOGIN) "
        "* [학과소개](https://ee.kgu.ac.kr/index.php?hCode=INTRO_01_01) "
        "### 학사/기타공지 "
        "| [학사공지] 인공지능 성적 미달 안내 | "
        "공지 본문입니다. "
        "Copyright (C) 2020 Kyonggi University. School of Electronic Engineering. "
        "All Rights Reserved."
    )

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://ee.kgu.ac.kr/index.php?hCode=BOARD&idx=1293",
    )

    assert "login.png" not in cleaned
    assert "학과소개" not in cleaned
    assert "Copyright" not in cleaned
    assert "학사/기타공지" in cleaned
    assert "공지 본문입니다" in cleaned


def test_clean_crawled_markdown_trims_kyonggi_portal_menu() -> None:
    content = """
[본문 바로가기](https://www.kyonggi.ac.kr/u_test/selectBbsNttView.do#contents)
* [경기대학교](https://www.kyonggi.ac.kr/www/index.do)
* [KUTIS](https://kutis.kyonggi.ac.kr/webkutis/view/indexWeb.jsp)
* [LMS](https://lms.kyonggi.ac.kr/login.php)
* [로그인](https://www.kyonggi.ac.kr/loginView.do)
## 주메뉴
* [학과소개](https://www.kyonggi.ac.kr/u_test/contents.do?key=1)
* [커뮤니티](https://www.kyonggi.ac.kr/u_test/sub.do?key=2)
## 공지사항
* 공유하기
SNS공유
[입학에서 취업까지] 기간제 근로자 채용 홍보 요청
_작성자 대학공학과_ 작성일 2026년 01월 09일 실제 공지 본문입니다.
[목록](https://www.kyonggi.ac.kr/u_test/selectBbsNttList.do)
게시물삭제 사유
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://www.kyonggi.ac.kr/u_test/selectBbsNttView.do?bbsNo=1073",
    )

    assert "본문 바로가기" not in cleaned
    assert "KUTIS" not in cleaned
    assert "LMS" not in cleaned
    assert "로그인" not in cleaned
    assert "주메뉴" not in cleaned
    assert "공유하기" not in cleaned
    assert "목록" not in cleaned
    assert "게시물삭제" not in cleaned
    assert "공지사항" in cleaned
    assert "실제 공지 본문입니다" in cleaned


def test_clean_crawled_markdown_removes_empty_and_escaped_markdown_links() -> None:
    content = r"""
## 공지사항
본문 앞부분입니다.
[](https://ee.kgu.ac.kr/index.php?page=view&idx=1)
!\[이미지 설명\]\(https://www.kyonggi.ac.kr/site/www/images/noise.png\)
](https://ee.kgu.ac.kr/)
본문 뒷부분입니다.
"""

    cleaned = clean_crawled_markdown(content, source_url="https://ee.kgu.ac.kr/index.php")

    assert "https://ee.kgu.ac.kr" not in cleaned
    assert "이미지 설명" not in cleaned
    assert "본문 앞부분입니다" in cleaned
    assert "본문 뒷부분입니다" in cleaned


def test_clean_crawled_markdown_removes_unclosed_image_alt_noise() -> None:
    content = """
## 학생생활
식당 운영시간은 10:30부터 16:00까지입니다.
![상단 제목: ENJOY K!K!DOG 메뉴 설명이 길게 이어지고 닫히지 않음
"""

    cleaned = clean_crawled_markdown(content, source_url="https://www.kyonggi.ac.kr/www/contents.do?key=5754")

    assert "![" not in cleaned
    assert "ENJOY K" not in cleaned
    assert "식당 운영시간" in cleaned
