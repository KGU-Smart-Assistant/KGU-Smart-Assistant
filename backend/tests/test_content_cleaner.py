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


def test_clean_crawled_markdown_trims_kyonggi_content_footer() -> None:
    content = """
LMS(Learning Management System)
LMS는 온라인 학습관리 시스템입니다.
문의: 031-249-8707
콘텐츠 정보 담당부서 원격교육지원센터 최종수정일2026.01.26
_관련정보_ 더 보시겠어요?
맞춤 설정 맞춤정보 어떤 정보를 찾고계시나요?
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://www.kyonggi.ac.kr/www/contents.do?key=7799",
    )

    assert "LMS는 온라인 학습관리 시스템입니다." in cleaned
    assert "콘텐츠 정보" not in cleaned
    assert "관련정보" not in cleaned
    assert "맞춤설정" not in cleaned
    assert "맞춤정보" not in cleaned


def test_clean_crawled_markdown_removes_kyonggi_widget_text() -> None:
    content = """
사무실 안내
경기대학교 수원캠퍼스 제1복지관 100m 확대축소초기화 로드뷰길찾기지도 크게 보기
주소 수원캠퍼스 제1복지관 2층
한국어 한국어 한국어 English 中文 日本語 한국어 English 中文 日本語 방문학생 프로그램 안내
FAQ 서비스별연락처 선택조건으로 조회 제한검색조건 검색항목 제목 작성자 **도서관이용** 전체 도서관이용 대출/반납/예약 시설이용 홈페이지 온라인컨텐츠 서비스/기타 총 5 건 ,1/1페이지 전체 열기 질문도서관 이용 시간은 어떻게 되나요?
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://www.kyonggi.ac.kr/www/contents.do?key=5283",
    )

    assert "100m 확대축소초기화" not in cleaned
    assert "한국어 한국어" not in cleaned
    assert "선택조건으로 조회" not in cleaned
    assert "주소 수원캠퍼스 제1복지관 2층" in cleaned
    assert "방문학생 프로그램 안내" in cleaned


def test_clean_crawled_markdown_removes_split_map_widget_text() -> None:
    content = """
사무실 안내
경기대학교 경기드림타워
100m
확대축소초기화
로드뷰길찾기지도 크게 보기
주소
수원캠퍼스 경기드림타워 1층 106호
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://www.kyonggi.ac.kr/www/contents.do?key=5478",
    )

    assert "100m" not in cleaned
    assert "확대축소초기화" not in cleaned
    assert "로드뷰길찾기지도" not in cleaned
    assert "수원캠퍼스 경기드림타워 1층 106호" in cleaned


def test_clean_crawled_markdown_removes_split_language_switcher() -> None:
    content = """
한국어
English
中文
日本語
방문학생 프로그램 안내
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://www.kyonggi.ac.kr/international_kgu/contents.do?key=7585",
    )

    assert "English" not in cleaned
    assert "中文" not in cleaned
    assert "日本語" not in cleaned
    assert "방문학생 프로그램 안내" in cleaned


def test_clean_crawled_markdown_trims_academic_affairs_contents_menu() -> None:
    content = (
        "FAQ 학사혁신팀소개 학사일정(학부) 교육과정 _교육실습·교육봉사_ "
        "수업업무 성적안내 학적업무 교직이수 _교육실습·교육봉사_ "
        "교육봉사 1봉사시기: 4학년 2학기 중 이수\n"
        "교육실습 신청 절차 안내"
    )

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://www.kyonggi.ac.kr/www/contents.do?key=8493",
    )

    assert "학사혁신팀소개" not in cleaned
    assert cleaned.startswith("교육봉사")
    assert "교육실습 신청 절차 안내" in cleaned


def test_clean_crawled_markdown_trims_lab_safety_footer_menu() -> None:
    content = """
안전교육안내
연구실안전교육 대상 대학·연구기관 등에서 과학기술분야 연구개발활동에 종사하는 연구원
안전센터 소개 조직도 안전관리구조 오시는 길 긴급연락망
자료실 문서양식 법정자료실 MSDS 안내
(우) 16227 경기도 수원시 영통구 광교산로 154-42(이의동)
Copyright(c) KyongGi University. All rights reserved.
전체메뉴 + 내정보Close
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://safety.kyonggi.ac.kr/ushm/edu/info.do",
    )

    assert "연구실안전교육 대상" in cleaned
    assert "안전센터 소개" not in cleaned
    assert "전체메뉴" not in cleaned
    assert "Copyright" not in cleaned


def test_clean_crawled_markdown_trims_inline_lab_safety_footer_menu() -> None:
    content = (
        "안전교육안내 연구실안전교육 대상 대학·연구기관 등에서 과학기술분야 연구개발활동에 종사하는 연구원 "
        "안전센터 소개 조직도 안전관리구조 오시는 길 긴급연락망 안전교육 안전교육안내 "
        "(우) 16227 경기도 수원시 영통구 광교산로 154-42 Copyright(c) KyongGi University. 전체메뉴 +"
    )

    cleaned = clean_crawled_markdown(
        content,
        source_url="https://safety.kyonggi.ac.kr/ushm/edu/info.do",
    )

    assert "연구실안전교육 대상" in cleaned
    assert "안전센터 소개" not in cleaned
    assert "(우) 16227" not in cleaned
    assert "전체메뉴" not in cleaned


def test_clean_crawled_markdown_cleans_rule_site_controls_and_footer() -> None:
    content = """
규정정보 상세 HOME > 규정정보 > 제1편 학교법인
1-0-1 학교법인 경기학원 정관 2026-04-13 개정 ;) ;) 글자 축소 글자 확대 ;)
개정내역 (32) 담당부서 : 법인사무처 _규정 목차 여닫이 버튼_
**수원캠퍼스 :** (16227) 경기도 수원시 영통구 광교산로 154-42
COPYRIGHT (C) 2021 KYONGGI UNIVERSITY. ALL RIGHTS RESERVED.
"""

    cleaned = clean_crawled_markdown(
        content,
        source_url="http://rule.kyonggi.ac.kr/lmxsrv/law/lawDetail.do?SEQ=26",
    )

    assert "학교법인 경기학원 정관" in cleaned
    assert ";)" not in cleaned
    assert "글자 축소" not in cleaned
    assert "규정 목차" not in cleaned
    assert "수원캠퍼스" not in cleaned
