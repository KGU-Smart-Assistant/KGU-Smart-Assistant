from __future__ import annotations

import re
from urllib.parse import urlparse

SKIP_LINE_TOKENS = (
    "본문 바로가기",
    "skip to content",
    "sns공유",
    "공유하기",
    "사이트맵",
    "트위터",
    "페이스북",
    "카카오스토리",
    "네이버블로그",
    "인스타그램",
    "카카오톡",
    "youtube",
    "login",
    "logout",
    "로그인",
    "닫기",
    "인쇄",
    "language",
    "통합검색",
    "추천검색어",
    "인기 검색어",
    "내가찾은 검색어",
    "내가 찾은 검색어",
    "자동완성",
)

MENU_ONLY_TOKENS = (
    "경기비전",
    "대학상징",
    "대학조직",
    "홍보미디어",
    "캠퍼스안내",
    "학칙찾아보기",
    "모바일서비스",
    "전화번호",
    "이사회회의록",
    "학칙",
    "학과소개",
    "전공소개",
    "교과과정",
    "연구실",
    "커뮤니티",
)

PORTAL_CONTENT_MARKERS = (
    "### ",
    "\n## 공지사항",
    "\n## 자료실",
    "\n## FAQ",
    "\n## 학사일정",
    "\n## 장학공지",
    "\n## 일반공지",
    "\n## 학사공지",
    "\n## 채용공고",
)

FOOTER_MARKERS = (
    "\n[목록]",
    "\n게시물삭제",
    "\n이전글",
    "\n다음글",
    "\n개인정보처리방침",
    "\n이메일무단수집거부",
    "수원캠퍼스(16227)",
    "대학정보공시",
    "Copyright (C) 2020 Kyonggi University. School of Electronic Engineering.",
    "All Rights Reserved.",
    "콘텐츠 정보",
    "_관련정보_",
    "관련정보",
    "맞춤 설정",
    "맞춤설정",
    "맞춤정보",
    "어떤 정보를 찾고계시나요?",
)

IGNORED_HEADINGS = {
    "## 주메뉴",
    "## 서브메뉴",
    "## 전체메뉴",
}

PREFERRED_HEADINGS = {
    "## 공지사항",
    "## 자료실",
    "## FAQ",
    "## 학사일정",
    "## 장학공지",
    "## 일반공지",
    "## 학사공지",
    "## 채용공고",
}


def clean_crawled_markdown(content: str, *, source_url: str = "") -> str:
    content = _preclean_source_document(content, source_url=source_url)
    lines = [line.strip() for line in content.splitlines()]
    lines = _trim_to_content_region(lines)

    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        line = _strip_markdown_noise(line)
        if not line:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        previous_blank = False

        if _should_drop_line(line, source_url=source_url):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _preclean_source_document(content: str, *, source_url: str) -> str:
    host = urlparse(source_url).netloc.lower()
    if host.endswith("kyonggi.ac.kr") or host.endswith("kgu.ac.kr"):
        content = _trim_by_markers(content, PORTAL_CONTENT_MARKERS)
        content = _trim_footer(content)
    return content


def _trim_by_markers(content: str, markers: tuple[str, ...]) -> str:
    positions = [content.find(marker) for marker in markers if marker in content]
    if not positions:
        return content
    return content[min(positions) :]


def _trim_footer(content: str) -> str:
    positions = [content.find(marker) for marker in FOOTER_MARKERS if marker in content]
    if not positions:
        return content
    return content[: min(positions)]


def _trim_to_content_region(lines: list[str]) -> list[str]:
    start = 0
    ignored_headings = {_normalize(value) for value in IGNORED_HEADINGS}
    preferred_headings = {_normalize(value) for value in PREFERRED_HEADINGS}

    for index, line in enumerate(lines):
        normalized = _normalize(line)
        if normalized in preferred_headings:
            start = index
            break
        if normalized in ignored_headings:
            continue
        if normalized.startswith("## ") or normalized in {"faq"}:
            start = index
            break
        if "## " in line:
            start = index
            break

    end = len(lines)
    for index in range(start, len(lines)):
        normalized = _normalize(lines[index])
        if normalized.startswith(
            (
                "이전글",
                "다음글",
                "목록",
                "첨부파일",
                "게시물삭제",
                "개인정보처리방침",
                "이메일무단수집거부",
                "콘텐츠 정보",
                "_관련정보_",
                "관련정보",
                "맞춤 설정",
                "맞춤설정",
                "맞춤정보",
                "어떤 정보를 찾고계시나요?",
            )
        ):
            end = index
            break
    return lines[start:end]


def _strip_markdown_noise(line: str) -> str:
    line = re.sub(r"!\\?\[[^\]]*\]\\?\([^)]+\)", "", line)
    line = re.sub(r"!\\?\[.*$", "", line)
    line = re.sub(r"\[([^\]]*)\]\(javascript:[^)]+\)", r"\1", line, flags=re.IGNORECASE)
    line = re.sub(r"javascript:\S+", "", line, flags=re.IGNORECASE)
    line = line.replace("100m 확대축소초기화 로드뷰길찾기지도 크게 보기", "")
    line = re.sub(
        r"FAQ\s+서비스별연락처\s+선택조건으로 조회\s+제한검색조건\s+검색항목\s+제목\s+작성자.*?전체 열기",
        "FAQ ",
        line,
    )
    line = re.sub(r"^(한국어\s+){2,}English\s+中文\s+日本語\s+(한국어\s+English\s+中文\s+日本語\s+)?", "", line)
    line = line.replace("카카오톡 닫기 인쇄", "")
    line = line.replace("SNS공유", "")
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    line = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"\\?\]\\?\(https?://[^)]+\)", "", line)
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"^#+\s*", "", line)
    return re.sub(r"\s+", " ", line).strip()


def _should_drop_line(line: str, *, source_url: str) -> bool:
    normalized = _normalize(line)
    if not normalized:
        return True
    if len(normalized) <= 1:
        return True
    if normalized in {"100m", "확대축소초기화", "로드뷰길찾기지도 크게 보기"}:
        return True
    if normalized in {"한국어", "english", "中文", "日本語"}:
        return True
    if any(token.casefold() in normalized for token in SKIP_LINE_TOKENS):
        return True
    if _looks_like_url_only(line):
        return True
    if _looks_like_menu_line(normalized, source_url=source_url):
        return True
    return False


def _looks_like_url_only(line: str) -> bool:
    if line.count("http://") + line.count("https://") >= 2:
        return True
    return bool(re.fullmatch(r"https?://\S+", line))


def _looks_like_menu_line(normalized: str, *, source_url: str) -> bool:
    if sum(token.casefold() in normalized for token in MENU_ONLY_TOKENS) >= 2:
        return True
    path = urlparse(source_url).path.lower()
    if "contents.do" in path and normalized in {"home", "경기소개", "대학생활"}:
        return True
    return False


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
