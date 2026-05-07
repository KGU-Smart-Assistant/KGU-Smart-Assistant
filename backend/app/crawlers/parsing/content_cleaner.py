from __future__ import annotations

import re
from urllib.parse import urlparse

SKIP_LINE_TOKENS = (
    "본문 바로가기",
    "skip to content",
    "sns공유",
    "공유하기",
    "페이스북",
    "트위터",
    "카카오스토리",
    "네이버블로그",
    "인스타그램",
    "youtube",
    "login",
    "logout",
    "home",
    "추천검색어",
    "인기 검색어",
    "내가찾은 검색어",
    "자동완성",
)

MENU_ONLY_TOKENS = (
    "경기비전",
    "대학상징",
    "대학ㆍ조직",
    "홍보미디어",
    "캠퍼스안내",
    "학교찾아오기",
    "모바일 서비스",
    "전화번호",
    "이사회 회의록",
    "학칙",
)


def clean_crawled_markdown(content: str, *, source_url: str = "") -> str:
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


def _trim_to_content_region(lines: list[str]) -> list[str]:
    start = 0
    for index, line in enumerate(lines):
        normalized = _normalize(line)
        if normalized.startswith("## ") or normalized in {
            "학사일정(학부)",
            "졸업요건",
            "faq",
        }:
            start = index
            break
        if "## " in line:
            start = index
            break

    end = len(lines)
    for index in range(start, len(lines)):
        normalized = _normalize(lines[index])
        if normalized.startswith(("이전글", "다음글", "목록", "첨부파일")):
            end = index
            break
    return lines[start:end]


def _strip_markdown_noise(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"^#+\s*", "", line)
    return re.sub(r"\s+", " ", line).strip()


def _should_drop_line(line: str, *, source_url: str) -> bool:
    normalized = _normalize(line)
    if not normalized:
        return True
    if len(normalized) <= 1:
        return True
    if any(token in normalized for token in SKIP_LINE_TOKENS):
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
    if sum(token in normalized for token in MENU_ONLY_TOKENS) >= 2:
        return True
    path = urlparse(source_url).path.lower()
    if "contents.do" in path and normalized in {"home", "경기소개", "대학생활"}:
        return True
    return False


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
