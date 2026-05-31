from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import KguContact

_STOPWORDS: tuple[str, ...] = (
    "전화번호",
    "연락처",
    "문의처",
    "전화",
    "번호",
    "문의",
    "알려줘",
    "알려",
    "찾아줘",
    "찾아",
    "어디",
    "뭐야",
    "주세요",
    "좀",
)


def _normalize_text(value: str) -> str:
    value = value.casefold()
    return re.sub(r"[^\w가-힣]+", "", value)


def _tokenize(value: str) -> list[str]:
    return [
        _normalize_text(token)
        for token in re.split(r"[^\w가-힣]+", value.casefold())
        if len(_normalize_text(token)) >= 2
    ]


def _strip_stopwords(value: str) -> str:
    normalized = _normalize_text(value)
    for word in sorted(_STOPWORDS, key=len, reverse=True):
        normalized = normalized.replace(_normalize_text(word), "")
    return normalized


def _query_terms(user_input: str) -> list[str]:
    terms = set(_tokenize(user_input))
    stripped = _strip_stopwords(user_input)
    if len(stripped) >= 2:
        terms.add(stripped)
    return sorted(terms, key=len, reverse=True)


def _score_contact(
    terms: Iterable[str],
    *,
    name: str,
    description: str | None,
) -> int:
    haystacks = (_normalize_text(name), _normalize_text(description or ""))
    score = 0

    for term in terms:
        if len(term) < 2:
            continue
        for haystack in haystacks:
            if not haystack:
                continue
            if term == haystack:
                score = max(score, 300 + len(term))
            elif term in haystack:
                score = max(score, 200 + len(term))
            elif haystack in term and len(haystack) >= 2:
                score = max(score, 120 + len(haystack))

    return score


def _rank_contacts(
    user_input: str,
    rows: list[tuple[str, str, str | None]],
) -> list[tuple[int, str, str]]:
    terms = _query_terms(user_input)
    ranked: list[tuple[int, str, str]] = []

    for name, phone, description in rows:
        score = _score_contact(terms, name=name, description=description)
        if score > 0:
            ranked.append((score, name, phone))

    return sorted(
        ranked,
        key=lambda item: (-item[0], -_contact_priority(item[1], user_input), item[1], item[2]),
    )


def _contact_priority(name: str, user_input: str) -> int:
    name_norm = _normalize_text(name)
    user_norm = _normalize_text(user_input)
    priority = 0

    if "팀장" in name_norm:
        priority += 30
    if any(keyword in name_norm for keyword in ("수업", "성적", "졸업", "학적", "교육과정", "장학")):
        priority += 20
    if "강사실" in name_norm and "강사실" not in user_norm:
        priority -= 30
    if "fax" in name_norm and "fax" not in user_norm and "팩스" not in user_norm:
        priority -= 30

    return priority


def get_phone(user_input: str, db: Session) -> str:
    try:
        rows = db.execute(select(KguContact.name, KguContact.phone, KguContact.description)).all()
    except SQLAlchemyError:
        return "연락처 DB를 조회하는 중 오류가 발생했습니다."

    candidates = [(row[0], str(row[1]).strip(), row[2]) for row in rows]
    ranked = _rank_contacts(user_input, candidates)
    if not ranked:
        return "요청하신 부서의 전화번호를 찾지 못했습니다. 정확한 부서명이나 업무명을 포함해서 다시 질문해 주세요."

    top_score = ranked[0][0]
    top_matches = [(name, phone) for score, name, phone in ranked if score == top_score][:5]

    if len(top_matches) > 1:
        lines = ["요청하신 내용과 관련된 전화번호입니다."]
        lines.extend(f"- {name}: {phone}" for name, phone in top_matches)
        return "\n".join(lines)

    name, phone = top_matches[0]
    return f"{name} 전화번호는 {phone}입니다."
