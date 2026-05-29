from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import KguPlace

_STOPWORDS: tuple[str, ...] = (
    "위치",
    "어디",
    "가는길",
    "가는",
    "길",
    "지도",
    "찾아줘",
    "찾아",
    "알려줘",
    "알려",
    "있어",
    "있나요",
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


def _score_place(
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


def _pick_best_match(
    user_input: str,
    rows: list[tuple[str, str | None, float, float]],
) -> tuple[str, float, float] | None:
    terms = _query_terms(user_input)
    ranked: list[tuple[int, str, float, float]] = []

    for name, description, latitude, longitude in rows:
        score = _score_place(terms, name=name, description=description)
        if score > 0:
            ranked.append((score, name, latitude, longitude))

    if not ranked:
        return None

    _, name, latitude, longitude = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    return name, latitude, longitude


def get_map_response(user_input: str, db: Session) -> str:
    try:
        rows = db.execute(
            select(KguPlace.name, KguPlace.description, KguPlace.latitude, KguPlace.longitude)
        ).all()
    except SQLAlchemyError:
        return "캠퍼스 위치 DB를 조회하는 중 오류가 발생했습니다."

    candidates = [(row[0], row[1], float(row[2]), float(row[3])) for row in rows]
    best = _pick_best_match(user_input, candidates)

    if best is None:
        return "요청하신 위치를 찾지 못했습니다. 정확한 건물명이나 장소명을 포함해서 다시 질문해 주세요."

    name, latitude, longitude = best
    maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
    return f"{name} 위치입니다. 위도/경도: ({latitude}, {longitude})\n지도: {maps_url}"
