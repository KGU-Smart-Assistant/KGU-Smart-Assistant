from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KguPlace

# DB가 비어 있거나 매칭 실패 시에도 자주 묻는 장소(중앙도서관) — JSON과 동일 좌표 유지
_FALLBACK_LIBRARY: tuple[str, float, float] = (
    "경기대학교 수원캠퍼스 중앙도서관",
    37.30125,
    127.03645,
)

# 질문에서 제거(긴 것부터 처리)
_STOPWORDS: tuple[str, ...] = (
    "어디있어",
    "어디있니",
    "어딨어",
    "어딨니",
    "어디야",
    "어디에",
    "어디",
    "어딨",
    "물어봐",
    "알려줘",
    "알려",
    "찾아줘",
    "찾아",
    "위치",
    "가는길",
    "가는",
    "길",
    "좀",
    "주세요",
    "주세",
    "해줘",
    "해봐",
    "알아",
    "뭐야",
    "뭐지",
    "있어",
    "있니",
    "없어",
    "없니",
)


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s가-힣]", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _strip_stopwords(s: str) -> str:
    for w in sorted(_STOPWORDS, key=len, reverse=True):
        s = s.replace(w, "")
    return s


def _name_match_tokens(name: str) -> list[str]:
    """장소명에서 비교에 쓸 토큰(띄어쓰기 단위 + 전체)."""
    n = _normalize_text(name)
    if not n:
        return []
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    tokens = {_normalize_text(p) for p in parts if len(_normalize_text(p)) >= 2}
    tokens.add(n)
    return list(tokens)


def _score_place(
    user_keywords: Iterable[str],
    name_norm: str,
    desc_norm: str,
) -> int:
    """질문 키워드가 이름/설명에 포함되면 점수. 길수록 우선."""
    haystacks = (name_norm, desc_norm)
    best = 0
    for kw in user_keywords:
        if len(kw) < 2:
            continue
        for hay in haystacks:
            if not hay:
                continue
            if kw in hay:
                best = max(best, 100 + len(kw))
    return best


def _pick_best_match(
    user_input: str,
    rows: list[tuple[str, str | None, float, float]],
) -> tuple[str, float, float] | None:
    user_norm = _normalize_text(user_input)
    user_after_stop = _strip_stopwords(user_norm)

    # 질문 핵심 키워드: 불용어 제거 후 남는 부분만 (예: "도서관어딨어" -> "도서관")
    user_keywords: list[str] = []
    if len(user_after_stop) >= 2:
        user_keywords.append(user_after_stop)

    best: tuple[str, float, float] | None = None
    best_score = -1

    for name, description, lat, lon in rows:
        name_norm = _normalize_text(name)
        desc_norm = _normalize_text(description or "")
        tokens = _name_match_tokens(name)

        score = _score_place(user_keywords, name_norm, desc_norm)
        # "도서관" 처럼 짧은 호칭이 장소명 일부와 일치
        for kw in user_keywords:
            if len(kw) < 2:
                continue
            for t in tokens:
                if len(t) < 2:
                    continue
                if kw in t or t in kw:
                    score = max(score, 80 + min(len(kw), len(t)))

        if score > best_score:
            best_score = score
            best = (name, float(lat), float(lon))

    return best if best_score > 0 else None


def get_map_response(user_input: str, db: Session) -> str:
    """
    사용자 자연어에서 장소를 유추하고 좌표/지도 링크 반환
    """
    rows = db.execute(
        select(KguPlace.name, KguPlace.description, KguPlace.latitude, KguPlace.longitude)
    ).all()
    candidates = [(r[0], r[1], float(r[2]), float(r[3])) for r in rows]
    best = _pick_best_match(user_input, candidates)

    if best is None:
        un = _normalize_text(user_input)
        us = _strip_stopwords(un)
        if (
            "도서관" in un
            or "중앙도서관" in un
            or "열람실" in un
            or "도서관" in us
            or "중앙도서관" in us
        ):
            best = (_FALLBACK_LIBRARY[0], _FALLBACK_LIBRARY[1], _FALLBACK_LIBRARY[2])

    if best is None:
        return "📍 요청하신 위치를 찾지 못했습니다. 정확한 건물/장소 이름을 포함해서 질문해주세요."

    name, lat, lon = best
    maps_url = f"https://www.google.com/maps?q={lat},{lon}"
    return f"📍 {name} 위치입니다. 위도/경도: ({lat}, {lon})\n지도: {maps_url}"
