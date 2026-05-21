from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KguContact

# 전화 문의 질문에서 제거 (긴 것부터)
_STOPWORDS: tuple[str, ...] = (
    "전화번호",
    "연락처",
    "전화",
    "번호",
    "알려줘",
    "알려",
    "찾아줘",
    "찾아",
    "좀",
    "주세요",
    "주세",
    "해줘",
    "뭐야",
    "뭐지",
    "있어",
    "있니",
    "문의",
    "가르쳐",
    "가르쳐줘",
    "알고싶",
    "알고싶어",
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
    n = _normalize_text(name)
    if not n:
        return []
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    tokens = {_normalize_text(p) for p in parts if len(_normalize_text(p)) >= 2}
    tokens.add(n)
    return list(tokens)


def _score_contact(
    user_keywords: Iterable[str],
    name_norm: str,
    desc_norm: str,
) -> int:
    best = 0
    for kw in user_keywords:
        if len(kw) < 2:
            continue
        for hay in (name_norm, desc_norm):
            if not hay:
                continue
            if kw in hay:
                best = max(best, 100 + len(kw))
    return best


def _pick_best_contact(
    user_input: str,
    rows: list[tuple[str, str, str | None]],
) -> tuple[str, str] | None:
    ranked = _rank_contacts(user_input, rows)
    return (ranked[0][1], ranked[0][2]) if ranked else None


def _rank_contacts(
    user_input: str,
    rows: list[tuple[str, str, str | None]],
) -> list[tuple[int, str, str]]:
    user_norm = _normalize_text(user_input)
    user_after_stop = _strip_stopwords(user_norm)

    user_keywords: list[str] = []
    if len(user_after_stop) >= 2:
        user_keywords.append(user_after_stop)

    ranked: list[tuple[int, str, str]] = []

    for name, phone, description in rows:
        name_norm = _normalize_text(name)
        desc_norm = _normalize_text(description or "")
        tokens = _name_match_tokens(name)

        score = _score_contact(user_keywords, name_norm, desc_norm)
        for kw in user_keywords:
            if len(kw) < 2:
                continue
            for t in tokens:
                if len(t) < 2:
                    continue
                if kw in t or t in kw:
                    score = max(score, 80 + min(len(kw), len(t)))

        if score > 0:
            ranked.append((score, name, phone))

    return sorted(
        ranked,
        key=lambda item: (-item[0], -_contact_priority(item[1], user_norm), item[1]),
    )


def _contact_priority(name: str, user_norm: str) -> int:
    name_norm = _normalize_text(name)
    priority = 0

    if "팀장" in name_norm:
        priority += 30
    if "행정실" in name_norm:
        priority += 20
    if any(keyword in name_norm for keyword in ("수업", "성적", "졸업", "학적", "증명", "교육과정", "교직")):
        priority += 10

    if "강사실" in name_norm and "강사실" not in user_norm:
        priority -= 30
    if "fax" in name_norm and "fax" not in user_norm and "팩스" not in user_norm:
        priority -= 30

    return priority


def get_phone(user_input: str, db: Session) -> str:
    """
    DB에서 부서/시설명·별칭(description)과 질문을 맞춰 전화번호를 반환합니다.
    """
    rows = db.execute(
        select(KguContact.name, KguContact.phone, KguContact.description)
    ).all()
    candidates = [(r[0], str(r[1]).strip(), r[2]) for r in rows]
    ranked = _rank_contacts(user_input, candidates)
    best = (ranked[0][1], ranked[0][2]) if ranked else None

    if best is None:
        un = _normalize_text(user_input)
        us = _strip_stopwords(un)
        if "도서관" in un or "도서" in us or "열람" in un:
            for name, phone, desc in candidates:
                dn = _normalize_text(desc or "")
                if "도서" in dn or "도서관" in _normalize_text(name):
                    best = (name, phone)
                    break
        if best is None and ("대표" in un or "학교" in un or "안내" in un or "총무" in un):
            for name, phone, desc in candidates:
                dn = _normalize_text(desc or "")
                if "대표" in dn or "안내" in dn or "총무" in dn or "일반" in dn:
                    best = (name, phone)
                    break

    if best is None:
        return "📞 요청하신 부서의 전화번호를 찾지 못했습니다. 정확한 부서·시설 이름을 입력해주세요."

    if ranked:
        top_score = ranked[0][0]
        top_matches = [(name, phone) for score, name, phone in ranked if score == top_score][:5]
        if len(top_matches) > 1:
            lines = ["📞 요청하신 부서와 관련된 전화번호입니다."]
            lines.extend(f"- {name}: {phone}" for name, phone in top_matches)
            return "\n".join(lines)

    name, phone = best
    return f"📞 {name} 전화번호는 {phone} 입니다."
