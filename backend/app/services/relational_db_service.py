from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import CrawlerDocument, CrawlerDocumentChunk, KguContact, KguInfoLink, KguPlace


@dataclass(frozen=True)
class RelationalDbAnswer:
    reply: str
    intent: str
    source_title: str
    source_url: str | None = None
    answered: bool = True


@dataclass(frozen=True)
class _Candidate:
    score: int
    kind: str
    title: str
    reply: str
    source_title: str
    source_url: str | None = None


_STOPWORDS = (
    "알려줘",
    "알려",
    "찾아줘",
    "찾아",
    "어디",
    "뭐야",
    "주세요",
    "관련",
    "정보",
    "조회",
    "검색",
    "링크",
    "url",
    "URL",
    "바로가기",
    "페이지",
    "홈페이지",
    "공지",
    "공지사항",
    "안내",
    "전화번호",
    "전화",
    "연락처",
    "문의처",
    "위치",
    "지도",
    "가는길",
)

_INFO_LINK_ALIASES: dict[str, tuple[str, ...]] = {
    "notice": ("공지", "공지사항", "자료실"),
    "academiccalendar": ("학사일정", "일정"),
    "classes": ("수업", "수강", "수강신청", "성적"),
    "schoolregister": ("학적", "휴학", "복학", "자퇴", "전과", "다전공"),
    "graduation": ("졸업", "졸업요건"),
    "scholarship": ("장학", "장학금", "국가장학금"),
    "tuition": ("등록금", "납부"),
    "certificate": ("증명서", "증명", "양식"),
    "career": ("취업", "진로", "현장실습"),
    "studentlife": ("학생생활", "식단", "학식", "셔틀", "lms", "kutis", "기숙사"),
    "integrated": ("통합",),
    "materials": ("자료실", "자료", "서류", "양식"),
    "schedule": ("일정",),
    "registration": ("수강신청", "신청"),
    "notice": ("공지", "공지사항"),
    "website": ("홈페이지", "사이트"),
    "faq": ("faq", "자주묻는질문"),
    "menu": ("식단", "학식", "메뉴"),
    "phone": ("전화", "연락처"),
}


def _normalize(value: str) -> str:
    return re.sub(r"[^\w가-힣]+", "", value.casefold())


def _tokens(value: str) -> tuple[str, ...]:
    raw_tokens = re.split(r"[^\w가-힣]+", value.casefold())
    tokens = {_normalize(token) for token in raw_tokens if len(_normalize(token)) >= 2}
    compact = _normalize(value)
    for stopword in sorted(_STOPWORDS, key=len, reverse=True):
        compact = compact.replace(_normalize(stopword), "")
    if len(compact) >= 2:
        tokens.add(compact)
    return tuple(sorted(tokens, key=len, reverse=True))


def _score(query_tokens: tuple[str, ...], *fields: str | None) -> int:
    haystacks = [_normalize(field or "") for field in fields]
    best = 0
    for token in query_tokens:
        if len(token) < 2:
            continue
        for haystack in haystacks:
            if not haystack:
                continue
            if token == haystack:
                best = max(best, 300 + len(token))
            elif token in haystack:
                best = max(best, 200 + len(token))
            elif haystack in token and len(haystack) >= 2:
                best = max(best, 120 + len(haystack))
    return best


def _wants_link(user_input: str) -> bool:
    normalized = _normalize(user_input)
    return any(keyword in normalized for keyword in ("링크", "url", "바로가기", "페이지", "홈페이지", "사이트"))


def _wants_phone(user_input: str) -> bool:
    normalized = _normalize(user_input)
    return any(keyword in normalized for keyword in ("전화", "전화번호", "연락처", "문의처"))


def _wants_map(user_input: str) -> bool:
    normalized = _normalize(user_input)
    return any(keyword in normalized for keyword in ("위치", "지도", "가는길", "어디"))


def _info_link_search_text(group_title: str, label: str, url: str) -> str:
    keys = re.split(r"[^A-Za-z0-9]+", f"{group_title} {label}")
    aliases: list[str] = []
    for key in keys:
        normalized_key = key.casefold()
        aliases.extend(_INFO_LINK_ALIASES.get(normalized_key, ()))
    return " ".join((group_title, label, url, *aliases))


def answer_from_relational_db_search(user_input: str, db: Session) -> RelationalDbAnswer:
    tokens = _tokens(user_input)
    candidates: list[_Candidate] = []

    try:
        contact_rows = db.execute(select(KguContact.name, KguContact.phone, KguContact.description)).all()
        place_rows = db.execute(
            select(KguPlace.name, KguPlace.description, KguPlace.latitude, KguPlace.longitude)
        ).all()
        link_rows = (
            db.execute(
                select(KguInfoLink.group_title, KguInfoLink.label, KguInfoLink.url)
                .where(KguInfoLink.is_active.is_(True))
                .order_by(KguInfoLink.group_order, KguInfoLink.link_order)
            )
            .all()
        )
        crawler_rows = (
            db.execute(
                select(
                    CrawlerDocumentChunk.title,
                    CrawlerDocumentChunk.text,
                    CrawlerDocumentChunk.source_url,
                    CrawlerDocument.domain,
                    CrawlerDocument.department,
                )
                .join(CrawlerDocument, CrawlerDocument.doc_id == CrawlerDocumentChunk.doc_id)
                .where(CrawlerDocumentChunk.status.in_(("active", "updated")))
                .order_by(CrawlerDocumentChunk.last_seen_at.desc(), CrawlerDocumentChunk.chunk_id)
                .limit(100)
            )
            .all()
        )
    except SQLAlchemyError:
        return RelationalDbAnswer(
            reply="PostgreSQL 데이터를 조회하는 중 오류가 발생했습니다.",
            intent="DB",
            source_title="relational_db",
            answered=False,
        )

    for name, phone, description in contact_rows:
        score = _score(tokens, name, description)
        if _wants_phone(user_input):
            score += 80
        if score > 0:
            candidates.append(
                _Candidate(
                    score=score,
                    kind="phone",
                    title=name,
                    reply=f"{name} 전화번호는 {phone}입니다.",
                    source_title="kgu_contacts",
                )
            )

    for name, description, latitude, longitude in place_rows:
        score = _score(tokens, name, description)
        if _wants_map(user_input):
            score += 80
        if score > 0:
            maps_url = f"https://www.google.com/maps?q={float(latitude)},{float(longitude)}"
            candidates.append(
                _Candidate(
                    score=score,
                    kind="map",
                    title=name,
                    reply=f"{name} 위치입니다. 위도/경도: ({float(latitude)}, {float(longitude)})\n지도: {maps_url}",
                    source_title="kgu_places",
                    source_url=maps_url,
                )
            )

    for group_title, label, url in link_rows:
        search_text = _info_link_search_text(group_title, label, url)
        score = _score(tokens, search_text)
        if _wants_link(user_input):
            score += 80
        if score > 0:
            candidates.append(
                _Candidate(
                    score=score,
                    kind="info_link",
                    title=label,
                    reply=f"{label} 바로가기입니다.\n{url}",
                    source_title="kgu_info_links",
                    source_url=url,
                )
            )

    for title, text, source_url, domain, department in crawler_rows:
        score = _score(tokens, title, text, domain, department)
        if score > 0:
            snippet = _snippet(text, tokens)
            candidates.append(
                _Candidate(
                    score=score,
                    kind="crawler_document",
                    title=title,
                    reply=f"{title}에서 찾은 내용입니다.\n{snippet}\n출처: {source_url}",
                    source_title=title,
                    source_url=source_url,
                )
            )

    if not candidates:
        return RelationalDbAnswer(
            reply=(
                "PostgreSQL에서 질문과 일치하는 데이터를 찾지 못했습니다. "
                "부서명, 장소명, 또는 바로가기 이름을 더 구체적으로 입력해 주세요."
            ),
            intent="DB",
            source_title="relational_db",
            answered=False,
        )

    best = sorted(candidates, key=lambda item: (-item.score, item.kind, item.title))[0]
    intent = {
        "phone": "전화",
        "map": "지도",
        "info_link": "바로가기",
        "crawler_document": "문서검색",
    }.get(best.kind, "DB")
    return RelationalDbAnswer(
        reply=best.reply,
        intent=intent,
        source_title=best.source_title,
        source_url=best.source_url,
    )


def _snippet(text: str, query_tokens: tuple[str, ...], max_length: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_length:
        return cleaned

    normalized = _normalize(cleaned)
    first_match = min(
        (
            normalized.find(token)
            for token in query_tokens
            if token and normalized.find(token) >= 0
        ),
        default=0,
    )
    start = max(first_match - 80, 0)
    end = min(start + max_length, len(cleaned))
    snippet = cleaned[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(cleaned):
        snippet = f"{snippet}..."
    return snippet
