from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from datetime import datetime

from app.models import Base, CrawlerDocument, CrawlerDocumentChunk, CrawlerSource, KguContact, KguInfoLink, KguPlace
from app.services.call_service import get_phone
from app.services.map_service import get_map_response
from app.services.relational_db_service import answer_from_relational_db_search


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add_all(
        [
            CrawlerSource(
                name="scholarship_notice",
                domain="scholarship",
                department=None,
                seed_urls_json="[]",
                status="active",
                last_seen_at=datetime(2026, 5, 29),
                last_crawled_at=datetime(2026, 5, 29),
            ),
            CrawlerDocument(
                doc_id="doc-scholarship-1",
                source_name="scholarship_notice",
                source_url="https://example.com/scholarship-doc",
                title="장학금 신청 안내",
                content="장학금 신청 기간과 제출 서류 안내입니다.",
                content_hash="hash-doc-scholarship-1",
                source_type="html",
                doc_type="scholarship",
                domain="scholarship",
                department=None,
                author_department=None,
                published_at=datetime(2026, 5, 1),
                collected_at=datetime(2026, 5, 29),
                last_seen_at=datetime(2026, 5, 29),
                status="active",
            ),
            CrawlerDocumentChunk(
                chunk_id="chunk-scholarship-1",
                doc_id="doc-scholarship-1",
                chunk_index=0,
                text="장학금 신청 기간은 5월 1일부터 5월 10일까지이며 제출 서류는 신청서와 성적증명서입니다.",
                content_hash="hash-chunk-scholarship-1",
                title="장학금 신청 안내",
                source_url="https://example.com/scholarship-doc",
                source_type="html",
                status="active",
                last_seen_at=datetime(2026, 5, 29),
            ),
            KguContact(
                name="장학지원팀",
                phone="031-249-8785",
                description="장학금 국가장학금 교내장학금",
            ),
            KguPlace(
                name="중앙도서관",
                description="수원캠퍼스 도서관",
                latitude=37.30125,
                longitude=127.03645,
            ),
            KguInfoLink(
                group_id="scholarship",
                group_title="kguInfo.scholarship.title",
                group_order=0,
                label="kguInfo.scholarship.notice",
                url="https://www.kyonggi.ac.kr/www/selectBbsNttList.do?key=7996&bbsNo=1073&dc=11D70",
                link_order=0,
                is_active=True,
            ),
        ]
    )
    session.commit()
    return session


def test_get_phone_searches_postgresql_contact_rows() -> None:
    session = _session()
    try:
        reply = get_phone("장학지원팀 전화번호 알려줘", session)
    finally:
        session.close()

    assert "장학지원팀" in reply
    assert "031-249-8785" in reply


def test_get_map_response_searches_postgresql_place_rows() -> None:
    session = _session()
    try:
        reply = get_map_response("중앙도서관 위치 알려줘", session)
    finally:
        session.close()

    assert "중앙도서관" in reply
    assert "37.30125" in reply
    assert "https://www.google.com/maps?q=37.30125,127.03645" in reply


def test_relational_db_fallback_searches_info_links() -> None:
    session = _session()
    try:
        answer = answer_from_relational_db_search("장학금 공지 링크 알려줘", session)
    finally:
        session.close()

    assert answer.intent == "바로가기"
    assert answer.source_title == "kgu_info_links"
    assert "kguInfo.scholarship.notice" in answer.reply
    assert "https://www.kyonggi.ac.kr/www/selectBbsNttList.do" in answer.reply


def test_relational_db_fallback_searches_crawled_chunks() -> None:
    session = _session()
    try:
        answer = answer_from_relational_db_search("장학금 제출 서류 알려줘", session)
    finally:
        session.close()

    assert answer.intent == "문서검색"
    assert answer.source_title == "장학금 신청 안내"
    assert "성적증명서" in answer.reply
    assert "https://example.com/scholarship-doc" in answer.reply
