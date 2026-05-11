from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, KguContact
from app.services.call_service import get_phone


def _session_with_contacts() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add_all(
        [
            KguContact(
                name="경기대학교 수원캠퍼스 학사혁신팀 수업/성적",
                phone="031-249-8732",
                description="교육혁신처, 학사혁신팀, 수업, 성적, 수강신청",
            ),
            KguContact(
                name="경기대학교 수원캠퍼스 학사혁신팀 졸업/복수전공",
                phone="031-249-9099",
                description="교육혁신처, 학사혁신팀, 졸업, 복수전공",
            ),
        ]
    )
    session.commit()
    return session


def test_get_phone_returns_multiple_department_matches() -> None:
    session = _session_with_contacts()

    try:
        reply = get_phone("학사혁신팀 번호 알려줘", session)
    finally:
        session.close()

    assert "학사혁신팀 수업/성적" in reply
    assert "031-249-8732" in reply
    assert "학사혁신팀 졸업/복수전공" in reply
    assert "031-249-9099" in reply


def test_get_phone_uses_task_keyword_to_pick_specific_contact() -> None:
    session = _session_with_contacts()

    try:
        reply = get_phone("졸업 관련 학사혁신팀 번호 알려줘", session)
    finally:
        session.close()

    assert "학사혁신팀 졸업/복수전공" in reply
    assert "031-249-9099" in reply
