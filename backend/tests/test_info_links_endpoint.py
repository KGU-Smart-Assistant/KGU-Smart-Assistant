from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.models import KguInfoLink


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _ExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _ScalarResult(self._rows)


class _FakeSession:
    def execute(self, statement):
        return _ExecuteResult(
            [
                KguInfoLink(
                    id=1,
                    group_id="notice",
                    group_title="kguInfo.notice.title",
                    group_order=0,
                    label="kguInfo.notice.integrated",
                    url="https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&key=7520",
                    link_order=0,
                    is_active=True,
                ),
                KguInfoLink(
                    id=2,
                    group_id="notice",
                    group_title="kguInfo.notice.title",
                    group_order=0,
                    label="kguInfo.notice.materials",
                    url="https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=684&key=5143",
                    link_order=1,
                    is_active=True,
                ),
            ]
        )


def _override_db():
    yield _FakeSession()


def setup_module() -> None:
    app.dependency_overrides[get_db] = _override_db


def teardown_module() -> None:
    app.dependency_overrides.pop(get_db, None)


def test_get_info_links_groups_links_from_db() -> None:
    response = TestClient(app).get("/api/v1/extra/info-links")

    assert response.status_code == 200
    assert response.json() == {
        "groups": [
            {
                "id": "notice",
                "title": "kguInfo.notice.title",
                "links": [
                    {
                        "label": "kguInfo.notice.integrated",
                        "url": "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=1073&key=7520",
                    },
                    {
                        "label": "kguInfo.notice.materials",
                        "url": "https://www.kyonggi.ac.kr/www/selectBbsNttList.do?bbsNo=684&key=5143",
                    },
                ],
            }
        ]
    }
