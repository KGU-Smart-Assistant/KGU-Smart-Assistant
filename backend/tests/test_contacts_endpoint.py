from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.models import KguContact


def _contacts():
    return [
        KguContact(
            id=1,
            name="학사지원팀",
            phone="031-249-0000",
            description="academic_affairs",
        ),
        KguContact(
            id=2,
            name="학생지원팀",
            phone="031-249-1111",
            description="student_support",
        ),
    ]


class _ScalarResult:
    def __init__(self, contacts):
        self._contacts = contacts

    def all(self):
        return self._contacts


class _ExecuteResult:
    def __init__(self, contacts):
        self._contacts = contacts

    def scalars(self):
        return _ScalarResult(self._contacts)


class _FakeSession:
    def __init__(self):
        self.contacts = _contacts()

    def execute(self, statement):
        return _ExecuteResult(self.contacts)

    def get(self, model, contact_id):
        return next((contact for contact in self.contacts if contact.id == contact_id), None)


def _override_db():
    yield _FakeSession()


def setup_module() -> None:
    app.dependency_overrides[get_db] = _override_db


def teardown_module() -> None:
    app.dependency_overrides.pop(get_db, None)


def test_get_department_contacts_returns_tel_uris() -> None:
    response = TestClient(app).get("/api/v1/extra/contacts")

    assert response.status_code == 200
    contacts = response.json()["contacts"]
    assert contacts[0] == {
        "department_id": "1",
        "department_name": "학사지원팀",
        "phone_number": "031-249-0000",
        "description": "academic_affairs",
        "tel_uri": "tel:0312490000",
    }


def test_get_department_contact_by_id() -> None:
    response = TestClient(app).get("/api/v1/extra/contacts/2")

    assert response.status_code == 200
    assert response.json() == {
        "department_id": "2",
        "department_name": "학생지원팀",
        "phone_number": "031-249-1111",
        "description": "student_support",
        "tel_uri": "tel:0312491111",
    }


def test_get_department_contact_by_id_returns_404_for_unknown_department() -> None:
    response = TestClient(app).get("/api/v1/extra/contacts/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Department contact not found"}
