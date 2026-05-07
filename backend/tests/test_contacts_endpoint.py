from fastapi.testclient import TestClient

from app.main import app


def test_get_department_contacts_returns_tel_uris() -> None:
    response = TestClient(app).get("/api/v1/extra/contacts")

    assert response.status_code == 200
    contacts = response.json()["contacts"]
    assert contacts[0] == {
        "department_id": "academic_affairs",
        "department_name": "학사지원팀",
        "phone_number": "031-249-0000",
        "tel_uri": "tel:0312490000",
    }


def test_get_department_contact_by_id() -> None:
    response = TestClient(app).get("/api/v1/extra/contacts/student_support")

    assert response.status_code == 200
    assert response.json() == {
        "department_id": "student_support",
        "department_name": "학생지원팀",
        "phone_number": "031-249-1111",
        "tel_uri": "tel:0312491111",
    }


def test_get_department_contact_by_id_returns_404_for_unknown_department() -> None:
    response = TestClient(app).get("/api/v1/extra/contacts/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Department contact not found"}
