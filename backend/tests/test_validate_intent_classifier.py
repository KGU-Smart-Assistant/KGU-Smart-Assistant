import pytest

from scripts.validate_intent_classifier import parse_model_label, validate_text


def test_parse_model_label_maps_internal_relational_db_labels() -> None:
    assert parse_model_label("relational_db:map") == ("relational_db", "map")
    assert parse_model_label("relational_db/phone") == ("relational_db", "phone")
    assert parse_model_label("relational_db__unknown") == ("relational_db", "unknown")


def test_parse_model_label_maps_route_only_labels_to_unknown_db_intent() -> None:
    assert parse_model_label("rag") == ("rag", "unknown")
    assert parse_model_label("weather") == ("weather", "unknown")
    assert parse_model_label("llm") == ("llm", "unknown")


def test_parse_model_label_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        parse_model_label("unsupported")


def test_validate_text_reports_expected_match(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_classify(text: str, model_name: str, device: int) -> dict[str, object]:
        return {"label": "relational_db:map", "score": 0.91}

    monkeypatch.setattr("scripts.validate_intent_classifier._classify", fake_classify)

    result = validate_text(
        text="8강의동은 어디야?",
        model_name="unused",
        threshold=0.7,
        device=-1,
        expected_route="relational_db",
        expected_db_intent="map",
    )

    assert result.route == "relational_db"
    assert result.db_intent == "map"
    assert result.accepted is True
    assert result.matched_expected is True


def test_validate_text_reports_expected_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_classify(text: str, model_name: str, device: int) -> dict[str, object]:
        return {"label": "rag", "score": 0.82}

    monkeypatch.setattr("scripts.validate_intent_classifier._classify", fake_classify)

    result = validate_text(
        text="성적향상 장학금은 어디에서 정보를 찾을 수 있어?",
        model_name="unused",
        threshold=0.7,
        device=-1,
        expected_route="relational_db",
        expected_db_intent="map",
    )

    assert result.route == "rag"
    assert result.db_intent == "unknown"
    assert result.accepted is True
    assert result.matched_expected is False
