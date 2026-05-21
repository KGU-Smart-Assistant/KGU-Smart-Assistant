from scripts.validate_chat_planner import parse_expected_action, validate_plan
from app.services import chat_orchestrator


def test_parse_expected_action_defaults_non_db_intents_to_unknown() -> None:
    assert parse_expected_action("weather").route == "weather"
    assert parse_expected_action("weather").db_intent == "unknown"
    assert parse_expected_action("rag:map").route == "rag"
    assert parse_expected_action("rag:map").db_intent == "unknown"


def test_parse_expected_action_preserves_relational_db_intent() -> None:
    action = parse_expected_action("relational_db:phone")

    assert action.route == "relational_db"
    assert action.db_intent == "phone"


def test_validate_plan_reports_expected_compound_match(monkeypatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", None)
    monkeypatch.setattr(chat_orchestrator, "_WEATHER_KEYWORDS", ("weather",))
    monkeypatch.setattr(
        chat_orchestrator,
        "_RAG_FORCE_GROUP_KEYWORDS",
        (("scholarship_support", ("scholarship",)),),
    )

    report = validate_plan(
        text="weather tomorrow, scholarship deadline",
        expected_actions=[
            parse_expected_action("weather"),
            parse_expected_action("rag"),
        ],
    )

    assert report["matched_expected"] is True
    assert report["actions"] == [
        {
            "route": "weather",
            "db_intent": "unknown",
            "query": "weather tomorrow",
            "reason": "compound segment weather keyword",
        },
        {
            "route": "rag",
            "db_intent": "unknown",
            "query": "scholarship deadline",
            "reason": "rag keyword: scholarship_support",
        },
    ]
