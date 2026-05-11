import json
from collections import Counter
from pathlib import Path


DATA_PATH = Path("app/data/intent_training_seed.jsonl")
EXPECTED_ROUTES = {"llm", "relational_db", "rag", "weather"}
EXPECTED_DB_INTENTS = {"unknown", "map", "phone"}


def test_intent_training_seed_data_has_expected_routes_and_db_intents() -> None:
    examples = [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    routes = Counter(example["route"] for example in examples)
    db_intents = Counter(example["db_intent"] for example in examples)

    assert set(routes) == EXPECTED_ROUTES
    assert set(db_intents) == EXPECTED_DB_INTENTS
    assert all(count >= 10 for route, count in routes.items() if route != "relational_db")
    assert db_intents["map"] >= 10
    assert db_intents["phone"] >= 10
    assert db_intents["unknown"] >= 10


def test_intent_training_seed_data_covers_ambiguous_where_questions() -> None:
    examples = [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    route_by_text = {example["text"]: example["route"] for example in examples}
    db_intent_by_text = {example["text"]: example["db_intent"] for example in examples}

    assert route_by_text["성적향상 장학금은 어디에서 정보를 찾을 수 있어?"] == "rag"
    assert db_intent_by_text["성적향상 장학금은 어디에서 정보를 찾을 수 있어?"] == "unknown"
    assert route_by_text["8강의동은 어디야?"] == "relational_db"
    assert db_intent_by_text["8강의동은 어디야?"] == "map"
