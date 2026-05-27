from pathlib import Path

import pytest

from scripts.evaluate_intent_classifier import (
    EvalExample,
    EvalPrediction,
    build_report,
    load_examples,
)


def test_load_examples_normalizes_non_db_intents_to_unknown(tmp_path: Path) -> None:
    data = tmp_path / "eval.jsonl"
    data.write_text(
        '{"text":"장학금 안내 알려줘","route":"rag","db_intent":"map"}\n'
        '{"text":"8강의동 어디야?","route":"map"}\n',
        encoding="utf-8",
    )

    examples = load_examples(data)

    assert examples == [
        EvalExample(text="장학금 안내 알려줘", route="rag", db_intent="unknown"),
        EvalExample(text="8강의동 어디야?", route="relational_db", db_intent="map"),
    ]


def test_load_examples_rejects_invalid_route(tmp_path: Path) -> None:
    data = tmp_path / "eval.jsonl"
    data.write_text(
        '{"text":"테스트","route":"multi","db_intent":"unknown"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported route"):
        load_examples(data)


def test_build_report_counts_accuracy_and_errors() -> None:
    predictions = [
        EvalPrediction(
            text="중앙도서관 위치 알려줘",
            expected_route="relational_db",
            expected_db_intent="map",
            predicted_route="relational_db",
            predicted_db_intent="map",
            predicted_label="relational_db:map",
            confidence=0.91,
            accepted=True,
        ),
        EvalPrediction(
            text="장학금 안내 어디서 봐?",
            expected_route="rag",
            expected_db_intent="unknown",
            predicted_route="relational_db",
            predicted_db_intent="map",
            predicted_label="relational_db:map",
            confidence=0.81,
            accepted=True,
        ),
        EvalPrediction(
            text="안녕",
            expected_route="llm",
            expected_db_intent="unknown",
            predicted_route="llm",
            predicted_db_intent="unknown",
            predicted_label="llm",
            confidence=0.42,
            accepted=False,
        ),
    ]

    report = build_report(predictions, threshold=0.7, include_errors=True)

    assert report["total"] == 3
    assert report["correct"] == 2
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["accepted_total"] == 2
    assert report["accepted_correct"] == 1
    assert report["accepted_accuracy"] == pytest.approx(1 / 2)
    assert report["expected_counts"] == {
        "llm": 1,
        "rag": 1,
        "relational_db:map": 1,
    }
    assert report["predicted_counts"] == {
        "llm": 1,
        "relational_db:map": 2,
    }
    assert report["errors"] == [
        {
            "text": "장학금 안내 어디서 봐?",
            "expected": "rag",
            "predicted": "relational_db:map",
            "raw_label": "relational_db:map",
            "confidence": 0.81,
            "accepted": True,
        }
    ]
