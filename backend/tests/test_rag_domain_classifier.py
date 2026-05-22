from pathlib import Path

import pytest

from app.services import chat_orchestrator
from app.services.rag_domain_classifier import (
    RAG_DOMAIN_LABELS,
    RagDomainPrediction,
    _predictions_from_output,
    labels_to_multihot,
)
from scripts.train_rag_domain_classifier import load_examples


def test_predictions_from_output_maps_label_indexes_and_domain_labels() -> None:
    predictions = _predictions_from_output(
        [
            {"label": "LABEL_0", "score": 0.91},
            {"label": "tuition", "score": 0.72},
            {"label": "unsupported", "score": 0.99},
        ]
    )

    assert predictions == [
        RagDomainPrediction(domain=RAG_DOMAIN_LABELS[0], score=0.91),
        RagDomainPrediction(domain="tuition", score=0.72),
    ]


def test_labels_to_multihot_marks_multiple_domains() -> None:
    vector = labels_to_multihot(["academic_status", "tuition", "scholarship"])

    assert vector[RAG_DOMAIN_LABELS.index("academic_status")] == 1.0
    assert vector[RAG_DOMAIN_LABELS.index("tuition")] == 1.0
    assert vector[RAG_DOMAIN_LABELS.index("scholarship")] == 1.0
    assert sum(vector) == 3.0


def test_load_examples_reads_expected_domains_from_rag_eval_data() -> None:
    examples = load_examples(Path("app/data/rag_intent_eval.jsonl"))

    complex_example = next(
        example
        for example in examples
        if example.text == "휴학하면 등록금이랑 장학금은 어떻게 돼?"
    )
    assert complex_example.domains == ("academic_status", "tuition", "scholarship")


def test_rag_domain_classifier_scores_replace_rule_based_domain_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: type("Prediction", (), {"route": "rag", "db_intent": "unknown", "confidence": 0.99, "label": "rag"})(),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_details_with_klue_bert",
        lambda _: (type("Prediction", (), {"detail": "period", "score": 0.99})(),),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_domains_with_klue_bert",
        lambda _: (
            RagDomainPrediction(domain="tuition", score=0.95),
            RagDomainPrediction(domain="scholarship", score=0.72),
        ),
    )

    decision = chat_orchestrator.decide_chat_route("scholarship deadline")

    assert decision.route == "rag"
    assert decision.rag_domain == "tuition"
    assert list(decision.rag_domains)[:2] == ["tuition", "scholarship"]
    assert decision.rag_ambiguity == "clear"
    assert decision.rewritten_queries
    assert "tuition" in decision.rewritten_queries[-1]


def test_rag_domain_is_unknown_without_model_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: type("Prediction", (), {"route": "rag", "db_intent": "unknown", "confidence": 0.99, "label": "rag"})(),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_domains_with_klue_bert",
        lambda _: (),
    )

    decision = chat_orchestrator.decide_chat_route("scholarship deadline")

    assert decision.route == "rag"
    assert decision.rag_domain == "unknown"
    assert decision.rag_domains == ()
    assert decision.rag_ambiguity == "needs_clarification"


def test_rag_domain_pipeline_marks_close_top_scores_as_multi_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_orchestrator.settings, "intent_classifier_model_name", "test-model")
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_with_klue_bert",
        lambda _: type("Prediction", (), {"route": "rag", "db_intent": "unknown", "confidence": 0.99, "label": "rag"})(),
    )
    monkeypatch.setattr(
        chat_orchestrator,
        "classify_rag_domains_with_klue_bert",
        lambda _: (
            RagDomainPrediction(domain="tuition", score=0.74),
            RagDomainPrediction(domain="scholarship", score=0.69),
        ),
    )

    decision = chat_orchestrator.decide_chat_route("scholarship deadline")

    assert decision.route == "rag"
    assert decision.rag_ambiguity == "multi_domain"
    assert decision.rag_confidence is not None
    assert decision.rewritten_queries[0]
