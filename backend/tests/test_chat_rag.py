from app.schemas import SearchResult
from app.services import gemini_service
from app.services import rag_service


def _search_result() -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=0.9,
        text="성적향상장학금 신청 기간은 공지 본문을 확인해야 합니다.",
        title="2026학년도 1학기 성적향상장학금 신청 안내",
        source_url="https://example.com/notice",
    )


def test_get_gemini_response_with_context_uses_search_results(monkeypatch) -> None:
    captured = {}

    def _fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return "공지 기준으로 성적향상장학금 신청 안내를 확인하세요."

    monkeypatch.setattr(gemini_service, "_call_gemini", _fake_call)

    reply = gemini_service.get_gemini_response_with_context(
        "성적향상장학금 알려줘",
        [_search_result()],
    )

    assert "아래 경기대학교 수집 자료만 근거로 답변하세요." in captured["prompt"]
    assert "2026학년도 1학기 성적향상장학금 신청 안내" in captured["prompt"]
    assert "성적향상장학금 신청 기간은 공지 본문을 확인해야 합니다." in captured["prompt"]
    assert "공지 기준으로 성적향상장학금 신청 안내를 확인하세요." in reply
    assert "출처:" in reply
    assert "https://example.com/notice" in reply


def test_get_rag_response_uses_search_results(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "search_documents", lambda query, top_k: [_search_result()])
    monkeypatch.setattr(
        rag_service,
        "get_gemini_response_with_context",
        lambda user_input, results: f"rag answer from {len(results)} result",
    )

    reply = rag_service.get_rag_response("장학금 알려줘")

    assert reply == "rag answer from 1 result"


def test_get_rag_response_returns_no_context_message_when_search_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "search_documents", lambda query, top_k: [])

    reply = rag_service.get_rag_response("없는 내용")

    assert reply == "관련 자료를 찾지 못했습니다."
