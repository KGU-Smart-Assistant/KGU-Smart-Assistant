from app.services.gemini_service import get_gemini_response_with_context
from app.services.search_service import search_documents


def get_rag_response(user_input: str, *, top_k: int = 3) -> str:
    search_results = search_documents(user_input, top_k=top_k)
    if not search_results:
        return "관련 자료를 찾지 못했습니다."
    return get_gemini_response_with_context(user_input, search_results)
