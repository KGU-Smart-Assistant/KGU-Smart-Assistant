"""Business logic services."""

__all__ = [
    "answer_chat",
    "get_gemini_response",
    "get_weather_response",
    "search",
    "search_documents",
]


def __getattr__(name: str):
    if name == "answer_chat":
        from app.services.chat_orchestrator import answer_chat

        return answer_chat

    if name == "get_gemini_response":
        from app.services.gemini_service import get_gemini_response

        return get_gemini_response

    if name == "get_weather_response":
        from app.services.weather_service import get_weather_response

        return get_weather_response

    if name in {"search", "search_documents"}:
        from app.services.search_service import search, search_documents

        exports = {
            "search": search,
            "search_documents": search_documents,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
