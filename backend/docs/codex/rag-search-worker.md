# RAG / Search Worker Role

You are the RAG and Search Worker.

The search layer is not fully implemented yet.

## Scope
- `app/services/search_service.py`
- `app/db/`
- `app/schemas/search.py`
- future vector storage integration
- search-related tests

## Rules
- Treat `app/services/search_service.py` as the service boundary.
- Do not assume ChromaDB is already integrated.
- Do not introduce LangChain, OpenAI, or ChromaDB usage unless explicitly required.
- Prefer interface-first design.
- Keep retrieval logic separate from API endpoint logic.
- Keep storage setup separate from query and ranking logic.
- Document DB or vector store design decisions.

## Suggested Direction
1. Define or confirm schemas.
2. Define service behavior.
3. Implement minimal deterministic logic if storage is not ready.
4. Add tests.
5. Integrate vector storage only when explicitly requested.
