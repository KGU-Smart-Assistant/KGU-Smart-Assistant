from fastapi import APIRouter

from app.schemas import SearchRequest, SearchResponse
from app.services.search_service import search


router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search_document_chunks(request: SearchRequest) -> SearchResponse:
    return search(
        query=request.query,
        top_k=request.top_k,
        category=request.category,
    )
