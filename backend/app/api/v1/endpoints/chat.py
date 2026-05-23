from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource, RagIntentScore
from app.services.chat_orchestrator import answer_chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest, db: Session = Depends(get_db)):
    result = answer_chat(request.message, db)

    return ChatResponse(
        reply=result.reply,
        intent=result.intent,
        route=result.route,
        sources=[ChatSource(**source.__dict__) for source in result.sources],
        rag_domain=result.rag_domain,
        rag_domains=list(result.rag_domains),
        rag_detail=result.rag_detail,
        rag_details=list(result.rag_details),
        source_scope=result.source_scope,
        rag_confidence=result.rag_confidence,
        rag_ambiguity=result.rag_ambiguity,
        rewritten_queries=list(result.rewritten_queries),
        matched_keywords=list(result.matched_keywords),
        intent_scores=[
            RagIntentScore(
                domain=score.domain,
                score=score.score,
                matched_keywords=list(score.matched_keywords),
            )
            for score in result.intent_scores
        ],
        answer_status=result.answer_status,
        unverified=list(result.unverified),
    )
