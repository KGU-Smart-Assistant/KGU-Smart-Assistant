from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
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
    )
