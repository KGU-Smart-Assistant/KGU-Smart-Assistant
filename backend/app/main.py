import logging
from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.seed import seed_contacts_from_json, seed_places_from_json
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.on_event("startup")
    def _seed_db() -> None:
        with SessionLocal() as db:
            seed_places_from_json(db)
            seed_contacts_from_json(db)
        if settings.chat_model_warmup:
            Thread(target=_warm_chat_models, name="chat-model-warmup", daemon=True).start()

    return app


app = create_app()


def _warm_chat_models() -> None:
    try:
        from app.services.klue_bert_intent_classifier import classify_with_klue_bert
        from app.services.rag_detail_classifier import classify_rag_details_with_klue_bert
        from app.services.rag_domain_classifier import classify_rag_domains_with_klue_bert

        classify_with_klue_bert("장학금 신청 기간 알려줘")
        classify_rag_domains_with_klue_bert("장학금 신청 기간 알려줘")
        classify_rag_details_with_klue_bert("장학금 신청 기간 알려줘")
    except Exception:
        logger.exception("Chat model warmup failed")
