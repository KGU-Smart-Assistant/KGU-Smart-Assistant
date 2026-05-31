from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "KGU Smart Assistant API"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    cors_allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    google_api_key: str
    gemini_model: str = "gemini-3-flash-preview"
    gemini_timeout_ms: int = 120_000
    kakao_rest_api_key: str | None = None
    kakao_map_api_key: str | None = None
    kakao_local_base_url: str = "https://dapi.kakao.com"
    vector_store_mode: str = "embedded"
    vector_store_path: str = ".tmp/chroma"
    vector_store_collection_name: str = "document_chunks"
    vector_store_host: str = "chroma"
    vector_store_port: int = 8000
    translation_api_key: str | None = None
    translation_provider: str = "google"
    google_translation_api_url: str = "https://translation.googleapis.com/language/translate/v2"
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/kgusmart"
    intent_classifier_model_name: str | None = None
    intent_classifier_confidence_threshold: float = 0.7
    intent_classifier_fast_fallback_threshold: float = 0.35
    intent_classifier_device: int = -1
    chat_planner_mode: str = "balanced"
    chat_model_warmup: bool = True
    rag_max_rewritten_queries: int = 5
    rag_domain_classifier_model_name: str | None = None
    rag_domain_classifier_confidence_threshold: float = 0.5
    rag_domain_classifier_top_k: int = 3
    rag_domain_classifier_device: int = -1
    rag_detail_classifier_model_name: str | None = None
    rag_detail_classifier_confidence_threshold: float = 0.45
    rag_detail_classifier_top_k: int = 3
    rag_detail_classifier_device: int = -1
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
