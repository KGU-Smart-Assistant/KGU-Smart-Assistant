from typing import List

from app.core.config import settings
from app.schemas import DocumentChunk, EmbeddedChunk

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def embed_text(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """Embed a single text string with Gemini embeddings."""
    client = _create_client()
    response = client.models.embed_content(
        model=model,
        contents=text,
    )

    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise RuntimeError("Embedding response did not contain any vectors.")

    values = getattr(embeddings[0], "values", None)
    if not values:
        raise RuntimeError("Embedding vector is empty.")

    return list(values)


def embed_texts(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[List[float]]:
    """Embed multiple texts one by one and return vectors in the same order."""
    return [embed_text(text=text, model=model) for text in texts]


def embed_chunks(
    chunks: List[DocumentChunk],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[EmbeddedChunk]:
    """Attach embeddings to document chunks for vector storage."""
    embedded_chunks: List[EmbeddedChunk] = []

    for chunk in chunks:
        embedded_chunks.append(
            EmbeddedChunk(
                **chunk.model_dump(),
                embedding=embed_text(text=chunk.text, model=model),
                embedding_model=model,
            )
        )

    return embedded_chunks


def _create_client():
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Install it before using embedding services."
        ) from exc

    return genai.Client(api_key=settings.google_api_key)
