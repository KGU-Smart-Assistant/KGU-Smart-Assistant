from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Callable, Iterable, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import ConfigDict, Field

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas.search import SearchResult
from app.services.gemini_service import get_gemini_response
from app.services.search_service import rerank_candidate_rows, search_documents

SearchFn = Callable[..., List[SearchResult]]
AnswerFn = Callable[[str], str]

DEFAULT_TRACE_PATH = ".tmp/rag_traces.jsonl"
DEFAULT_EXPANDED_QUERY_LIMIT = 5
DEFAULT_COMPRESSION_SENTENCE_LIMIT = 4
DEFAULT_COMPRESSED_CHARS = 900

RAG_PROMPT = PromptTemplate.from_template(
    """
You are a Korean university assistant for Kyonggi University.
Answer using only the retrieved context below.
If the context is insufficient, say that the available document evidence is insufficient.
Do not invent dates, eligibility rules, amounts, office names, or URLs.

Retrieved context:
{context}

User question:
{question}

Answer in Korean:
""".strip()
)

DOMAIN_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "성적장학": ("성적우수장학금 신청 안내", "교내장학금 신청 기간", "장학금 제출 서류"),
    "장학": ("장학금 신청 안내", "국가장학금 신청", "교내장학금"),
    "졸업": ("졸업요건", "졸업학점", "졸업인증", "전공학점"),
    "수강": ("수강신청", "수강 정정", "수강취소", "수강신청 기간"),
    "학사": ("학사일정", "개강", "종강", "시험 기간"),
    "취업": ("취업지원", "현장실습", "인턴", "채용", "비교과"),
    "자료": ("자료실", "첨부파일", "신청서", "제출 서류"),
    "기숙사": ("기숙사 신청", "생활관 안내", "학생생활"),
}


@dataclass(frozen=True)
class LangChainRagResult:
    reply: str
    documents: list[Document]
    context: str
    expanded_queries: list[str]
    trace_id: str | None = None


class HybridSearchRetriever(BaseRetriever):
    """LangChain retriever wrapper around the project hybrid search pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = 5
    category: str | None = None
    expand_queries: bool = True
    compress_documents: bool = True
    search_fn: SearchFn = Field(default=search_documents, exclude=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        queries = expand_search_queries(query) if self.expand_queries else [query]
        documents_by_chunk: dict[str, Document] = {}
        for expanded_query in queries:
            for result in self.search_fn(
                query=expanded_query,
                top_k=self.top_k,
                category=self.category,
            ):
                document = search_result_to_document(result)
                document.metadata.setdefault("matched_queries", [])
                document.metadata["matched_queries"].append(expanded_query)
                key = str(document.metadata.get("chunk_id") or document.page_content)
                existing = documents_by_chunk.get(key)
                if existing is None or float(document.metadata.get("score") or 0) > float(
                    existing.metadata.get("score") or 0
                ):
                    documents_by_chunk[key] = document

        documents = sorted(
            documents_by_chunk.values(),
            key=lambda document: float(document.metadata.get("score") or 0.0),
            reverse=True,
        )[: self.top_k]
        if self.compress_documents:
            return compress_documents_for_query(query, documents)
        return documents


class ChromaVectorStoreRetriever(BaseRetriever):
    """LangChain adapter for the configured Chroma vector store."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = 5
    category: str | None = None
    collection_name: str | None = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        rows = query_embedded_chunks(
            query_embedding=embed_text(query),
            top_k=self.top_k,
            category=self.category,
            collection_name=self.collection_name,
        )
        ranked_rows = rerank_candidate_rows(rows=rows, query=query, category=self.category)
        return [row_to_document(row) for row in ranked_rows[: self.top_k]]


def answer_with_langchain_rag(
    user_input: str,
    *,
    top_k: int = 5,
    category: str | None = None,
    search_fn: SearchFn = search_documents,
    answer_fn: AnswerFn = get_gemini_response,
    retriever: BaseRetriever | None = None,
    trace_id: str | None = None,
    trace_path: str | None = None,
) -> LangChainRagResult:
    effective_retriever = retriever or HybridSearchRetriever(
        top_k=top_k,
        category=category,
        search_fn=search_fn,
    )
    chain = build_rag_chain(retriever=effective_retriever, answer_fn=answer_fn)
    expanded_queries = expand_search_queries(user_input)
    run_config = {
        "run_name": "kgu_langchain_rag",
        "tags": ["kgu", "rag", "langchain"],
        "metadata": {
            "trace_id": trace_id,
            "expanded_queries": expanded_queries,
            "retriever": effective_retriever.__class__.__name__,
        },
    }
    result = chain.invoke(user_input, config=run_config)
    traced = LangChainRagResult(
        reply=result.reply,
        documents=result.documents,
        context=result.context,
        expanded_queries=expanded_queries,
        trace_id=trace_id,
    )
    trace_rag_result(user_input, traced, trace_path=trace_path)
    return traced


def build_rag_chain(
    *,
    retriever: BaseRetriever,
    answer_fn: AnswerFn = get_gemini_response,
):
    return (
        RunnableParallel(
            {
                "question": RunnablePassthrough(),
                "documents": retriever,
            }
        )
        | RunnableLambda(_prepare_prompt_payload)
        | RunnableLambda(lambda payload: _generate_answer(payload, answer_fn=answer_fn))
    )


def expand_search_queries(
    query: str,
    *,
    max_queries: int = DEFAULT_EXPANDED_QUERY_LIMIT,
) -> list[str]:
    normalized = query.casefold()
    expanded = [query]
    for keyword, alternatives in DOMAIN_QUERY_EXPANSIONS.items():
        if keyword.casefold() not in normalized:
            continue
        for alternative in alternatives:
            if alternative not in expanded:
                expanded.append(alternative)
            if len(expanded) >= max_queries:
                return expanded
    return expanded


def compress_documents_for_query(
    query: str,
    documents: list[Document],
    *,
    sentence_limit: int = DEFAULT_COMPRESSION_SENTENCE_LIMIT,
    max_chars: int = DEFAULT_COMPRESSED_CHARS,
) -> list[Document]:
    tokens = set(_tokenize(query))
    if not tokens:
        return documents

    compressed = []
    for document in documents:
        sentences = _split_sentences(document.page_content)
        scored_sentences = [
            (
                _sentence_overlap_score(sentence, tokens),
                index,
                sentence,
            )
            for index, sentence in enumerate(sentences)
        ]
        selected = [
            sentence
            for score, _index, sentence in sorted(scored_sentences, reverse=True)
            if score > 0
        ][:sentence_limit]
        if not selected:
            selected = sentences[:sentence_limit]
        compressed_text = " ".join(selected).strip()[:max_chars]
        metadata = dict(document.metadata)
        metadata["compressed"] = True
        metadata["original_length"] = len(document.page_content)
        metadata["compressed_length"] = len(compressed_text)
        compressed.append(Document(page_content=compressed_text, metadata=metadata))
    return compressed


def search_result_to_document(result: SearchResult) -> Document:
    metadata = {
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "title": result.title,
        "source_url": result.source_url,
        "score": result.score,
    }
    if result.category:
        metadata["category"] = result.category
    if result.department:
        metadata["department"] = result.department
    if result.published_at:
        metadata["published_at"] = result.published_at
    if result.score_breakdown:
        metadata["score_breakdown"] = result.score_breakdown
    return Document(page_content=result.text, metadata=metadata)


def row_to_document(row: dict) -> Document:
    metadata = {
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "title": row.get("title", ""),
        "source_url": row.get("source_url", ""),
        "score": row.get("score", 0.0),
    }
    for key in ("category", "department", "published_at", "score_breakdown"):
        if row.get(key):
            metadata[key] = row[key]
    return Document(page_content=row.get("text", ""), metadata=metadata)


def _prepare_prompt_payload(payload: dict) -> dict:
    documents = payload["documents"]
    question = payload["question"]
    context = format_documents(documents)
    prompt = RAG_PROMPT.format(context=context, question=question)
    return {
        "question": question,
        "documents": documents,
        "context": context,
        "prompt": prompt,
    }


def _generate_answer(payload: dict, *, answer_fn: AnswerFn) -> LangChainRagResult:
    documents = payload["documents"]
    if not documents:
        return LangChainRagResult(
            reply="관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요.",
            documents=[],
            context="",
            expanded_queries=[],
        )
    return LangChainRagResult(
        reply=answer_fn(payload["prompt"]),
        documents=documents,
        context=payload["context"],
        expanded_queries=[],
    )


def format_documents(documents: list[Document]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        lines = [
            f"[{index}] {metadata.get('title', 'Untitled')}",
            f"source_url: {metadata.get('source_url')}",
            f"score: {metadata.get('score')}",
        ]
        if metadata.get("category"):
            lines.append(f"category: {metadata['category']}")
        if metadata.get("department"):
            lines.append(f"department: {metadata['department']}")
        if metadata.get("published_at"):
            lines.append(f"published_at: {metadata['published_at']}")
        if metadata.get("matched_queries"):
            lines.append(f"matched_queries: {', '.join(metadata['matched_queries'])}")
        lines.append(document.page_content)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def trace_rag_result(
    query: str,
    result: LangChainRagResult,
    *,
    trace_path: str | None = None,
) -> None:
    path = trace_path or os.getenv("RAG_TRACE_PATH") or DEFAULT_TRACE_PATH
    if os.getenv("RAG_TRACE_ENABLED", "true").casefold() in {"0", "false", "no"}:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": result.trace_id,
        "query": query,
        "expanded_queries": result.expanded_queries,
        "document_count": len(result.documents),
        "documents": [
            {
                "chunk_id": document.metadata.get("chunk_id"),
                "title": document.metadata.get("title"),
                "source_url": document.metadata.get("source_url"),
                "score": document.metadata.get("score"),
                "matched_queries": document.metadata.get("matched_queries", []),
                "compressed": document.metadata.get("compressed", False),
            }
            for document in result.documents
        ],
        "reply_preview": result.reply[:300],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？\n])\s+", text) if sentence.strip()]
    return sentences or [text.strip()]


def _sentence_overlap_score(sentence: str, query_tokens: set[str]) -> float:
    sentence_tokens = set(_tokenize(sentence))
    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & query_tokens) / len(query_tokens)


def _tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣]+", text)
        if len(token) >= 2
    ]
