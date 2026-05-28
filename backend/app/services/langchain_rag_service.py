from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
from typing import Callable, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import ConfigDict, Field

from app.crawlers.embedding_pipeline import embed_text
from app.db.vector_store import query_embedded_chunks
from app.schemas.search import SearchResult
from app.services.gemini_service import get_gemini_response
from app.services.search_service import LOW_CONFIDENCE_THRESHOLD, RetrievalPolicy, rerank_candidate_rows, search_documents

SearchFn = Callable[..., List[SearchResult]]
AnswerFn = Callable[[str], str]

DEFAULT_TRACE_PATH = ".tmp/rag_traces.jsonl"
DEFAULT_EXPANDED_QUERY_LIMIT = 5
DEFAULT_COMPRESSION_SENTENCE_LIMIT = 4
DEFAULT_COMPRESSED_CHARS = 900
DEFAULT_RAG_CONFIDENCE_THRESHOLD = LOW_CONFIDENCE_THRESHOLD

RAG_PROMPT = PromptTemplate.from_template(
    """
너는 경기대학교 스마트 어시스턴트다.
아래 검색 근거에 포함된 내용만 사용해서 한국어로 답변해라.
근거가 부족하면 확정적으로 말하지 말고, 확인 가능한 출처 URL을 안내해라.
날짜, 자격, 금액, 부서명, URL은 근거에 없으면 만들지 마라.
영어 답변을 쓰지 말고, URL·고유명사·공식 명칭을 제외한 모든 설명은 자연스러운 한국어로 작성해라.
근거에 없는 항목은 "검색된 자료에서는 확인할 수 없습니다"라고 말해라.
답변 본문에서 근거를 사용할 때는 해당 검색 근거 번호를 [1]처럼 표시해라.
답변 끝에는 반드시 "출처:" 섹션을 포함하고 "[1] 제목: URL" 형식으로 사용한 URL을 적어라.

검색 근거:
{context}

사용자 질문:
{question}

한국어 답변:
""".strip()
)

DOMAIN_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "성적장학": ("성적향상장학금 신청 안내", "성적향상장학금 신청 기간", "장학금 제출 서류"),
    "성적향상장학": ("성적향상장학금 신청 안내", "성적향상장학금 신청 기간", "장학금 제출 서류"),
    "장학": ("장학금 신청 안내", "국가장학금 신청", "교내장학금 신청 기간"),
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
    confidence: float = 0.0
    low_confidence: bool = False


class HybridSearchRetriever(BaseRetriever):
    """LangChain retriever wrapper around the project hybrid search pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = 5
    category: str | None = None
    detail: str | None = None
    rag_domain: str | None = None
    rag_domains: tuple[str, ...] = ()
    rag_detail: str | None = None
    rag_details: tuple[str, ...] = ()
    rag_confidence: float | None = None
    source_scope: str | None = None
    rewritten_queries: tuple[str, ...] = ()
    retrieval_policy: RetrievalPolicy | None = None
    trace_id: str | None = None
    trace_path: str | None = None
    low_confidence_threshold: float = DEFAULT_RAG_CONFIDENCE_THRESHOLD
    expand_queries: bool = True
    compress_documents: bool = True
    search_fn: SearchFn = Field(default=search_documents, exclude=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        queries = self._queries_for(query)
        documents_by_chunk: dict[str, Document] = {}

        for query_index, expanded_query in enumerate(queries):
            for result in _call_search_fn(
                self.search_fn,
                query=expanded_query,
                top_k=self.top_k,
                category=self.category,
                detail=self.detail,
                rag_domain=self.rag_domain or self.category,
                rag_domains=list(self.rag_domains),
                rag_detail=self.rag_detail or self.detail,
                rag_details=list(self.rag_details),
                rag_confidence=self.rag_confidence,
                source_scope=self.source_scope,
                low_confidence_threshold=self.low_confidence_threshold,
                retrieval_policy=self.retrieval_policy,
                trace_id=self.trace_id,
                trace_path=self.trace_path,
            ):
                document = search_result_to_document(result)
                _annotate_document_for_query(
                    document=document,
                    original_query=query,
                    matched_query=expanded_query,
                    query_index=query_index,
                )
                key = str(document.metadata.get("chunk_id") or document.page_content)
                existing = documents_by_chunk.get(key)
                if existing is None:
                    documents_by_chunk[key] = document
                    continue
                _merge_document_metadata(existing, document)

        ranked_documents = sorted(
            documents_by_chunk.values(),
            key=lambda document: float(document.metadata.get("rerank_score") or 0.0),
            reverse=True,
        )
        documents = _filter_documents_for_broad_topic(
            query=query,
            documents=ranked_documents,
            domain=self.rag_domain or self.category,
            source_scope=self.source_scope,
        )[: self.top_k]
        if self.compress_documents:
            return compress_documents_for_query(query, documents)
        return documents

    def _queries_for(self, query: str) -> list[str]:
        queries: list[str] = []
        for candidate in (query, *self.rewritten_queries):
            if candidate and candidate not in queries:
                queries.append(candidate)
        if self.expand_queries:
            for candidate in expand_search_queries(query):
                if candidate not in queries:
                    queries.append(candidate)
        return queries or [query]


class ChromaVectorStoreRetriever(BaseRetriever):
    """LangChain adapter for direct Chroma vector search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = 5
    category: str | None = None
    collection_name: str | None = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        rows = query_embedded_chunks(
            query_embedding=embed_text(query),
            top_k=self.top_k,
            domain=self.category,
            collection_name=self.collection_name,
        )
        ranked_rows = rerank_candidate_rows(rows=rows, query=query, category=self.category)
        return [row_to_document(row) for row in ranked_rows[: self.top_k]]


def answer_with_langchain_rag(
    user_input: str,
    *,
    top_k: int = 5,
    category: str | None = None,
    detail: str | None = None,
    rag_domain: str | None = None,
    rag_domains: list[str] | tuple[str, ...] = (),
    rag_detail: str | None = None,
    rag_details: list[str] | tuple[str, ...] = (),
    rag_confidence: float | None = None,
    source_scope: str | None = None,
    rewritten_queries: list[str] | tuple[str, ...] = (),
    retrieval_policy: RetrievalPolicy | None = None,
    search_fn: SearchFn = search_documents,
    answer_fn: AnswerFn = get_gemini_response,
    retriever: BaseRetriever | None = None,
    trace_id: str | None = None,
    trace_path: str | None = None,
    confidence_threshold: float = DEFAULT_RAG_CONFIDENCE_THRESHOLD,
) -> LangChainRagResult:
    policy = retrieval_policy or RetrievalPolicy.from_inputs(
        category=category,
        detail=detail,
        rag_domain=rag_domain,
        rag_domains=tuple(rag_domains),
        rag_detail=rag_detail,
        rag_details=tuple(rag_details),
        rag_confidence=rag_confidence,
        source_scope=source_scope,
        rewritten_queries=tuple(rewritten_queries),
        trace_id=trace_id,
        trace_path=trace_path,
    )
    effective_retriever = retriever or HybridSearchRetriever(
        top_k=top_k,
        category=category or rag_domain,
        detail=detail or rag_detail,
        rag_domain=rag_domain or category,
        rag_domains=tuple(rag_domains),
        rag_detail=rag_detail or detail,
        rag_details=tuple(rag_details),
        rag_confidence=rag_confidence,
        source_scope=source_scope,
        rewritten_queries=tuple(rewritten_queries),
        retrieval_policy=policy,
        trace_id=trace_id,
        trace_path=trace_path,
        low_confidence_threshold=confidence_threshold,
        search_fn=search_fn,
    )
    chain = build_rag_chain(retriever=effective_retriever, answer_fn=answer_fn, confidence_threshold=confidence_threshold)
    expanded_queries = expand_search_queries(user_input)
    result = chain.invoke(
        user_input,
        config={
            "run_name": "kgu_langchain_rag",
            "tags": ["kgu", "rag", "langchain"],
            "metadata": {
                "trace_id": trace_id,
                "expanded_queries": expanded_queries,
                "retriever": effective_retriever.__class__.__name__,
                "category": category or rag_domain,
                "detail": detail or rag_detail,
                "rag_domain": rag_domain or category,
                "rag_domains": list(rag_domains),
                "rag_detail": rag_detail or detail,
                "rag_details": list(rag_details),
                "rag_confidence": rag_confidence,
                "source_scope": source_scope,
                "retrieval_policy": policy.to_trace_dict(),
            },
        },
    )
    traced = LangChainRagResult(
        reply=result.reply,
        documents=result.documents,
        context=result.context,
        expanded_queries=expanded_queries,
        trace_id=trace_id or policy.trace_id,
        confidence=result.confidence,
        low_confidence=result.low_confidence,
    )
    trace_rag_result(user_input, traced, trace_path=trace_path)
    return traced


def build_rag_chain(
    *,
    retriever: BaseRetriever,
    answer_fn: AnswerFn = get_gemini_response,
    confidence_threshold: float = DEFAULT_RAG_CONFIDENCE_THRESHOLD,
):
    return (
        RunnableParallel({"question": RunnablePassthrough(), "documents": retriever})
        | RunnableLambda(_prepare_prompt_payload)
        | RunnableLambda(lambda payload: _generate_answer(payload, answer_fn=answer_fn, confidence_threshold=confidence_threshold))
    )


def expand_search_queries(query: str, *, max_queries: int = DEFAULT_EXPANDED_QUERY_LIMIT) -> list[str]:
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


def _filter_documents_for_broad_topic(
    *,
    query: str,
    documents: list[Document],
    domain: str | None,
    source_scope: str | None,
) -> list[Document]:
    if domain != "graduation" or source_scope == "department":
        return _dedupe_documents_by_source_url(documents)

    university_documents = [
        document for document in documents if _is_university_wide_document(document)
    ]
    if not university_documents:
        return _dedupe_documents_by_source_url(documents)

    title_documents = [
        document
        for document in university_documents
        if _document_title_matches_topic(query=query, document=document)
    ]
    if title_documents:
        return _dedupe_documents_by_source_url(title_documents)

    topic_documents = [
        document
        for document in university_documents
        if _document_matches_topic(query=query, document=document)
    ]
    return _dedupe_documents_by_source_url(topic_documents or university_documents)


def _is_university_wide_document(document: Document) -> bool:
    metadata = document.metadata
    department = str(metadata.get("department") or "").strip().casefold()
    source_url = str(metadata.get("source_url") or "").casefold()
    return department == "university" or "/www/contents.do" in source_url


def _document_matches_topic(*, query: str, document: Document) -> bool:
    terms = _query_topic_terms(query)
    if not terms:
        return True
    metadata = document.metadata
    title = _normalize_text(str(metadata.get("title") or ""))
    text = _normalize_text(document.page_content)
    haystack = f"{title} {text[:1200]}"
    return any(term in haystack for term in terms)


def _document_title_matches_topic(*, query: str, document: Document) -> bool:
    terms = _query_topic_terms(query)
    if not terms:
        return False
    title = _normalize_text(str(document.metadata.get("title") or ""))
    return any(term in title for term in terms)


def _query_topic_terms(query: str) -> list[str]:
    normalized = _normalize_text(query)
    stopwords = {
        "알려줘",
        "알려주세요",
        "궁금해",
        "뭐야",
        "어떻게",
        "어디서",
        "확인",
        "보고",
        "싶어",
    }
    terms = [token for token in _tokenize(normalized) if token not in stopwords]
    compounds: list[str] = []
    if "졸업" in normalized and "요건" in normalized:
        compounds.extend(["졸업요건", "졸업 요건", "졸업안내"])
    if "신청" in normalized and "기간" in normalized:
        compounds.extend(["신청기간", "신청 기간"])
    if "제출" in normalized and "서류" in normalized:
        compounds.extend(["제출서류", "제출 서류"])
    return list(dict.fromkeys([*compounds, *terms]))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _dedupe_documents_by_source_url(documents: list[Document]) -> list[Document]:
    selected: list[Document] = []
    seen_urls: set[str] = set()
    for document in documents:
        source_url = str(document.metadata.get("source_url") or "").strip()
        key = source_url or str(document.metadata.get("chunk_id") or document.page_content)
        if key in seen_urls:
            continue
        seen_urls.add(key)
        selected.append(document)
    return selected


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
            (_sentence_overlap_score(sentence, tokens), index, sentence)
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


def _call_search_fn(search_fn: SearchFn, **kwargs) -> list[SearchResult]:
    try:
        parameters = inspect.signature(search_fn).parameters
    except (TypeError, ValueError):
        return search_fn(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        supported = kwargs
    else:
        supported = {key: value for key, value in kwargs.items() if key in parameters}
    return search_fn(**supported)


def search_result_to_document(result: SearchResult) -> Document:
    metadata = {
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "title": result.title,
        "source_url": result.source_url,
        "score": result.score,
        "retrieval_score": result.score,
    }
    result_domain = result.domain or result.category
    if result_domain:
        metadata["domain"] = result_domain
    if result.department:
        metadata["department"] = result.department
    if result.published_at:
        metadata["published_at"] = result.published_at
    if result.score_breakdown:
        metadata["score_breakdown"] = result.score_breakdown
        metadata["confidence"] = result.score_breakdown.get("confidence", result.score)
        metadata["fallback_used"] = bool(result.score_breakdown.get("fallback_used"))
        metadata["parent_expanded"] = bool(result.score_breakdown.get("parent_expanded"))
    else:
        metadata["confidence"] = result.score
    return Document(page_content=result.text, metadata=metadata)


def row_to_document(row: dict) -> Document:
    metadata = {
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "title": row.get("title", ""),
        "source_url": row.get("source_url", ""),
        "score": row.get("score", 0.0),
        "retrieval_score": row.get("score", 0.0),
        "confidence": row.get("score_breakdown", {}).get("confidence", row.get("score", 0.0)),
    }
    for key in ("domain", "category", "department", "published_at", "score_breakdown", "parent_expanded"):
        if row.get(key):
            metadata[key] = row[key]
    return Document(page_content=row.get("text", ""), metadata=metadata)


def format_documents(documents: list[Document]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        source_number = int(metadata.get("source_number") or index)
        lines = [
            f"[{source_number}] {metadata.get('title', 'Untitled')}",
            f"source_url: {metadata.get('source_url')}",
            f"score: {metadata.get('score')}",
            f"rerank_score: {metadata.get('rerank_score', metadata.get('score'))}",
            f"confidence: {metadata.get('confidence', metadata.get('score'))}",
        ]
        if metadata.get("domain"):
            lines.append(f"domain: {metadata['domain']}")
        if metadata.get("department"):
            lines.append(f"department: {metadata['department']}")
        if metadata.get("published_at"):
            lines.append(f"published_at: {metadata['published_at']}")
        if metadata.get("matched_queries"):
            lines.append(f"matched_queries: {', '.join(metadata['matched_queries'])}")
        if metadata.get("parent_expanded"):
            lines.append("parent_expanded: true")
        lines.append(document.page_content)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def trace_rag_result(query: str, result: LangChainRagResult, *, trace_path: str | None = None) -> None:
    path = trace_path or os.getenv("RAG_TRACE_PATH") or DEFAULT_TRACE_PATH
    if os.getenv("RAG_TRACE_ENABLED", "true").casefold() in {"0", "false", "no"}:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": result.trace_id,
        "query": query,
        "expanded_queries": result.expanded_queries,
        "document_count": len(result.documents),
        "confidence": result.confidence,
        "low_confidence": result.low_confidence,
        "documents": [
            {
                "source_number": document.metadata.get("source_number"),
                "chunk_id": document.metadata.get("chunk_id"),
                "doc_id": document.metadata.get("doc_id"),
                "title": document.metadata.get("title"),
                "source_url": document.metadata.get("source_url"),
                "domain": document.metadata.get("domain"),
                "department": document.metadata.get("department"),
                "published_at": document.metadata.get("published_at"),
                "score": document.metadata.get("score"),
                "retrieval_score": document.metadata.get("retrieval_score"),
                "rerank_score": document.metadata.get("rerank_score"),
                "confidence": document.metadata.get("confidence"),
                "query_relevance": document.metadata.get("query_relevance"),
                "best_query_index": document.metadata.get("best_query_index"),
                "matched_queries": document.metadata.get("matched_queries", []),
                "score_breakdown": document.metadata.get("score_breakdown", {}),
                "compressed": document.metadata.get("compressed", False),
                "fallback_used": document.metadata.get("fallback_used", False),
                "parent_expanded": document.metadata.get("parent_expanded", False),
            }
            for document in result.documents
        ],
        "reply_preview": result.reply[:300],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _prepare_prompt_payload(payload: dict) -> dict:
    documents = _with_source_numbers(payload["documents"])
    question = payload["question"]
    context = format_documents(documents)
    prompt = RAG_PROMPT.format(context=context, question=question)
    return {"question": question, "documents": documents, "context": context, "prompt": prompt}


def _generate_answer(payload: dict, *, answer_fn: AnswerFn, confidence_threshold: float) -> LangChainRagResult:
    documents = payload["documents"]
    if not documents:
        return LangChainRagResult(
            reply="관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요.",
            documents=[],
            context="",
            expanded_queries=[],
            confidence=0.0,
            low_confidence=True,
        )

    confidence = max(_document_confidence(document) for document in documents)
    if confidence < confidence_threshold:
        return LangChainRagResult(
            reply=_low_confidence_reply(documents),
            documents=documents,
            context=payload["context"],
            expanded_queries=[],
            confidence=round(confidence, 6),
            low_confidence=True,
        )

    reply = answer_fn(payload["prompt"])
    reply = _control_answer_language(payload, reply, answer_fn=answer_fn)
    reply = ensure_source_urls(reply, documents)
    return LangChainRagResult(
        reply=reply,
        documents=documents,
        context=payload["context"],
        expanded_queries=[],
        confidence=round(confidence, 6),
        low_confidence=False,
    )


def ensure_source_urls(reply: str, documents: list[Document]) -> str:
    source_lines = _numbered_source_lines(documents)
    urls = [url for _number, _title, url in source_lines]
    if not urls:
        return reply
    missing_urls = [url for url in urls if url not in reply]
    if not missing_urls and "출처" in reply:
        return reply
    formatted_sources = ["", "", "출처:"] + [
        f"- [{number}] {title}: {url}" for number, title, url in source_lines
    ]
    return reply.rstrip() + "\n".join(formatted_sources)


def _control_answer_language(payload: dict, reply: str, *, answer_fn: AnswerFn) -> str:
    """Keep RAG answers in Korean even when the model drifts into English."""

    normalized_reply = reply.strip()
    if not normalized_reply:
        return _source_only_reply(payload["documents"])
    if not _needs_korean_rewrite(normalized_reply):
        return normalized_reply

    rewrite_prompt = _korean_rewrite_prompt(
        answer=normalized_reply,
        question=str(payload["question"]),
        context=str(payload["context"]),
    )
    rewritten = answer_fn(rewrite_prompt).strip()
    if rewritten and not _needs_korean_rewrite(rewritten):
        return rewritten
    return _source_only_reply(payload["documents"])


def _needs_korean_rewrite(reply: str) -> bool:
    text_without_urls = re.sub(r"https?://\S+", "", reply)
    hangul_count = len(re.findall(r"[가-힣]", text_without_urls))
    latin_count = len(re.findall(r"[A-Za-z]", text_without_urls))
    if hangul_count == 0:
        return True
    return latin_count > hangul_count * 2


def _korean_rewrite_prompt(*, answer: str, question: str, context: str) -> str:
    return f"""
다음 답변을 경기대학교 스마트 어시스턴트의 최종 답변으로 다시 작성하세요.

규칙:
- 한국어로만 작성하세요. URL, 고유명사, 공식 영문 명칭은 유지할 수 있습니다.
- 아래 검색 근거에 없는 날짜, 자격, 금액, 부서명, URL은 추가하지 마세요.
- 근거에 없는 항목은 "검색된 자료에서는 확인할 수 없습니다"라고 쓰세요.
- 답변 끝에는 "출처:" 섹션과 검색 근거의 URL만 포함하세요.

사용자 질문:
{question}

검색 근거:
{context}

다시 작성할 답변:
{answer}

한국어 최종 답변:
""".strip()


def _source_only_reply(documents: list[Document]) -> str:
    lines = [
        "검색된 자료를 바탕으로 한국어 답변을 생성하지 못했습니다.",
        "아래 출처에서 직접 확인해 주세요.",
        "",
        "출처:",
    ]
    for number, title, url in _numbered_source_lines(documents):
        lines.append(f"- [{number}] {title}: {url}")
    return "\n".join(lines)


def _low_confidence_reply(documents: list[Document]) -> str:
    lines = [
        "검색된 문서의 관련도가 낮아서 확정 답변은 제한합니다.",
        "아래 출처에서 직접 확인해 주세요.",
        "",
        "출처:",
    ]
    for number, title, url in _numbered_source_lines(documents):
        lines.append(f"- [{number}] {title}: {url}")
    return "\n".join(lines)


def _with_source_numbers(documents: list[Document]) -> list[Document]:
    numbered: list[Document] = []
    for index, document in enumerate(documents, start=1):
        metadata = dict(document.metadata)
        metadata["source_number"] = int(metadata.get("source_number") or index)
        numbered.append(Document(page_content=document.page_content, metadata=metadata))
    return numbered


def _numbered_source_lines(documents: list[Document]) -> list[tuple[int, str, str]]:
    source_lines: list[tuple[int, str, str]] = []
    seen_urls: set[str] = set()
    for index, document in enumerate(documents, start=1):
        url = str(document.metadata.get("source_url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        number = int(document.metadata.get("source_number") or index)
        title = str(document.metadata.get("title") or "문서")
        source_lines.append((number, title, url))
    return source_lines


def _annotate_document_for_query(*, document: Document, original_query: str, matched_query: str, query_index: int) -> None:
    metadata = document.metadata
    metadata.setdefault("matched_queries", [])
    metadata["matched_queries"].append(matched_query)
    metadata["best_query_index"] = min(int(metadata.get("best_query_index", query_index)), query_index)
    metadata["query_relevance"] = _document_query_relevance(original_query, document)
    metadata["rerank_score"] = _document_rerank_score(
        retrieval_score=float(metadata.get("score") or 0.0),
        query_relevance=float(metadata["query_relevance"]),
        query_index=query_index,
    )


def _merge_document_metadata(existing: Document, candidate: Document) -> None:
    existing_queries = list(existing.metadata.get("matched_queries", []))
    for query in candidate.metadata.get("matched_queries", []):
        if query not in existing_queries:
            existing_queries.append(query)
    existing.metadata["matched_queries"] = existing_queries

    if float(candidate.metadata.get("rerank_score") or 0.0) > float(existing.metadata.get("rerank_score") or 0.0):
        preserved_queries = existing.metadata["matched_queries"]
        existing.page_content = candidate.page_content
        existing.metadata.update(candidate.metadata)
        existing.metadata["matched_queries"] = preserved_queries


def _document_rerank_score(*, retrieval_score: float, query_relevance: float, query_index: int) -> float:
    expansion_penalty = min(query_index * 0.08, 0.24)
    original_bonus = 0.08 if query_index == 0 else 0.0
    score = retrieval_score * 0.55 + query_relevance * 0.37 + original_bonus - expansion_penalty
    return round(max(min(score, 1.0), 0.0), 6)


def _document_query_relevance(query: str, document: Document) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    title = str(document.metadata.get("title") or "")
    title_text = title.casefold()
    body_text = document.page_content.casefold()
    matched = 0.0
    for token in tokens:
        if token in title_text:
            matched += 1.5
        elif token in body_text:
            matched += 0.8
    exact_bonus = _long_token_bonus(tokens=tokens, title=title_text, text=body_text)
    return min((matched / (len(tokens) * 1.5)) + exact_bonus, 1.0)


def _long_token_bonus(*, tokens: list[str], title: str, text: str) -> float:
    bonus = 0.0
    for token in tokens:
        if len(token) < 4:
            continue
        if token in title:
            bonus += 0.08
        elif token in text:
            bonus += 0.04
    return min(bonus, 0.25)


def _document_confidence(document: Document) -> float:
    metadata = document.metadata
    if metadata.get("confidence") is not None:
        return float(metadata.get("confidence") or 0.0)
    breakdown = metadata.get("score_breakdown") or {}
    if breakdown.get("confidence") is not None:
        return float(breakdown.get("confidence") or 0.0)
    return float(metadata.get("score") or 0.0)


def _split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+", text) if sentence.strip()]
    return sentences or [text.strip()]


def _sentence_overlap_score(sentence: str, query_tokens: set[str]) -> float:
    sentence_tokens = set(_tokenize(sentence))
    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & query_tokens) / len(query_tokens)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"[0-9A-Za-z가-힣]+", text) if len(token) >= 2]
