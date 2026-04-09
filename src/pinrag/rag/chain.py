"""RAG chain: retrieval, context formatting, prompt, LLM, and response with citations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langsmith import traceable

from pinrag.config import (
    get_collection_name,
    get_multi_query_count,
    get_rerank_retrieve_k,
    get_rerank_top_n,
    get_retrieve_k,
    get_use_multi_query,
    get_use_rerank,
)
from pinrag.rag.formatting import format_docs, format_sources
from pinrag.rag.multiquery import wrap_retriever_with_multiquery
from pinrag.rag.prompts import get_rag_prompt
from pinrag.rag.query_preprocess import preprocess_query
from pinrag.rag.rerank import is_rerank_available, wrap_retriever_with_rerank
from pinrag.vectorstore.chroma_client import DEFAULT_PERSIST_DIR
from pinrag.vectorstore.retriever import create_retriever

PathLike = str | Path
logger = logging.getLogger(__name__)

_MSG_LLM_RATE = "Answer generation failed: rate limit exceeded. Please try again later."
_MSG_LLM_TIMEOUT = "Answer generation failed: request timed out. Please try again."
_MSG_LLM_CONTEXT = (
    "Answer generation failed: the prompt or retrieved context is too large "
    "for the model. Try a narrower query or fewer documents."
)
_MSG_LLM_GENERIC = "Answer generation failed. Please try again."


def _user_facing_llm_error_message(exc: BaseException) -> str:
    """Map provider/transport errors to stable user-facing strings."""
    if isinstance(exc, ContextOverflowError):
        return _MSG_LLM_CONTEXT
    if isinstance(exc, TimeoutError):
        return _MSG_LLM_TIMEOUT

    try:
        import httpx
    except ImportError:
        pass
    else:
        if isinstance(exc, httpx.TimeoutException):
            return _MSG_LLM_TIMEOUT

    try:
        import openai as openai_sdk
    except ImportError:
        pass
    else:
        if isinstance(exc, openai_sdk.RateLimitError):
            return _MSG_LLM_RATE
        if isinstance(exc, openai_sdk.APITimeoutError):
            return _MSG_LLM_TIMEOUT

    try:
        import anthropic as anthropic_sdk
    except ImportError:
        pass
    else:
        if isinstance(exc, anthropic_sdk.RateLimitError):
            return _MSG_LLM_RATE
        if isinstance(exc, anthropic_sdk.APITimeoutError):
            return _MSG_LLM_TIMEOUT

    err = str(exc).lower()
    if "rate" in err and "limit" in err:
        return _MSG_LLM_RATE
    if "timeout" in err:
        return _MSG_LLM_TIMEOUT
    if "context" in err and ("length" in err or "token" in err or "maximum" in err):
        return _MSG_LLM_CONTEXT
    return _MSG_LLM_GENERIC


@dataclass
class RAGResult:
    """Result of running the RAG chain: answer text and source citations."""

    answer: str
    sources: list[dict[str, str | int]]
    documents: list[Document] = field(default_factory=list)


def _docs_to_langsmith_output(docs: list[Document]) -> list[dict]:
    """Convert Documents for LangSmith retriever trace rendering."""
    return [
        {"page_content": d.page_content, "type": "Document", "metadata": d.metadata}
        for d in docs
    ]


@traceable(
    name="retrieve",
    run_type="retriever",
    process_outputs=lambda out: _docs_to_langsmith_output(out) if out else [],
)  # type: ignore[call-overload]
def _retrieve(retriever: BaseRetriever, query: str) -> list[Document]:
    """Retrieve documents; LangSmith logs output in retriever format for special rendering."""
    return retriever.invoke(query)


def _apply_post_retrieval_doc_limit(
    docs: list[Document], truncate_k: int | None
) -> list[Document]:
    """Cap merged multi-query union before generation when reranking is off."""
    if truncate_k is None or truncate_k <= 0:
        return docs
    return docs[:truncate_k]


def _build_standard_retriever(
    *,
    llm: BaseChatModel,
    use_multi_query: bool,
    effective_k: int,
    persist_directory: PathLike,
    collection_name: str,
    embedding: Embeddings | None,
    document_id: str | None,
    page_min: int | None,
    page_max: int | None,
    tag: str | None,
    document_type: str | None,
) -> tuple[BaseRetriever, bool, int | None]:
    """Build standard retriever and optionally wrap with multi-query.

    Returns (retriever, was_multi_query_wrapped_without_rerank, truncate_k).
    """
    base_retriever = create_retriever(
        k=effective_k,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
        document_id=document_id,
        page_min=page_min,
        page_max=page_max,
        tag=tag,
        document_type=document_type,
    )
    if use_multi_query:
        return (
            wrap_retriever_with_multiquery(
                base_retriever,
                llm,
                num_queries=get_multi_query_count(),
            ),
            True,
            effective_k,
        )
    return base_retriever, False, None


def build_retriever(
    llm: BaseChatModel,
    *,
    k: int | None = None,
    use_rerank: bool | None = None,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    document_id: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    tag: str | None = None,
    document_type: str | None = None,
) -> tuple[BaseRetriever, int | None]:
    """Build a retriever and optional post-retrieval doc cap.

    Returns ``(retriever, truncate_k)``. When multi-query is enabled and reranking
    is not used, ``truncate_k`` is the effective ``k``; callers must slice
    retrieved documents to at most ``truncate_k`` before generation. Otherwise
    ``truncate_k`` is ``None``.
    """
    if collection_name is None:
        collection_name = get_collection_name()
    use_rerank = use_rerank if use_rerank is not None else get_use_rerank()
    use_multi_query = get_use_multi_query()

    if use_rerank:
        available, err = is_rerank_available()
        if available:
            base_k = k if k is not None else get_rerank_retrieve_k()
            top_n = get_rerank_top_n()
            base_retriever = create_retriever(
                k=base_k,
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding=embedding,
                document_id=document_id,
                page_min=page_min,
                page_max=page_max,
                tag=tag,
                document_type=document_type,
            )
            if use_multi_query:
                base_retriever = wrap_retriever_with_multiquery(
                    base_retriever,
                    llm,
                    num_queries=get_multi_query_count(),
                )
            return wrap_retriever_with_rerank(base_retriever, top_n=top_n), None
        else:
            logger.warning("Re-ranking disabled: %s. Using standard retrieval.", err)

    effective_k = k if k is not None else get_retrieve_k()
    retriever, _, truncate_k = _build_standard_retriever(
        llm=llm,
        use_multi_query=use_multi_query,
        effective_k=effective_k,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
        document_id=document_id,
        page_min=page_min,
        page_max=page_max,
        tag=tag,
        document_type=document_type,
    )
    return retriever, truncate_k


def generate_answer(
    query: str,
    docs: list[Document],
    llm: BaseChatModel,
    *,
    response_style: Literal["thorough", "concise"] = "thorough",
) -> RAGResult:
    """Run the LLM generation step given already-retrieved docs.

    Separated from retrieval so callers can emit progress between the two phases.
    """
    if not docs:
        return RAGResult(
            answer="No relevant passages found; try a different query or add more documents.",
            sources=[],
            documents=[],
        )
    prompt = get_rag_prompt(response_style)
    messages = prompt.invoke(
        {"context": format_docs(docs), "question": query}
    ).to_messages()
    try:
        response = llm.invoke(messages)
        raw = getattr(response, "content", response)
        answer = raw if isinstance(raw, str) else str(raw)
        return RAGResult(answer=answer, sources=format_sources(docs), documents=docs)
    except Exception as e:
        logger.exception("LLM invocation failed in generate_answer")
        msg = _user_facing_llm_error_message(e)
        return RAGResult(answer=msg, sources=[], documents=docs)


@traceable(name="run_rag", run_type="chain")
def run_rag(
    query: str,
    llm: BaseChatModel,
    *,
    retriever: BaseRetriever | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    document_id: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    tag: str | None = None,
    document_type: str | None = None,
    response_style: Literal["thorough", "concise"] = "thorough",
) -> RAGResult:
    """Run a 2-step RAG pipeline: retrieve chunks, format context, prompt LLM, return answer with citations.

    Uses LangChain Retriever (store.as_retriever()). Pass a retriever directly for maximum
    flexibility (e.g. wrapped with reranking), or use legacy params to build one from Chroma.

    When PINRAG_USE_MULTI_QUERY=true, generates query variants via LLM, retrieves per
    variant, merges (unique union), then optionally reranks. Improves recall for terse queries.

    Args:
        query: Natural language question to answer.
        llm: Chat model for generation (e.g. from get_chat_model()).
        retriever: Optional BaseRetriever. If provided, used directly; else built from legacy params.
        k: Number of chunks to retrieve. If None, uses PINRAG_RETRIEVE_K (default 10). Ignored when retriever is provided.
        use_rerank: Override config to enable/disable Cohere re-ranking. If None, uses PINRAG_USE_RERANK.
        persist_directory: Chroma persistence directory (used when retriever is None).
        collection_name: Chroma collection name (used when retriever is None). If None, uses provider-based name.
        embedding: Optional embedding model for retrieval (used when retriever is None).
        document_id: Optional document ID to filter retrieval (e.g. PDF file name).
        page_min: Optional start of page range (inclusive). Use with page_max.
        page_max: Optional end of page range (inclusive). Single page: page_min=64, page_max=64.
        tag: Optional tag to filter retrieval (e.g. "PI_PICO").
        document_type: Optional type to filter: "pdf", "youtube", "discord", "github", or "plaintext".
        response_style: Answer style for prompt instructions ("thorough" or "concise").

    Returns:
        RAGResult with answer (str) and sources (list of {document_id, page}).

    """
    query_for_retrieval = preprocess_query(query)
    truncate_k: int | None = None
    if retriever is None:
        retriever, truncate_k = build_retriever(
            llm,
            k=k,
            use_rerank=use_rerank,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
            document_id=document_id,
            page_min=page_min,
            page_max=page_max,
            tag=tag,
            document_type=document_type,
        )
    docs = _retrieve(retriever, query_for_retrieval)
    docs = _apply_post_retrieval_doc_limit(docs, truncate_k)
    return generate_answer(query, docs, llm, response_style=response_style)
