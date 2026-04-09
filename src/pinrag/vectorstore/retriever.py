"""Retriever creation from Chroma vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinrag.config import (
    get_child_chunk_size,
    get_collection_name,
    get_use_parent_child,
)
from pinrag.vectorstore.chroma_client import (
    DEFAULT_PERSIST_DIR,
    get_chroma_store,
)
from pinrag.vectorstore.chroma_filters import build_retrieval_filter
from pinrag.vectorstore.docstore import get_parent_docstore

PathLike = str | Path


def create_retriever(
    *,
    k: int = 5,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    document_id: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    tag: str | None = None,
    document_type: str | None = None,
) -> BaseRetriever:
    """Create a LangChain retriever from the Chroma vector store.

    Uses store.as_retriever() with search_kwargs for k and metadata filters.
    Compatible with run_rag() and other LangChain components that expect a BaseRetriever.

    Args:
        k: Number of chunks to retrieve (default: 5).
        persist_directory: Chroma persistence directory.
        collection_name: Chroma collection name. If None, uses provider-based name (e.g. pinrag_openai).
        embedding: Optional embedding model; if None, uses default.
        document_id: Optional document ID to filter retrieval.
        page_min: Optional start of page range (inclusive). Use with page_max.
        page_max: Optional end of page range (inclusive).
        tag: Optional tag to filter retrieval.
        document_type: Optional type to filter: "pdf", "youtube", "discord", "github", or "plaintext".

    Returns:
        BaseRetriever (Chroma retriever with configured search_kwargs).

    """
    if collection_name is None:
        collection_name = get_collection_name()
    store = get_chroma_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
    )
    filter_dict = build_retrieval_filter(
        document_id=document_id,
        page_min=page_min,
        page_max=page_max,
        tag=tag,
        document_type=document_type,
    )
    search_kwargs: dict = {"k": k}
    if filter_dict is not None:
        search_kwargs["filter"] = filter_dict

    if get_use_parent_child():
        from langchain_classic.retrievers import ParentDocumentRetriever

        child_size = get_child_chunk_size()
        child_overlap = min(50, child_size // 10)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        docstore = get_parent_docstore(persist_directory, collection_name)
        return ParentDocumentRetriever(
            vectorstore=store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=None,
            id_key="doc_id",
            search_kwargs=search_kwargs,
        )

    return store.as_retriever(search_kwargs=search_kwargs)
