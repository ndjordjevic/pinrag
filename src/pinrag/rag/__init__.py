"""RAG chain: retrieval-augmented generation with citations."""

from pinrag.rag.chain import RAGResult, build_retriever, generate_answer, run_rag
from pinrag.rag.formatting import format_docs, format_sources
from pinrag.rag.prompts import RAG_PROMPT

__all__ = [
    "RAGResult",
    "RAG_PROMPT",
    "build_retriever",
    "format_docs",
    "format_sources",
    "generate_answer",
    "run_rag",
]
