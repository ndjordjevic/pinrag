"""Index PDF documents into the Chroma vector store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pinrag.chunking import chunk_documents
from pinrag.config import (
    get_child_chunk_size,
    get_chunk_overlap,
    get_chunk_size,
    get_collection_name,
    get_parent_chunk_size,
    get_structure_aware_chunking,
    get_use_parent_child,
)
from pinrag.indexing.pdf_loader import PdfLoadResult, load_pdf_as_documents
from pinrag.vectorstore.chroma_client import (
    DEFAULT_PERSIST_DIR,
    get_chroma_store,
)
from pinrag.vectorstore.chroma_filters import build_retrieval_filter
from pinrag.vectorstore.docstore import (
    get_parent_docstore,
    remove_parent_docs_for_document,
)

PathLike = str | Path


def pdf_doc_title(*, pdf_path: Path, page_documents: list[Document]) -> str:
    """Human-facing title for Chroma metadata: PDF /Title if present, else file stem."""
    for doc in page_documents:
        t = doc.metadata.get("document_title")
        if t is not None and str(t).strip():
            return str(t).strip()
    return pdf_path.stem


@dataclass(frozen=True)
class IndexResult:
    """Summary of indexing a single PDF into Chroma."""

    source_path: Path
    total_pages: int
    total_chunks: int
    persist_directory: Path
    collection_name: str


def index_pdf(
    path: PathLike,
    *,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    tag: str | None = None,
) -> IndexResult:
    """Load, chunk, and index a single PDF into Chroma.

    Steps:
    - Load per-page `Document`s via `load_pdf_as_documents`.
    - Chunk them with `chunk_documents`.
    - Get (or create) a Chroma store via `get_chroma_store`.
    - Delete any existing chunks for this document (by file name) to avoid duplicates.
    - Add chunk documents to the store (embeddings generated automatically).

    Args:
        path: PDF path to index.
        persist_directory: Chroma persistence directory.
        collection_name: Chroma collection name. If None, uses provider-based name (e.g. pinrag_openai).
        embedding: Optional embedding model; if None, uses default local Nomic embeddings.
        chunk_size: Chunk size in chars; if None, uses PINRAG_CHUNK_SIZE env or 1000.
        chunk_overlap: Chunk overlap in chars; if None, uses PINRAG_CHUNK_OVERLAP env or 200.
        tag: Optional single tag for this document (e.g. "amiga"); stored on all chunks for filtering.

    Returns:
        IndexResult with basic stats about the indexed PDF.

    """
    if collection_name is None:
        collection_name = get_collection_name()
    respect_structure = get_structure_aware_chunking()
    pdf_path = Path(path).expanduser().resolve()
    pdf_result = load_pdf_as_documents(pdf_path)
    document_id = pdf_path.name
    display_title = pdf_doc_title(pdf_path=pdf_path, page_documents=pdf_result.documents)

    if get_use_parent_child():
        chunk_docs, total_chunks = _index_pdf_parent_child(
            pdf_result=pdf_result,
            pdf_path=pdf_path,
            display_title=display_title,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
            tag=tag,
            respect_structure=respect_structure,
        )
    else:
        # Turn per-page documents into chunk documents (preserving metadata).
        size = chunk_size if chunk_size is not None else get_chunk_size()
        overlap = chunk_overlap if chunk_overlap is not None else get_chunk_overlap()
        chunk_docs = chunk_documents(
            pdf_result.documents,
            chunk_size=size,
            chunk_overlap=overlap,
            respect_structure=respect_structure,
        )

        # Get/create the Chroma store.
        store = get_chroma_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
        )

        # Add document-level metadata to each chunk (upload time, size stats).
        doc_bytes = pdf_path.stat().st_size
        upload_ts = datetime.now(UTC).isoformat()
        doc_pages = pdf_result.total_pages
        doc_total_chunks = len(chunk_docs)
        for doc in chunk_docs:
            doc.metadata["document_type"] = "pdf"
            doc.metadata["doc_title"] = display_title
            doc.metadata["upload_timestamp"] = upload_ts
            doc.metadata["doc_pages"] = doc_pages
            doc.metadata["doc_bytes"] = doc_bytes
            doc.metadata["doc_total_chunks"] = doc_total_chunks
            if tag is not None and str(tag).strip():
                doc.metadata["tag"] = str(tag).strip()

        # Replace existing chunks for this document to avoid duplicates.
        store._collection.delete(where={"document_id": document_id})

        # Add in batches to avoid OpenAI/Chroma batch size limits (e.g. token or record limits)
        batch_size = 100
        if chunk_docs:
            for i in range(0, len(chunk_docs), batch_size):
                batch = chunk_docs[i : i + batch_size]
                store.add_documents(batch)
        total_chunks = len(chunk_docs)

    return IndexResult(
        source_path=pdf_result.source_path,
        total_pages=pdf_result.total_pages,
        total_chunks=total_chunks,
        persist_directory=Path(persist_directory).expanduser().resolve(),
        collection_name=collection_name,
    )


def _index_pdf_parent_child(
    *,
    pdf_result: PdfLoadResult,
    pdf_path: Path,
    display_title: str,
    persist_directory: PathLike,
    collection_name: str,
    embedding: Embeddings | None,
    tag: str | None,
    respect_structure: bool,
) -> tuple[list[Document], int]:
    """Index PDF using parent-child retrieval: embed small chunks, store large parents in docstore."""
    parent_size = get_parent_chunk_size()
    parent_overlap = min(200, parent_size // 10)
    child_size = get_child_chunk_size()
    child_overlap = min(50, child_size // 10)

    # Split into parent chunks.
    parent_chunks = chunk_documents(
        pdf_result.documents,
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        respect_structure=respect_structure,
    )

    store = get_chroma_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
    )
    docstore = get_parent_docstore(persist_directory, collection_name)

    doc_bytes = pdf_path.stat().st_size
    upload_ts = datetime.now(UTC).isoformat()
    doc_pages = pdf_result.total_pages
    document_id = pdf_path.name

    all_children: list[Document] = []
    parent_docstore_entries: list[tuple[str, Document]] = []

    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent.metadata["doc_id"] = parent_id
        parent.metadata["document_id"] = document_id
        parent.metadata["document_type"] = "pdf"
        parent.metadata["doc_title"] = display_title
        parent.metadata["upload_timestamp"] = upload_ts
        parent.metadata["doc_pages"] = doc_pages
        parent.metadata["doc_bytes"] = doc_bytes
        if tag is not None and str(tag).strip():
            parent.metadata["tag"] = str(tag).strip()

        child_chunks = chunk_documents(
            [parent],
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            respect_structure=respect_structure,
        )
        for c in child_chunks:
            c.metadata["doc_id"] = parent_id
            c.metadata["document_id"] = document_id
            c.metadata["document_type"] = "pdf"
            c.metadata["doc_title"] = display_title
            c.metadata["upload_timestamp"] = upload_ts
            c.metadata["doc_pages"] = doc_pages
            c.metadata["doc_bytes"] = doc_bytes
            if tag is not None and str(tag).strip():
                c.metadata["tag"] = str(tag).strip()
            all_children.append(c)

        parent_docstore_entries.append((parent_id, parent))

    total_chunks = len(all_children)
    for c in all_children:
        c.metadata["doc_total_chunks"] = total_chunks
    for _, parent_doc in parent_docstore_entries:
        parent_doc.metadata["doc_total_chunks"] = total_chunks
    remove_parent_docs_for_document(
        store=store,
        docstore=docstore,
        document_id=document_id,
    )

    store._collection.delete(where={"document_id": document_id})

    if parent_docstore_entries:
        docstore.mset(parent_docstore_entries)

    batch_size = 100
    for i in range(0, len(all_children), batch_size):
        batch = all_children[i : i + batch_size]
        store.add_documents(batch)

    return all_children, total_chunks


def query_index(
    query: str,
    *,
    k: int = 5,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    document_id: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    tag: str | None = None,
) -> list[Document]:
    """Run a similarity search over the indexed chunks in Chroma.

    Args:
        query: Natural language query string.
        k: Number of top matching chunks to return.
        persist_directory: Chroma persistence directory (must match indexing).
        collection_name: Chroma collection name (must match indexing). If None, uses provider-based name.
        embedding: Optional embedding model; if None, uses default local Nomic embeddings.
        document_id: Optional document ID to filter by (e.g. PDF file name). Only chunks
            from this document are considered for retrieval.
        page_min: Optional start of page range (inclusive). Use with page_max.
        page_max: Optional end of page range (inclusive). Use with page_min.
            Single page: page_min=64, page_max=64 filters to page 64 only.
        tag: Optional tag to filter by (e.g. "PI_PICO"). Only chunks from documents
            indexed with this tag are considered.

    Returns:
        List of matching chunk `Document`s from the vector store.

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
    )
    return store.similarity_search(query, k=k, filter=filter_dict)
