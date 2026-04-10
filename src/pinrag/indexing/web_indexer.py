"""Index web documentation sites into the Chroma vector store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
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
    get_web_concurrency,
    get_web_max_depth,
    get_web_max_page_bytes,
    get_web_max_pages,
    get_web_prefer_llms_txt,
    get_web_rate_limit_per_host,
    get_web_request_timeout,
    get_web_respect_robots,
    get_web_user_agent,
)
from pinrag.indexing.web_loader import (
    CrawlLimits,
    WebLoadResult,
    load_web_docs_as_documents,
)
from pinrag.vectorstore.chroma_client import (
    DEFAULT_PERSIST_DIR,
    get_chroma_store,
)
from pinrag.vectorstore.docstore import (
    get_parent_docstore,
    remove_parent_docs_for_document,
)

PathLike = str | Path


@dataclass(frozen=True)
class WebIndexResult:
    """Summary of indexing a web documentation site into Chroma."""

    seed_url: str
    host: str
    path_prefix: str
    document_id: str
    pages_indexed: int
    pages_failed: int
    total_chunks: int
    discovery: str
    persist_directory: Path
    collection_name: str
    failed_pages: list[dict[str, str]] = field(default_factory=list)


def _crawl_limits_from_config() -> CrawlLimits:
    return CrawlLimits(
        max_pages=get_web_max_pages(),
        max_depth=get_web_max_depth(),
        max_page_bytes=get_web_max_page_bytes(),
        request_timeout=get_web_request_timeout(),
        concurrency=get_web_concurrency(),
        rate_limit_per_host=get_web_rate_limit_per_host(),
        user_agent=get_web_user_agent(),
        respect_robots=get_web_respect_robots(),
        prefer_llms_txt=get_web_prefer_llms_txt(),
    )


def index_web(
    seed_url: str,
    *,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    tag: str | None = None,
) -> WebIndexResult:
    """Discover, fetch, chunk, and index a documentation site into Chroma.

    Upserts by ``document_id = "<host><path_prefix>"`` — existing chunks for that
    site are wiped before new ones are written. Discovery falls through
    ``llms.txt`` → ``sitemap.xml`` → scoped BFS per the loader.
    """
    if collection_name is None:
        collection_name = get_collection_name()
    respect_structure = get_structure_aware_chunking()

    limits = _crawl_limits_from_config()
    load_result: WebLoadResult = load_web_docs_as_documents(seed_url, limits=limits)
    document_id = load_result.document_id

    if not load_result.documents:
        store = get_chroma_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
        )
        if get_use_parent_child():
            docstore = get_parent_docstore(persist_directory, collection_name)
            remove_parent_docs_for_document(
                store=store, docstore=docstore, document_id=document_id
            )
        store._collection.delete(where={"document_id": document_id})
        return WebIndexResult(
            seed_url=load_result.seed_url,
            host=load_result.host,
            path_prefix=load_result.path_prefix,
            document_id=document_id,
            pages_indexed=0,
            pages_failed=len(load_result.failed_pages),
            total_chunks=0,
            discovery=load_result.discovery,
            persist_directory=Path(persist_directory).expanduser().resolve(),
            collection_name=collection_name,
            failed_pages=list(load_result.failed_pages),
        )

    store = get_chroma_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
    )

    if get_use_parent_child():
        docstore = get_parent_docstore(persist_directory, collection_name)
        remove_parent_docs_for_document(
            store=store, docstore=docstore, document_id=document_id
        )
    store._collection.delete(where={"document_id": document_id})

    upload_ts = datetime.now(UTC).isoformat()

    if get_use_parent_child():
        total_chunks = _index_web_parent_child(
            load_result=load_result,
            store=store,
            persist_directory=persist_directory,
            collection_name=collection_name,
            document_id=document_id,
            upload_ts=upload_ts,
            tag=tag,
            respect_structure=respect_structure,
        )
    else:
        total_chunks = _index_web_flat(
            load_result=load_result,
            store=store,
            document_id=document_id,
            upload_ts=upload_ts,
            tag=tag,
            respect_structure=respect_structure,
        )

    return WebIndexResult(
        seed_url=load_result.seed_url,
        host=load_result.host,
        path_prefix=load_result.path_prefix,
        document_id=document_id,
        pages_indexed=len(load_result.documents),
        pages_failed=len(load_result.failed_pages),
        total_chunks=total_chunks,
        discovery=load_result.discovery,
        persist_directory=Path(persist_directory).expanduser().resolve(),
        collection_name=collection_name,
        failed_pages=list(load_result.failed_pages),
    )


def _index_web_flat(
    *,
    load_result: WebLoadResult,
    store,
    document_id: str,
    upload_ts: str,
    tag: str | None,
    respect_structure: bool,
) -> int:
    size = get_chunk_size()
    overlap = get_chunk_overlap()
    all_chunks: list[Document] = []

    for page_doc in load_result.documents:
        doc_bytes = page_doc.metadata.get("doc_bytes", 0)
        chunk_docs = chunk_documents(
            [page_doc],
            chunk_size=size,
            chunk_overlap=overlap,
            document_id_key="document_id",
            respect_structure=respect_structure,
        )
        for doc in chunk_docs:
            doc.metadata["document_type"] = "web"
            doc.metadata["upload_timestamp"] = upload_ts
            doc.metadata["doc_bytes"] = doc_bytes
            if tag is not None and str(tag).strip():
                doc.metadata["tag"] = str(tag).strip()
            all_chunks.append(doc)

    total_chunks = len(all_chunks)
    for doc in all_chunks:
        doc.metadata["doc_total_chunks"] = total_chunks

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        store.add_documents(all_chunks[i : i + batch_size])
    return total_chunks


def _index_web_parent_child(
    *,
    load_result: WebLoadResult,
    store,
    persist_directory: PathLike,
    collection_name: str,
    document_id: str,
    upload_ts: str,
    tag: str | None,
    respect_structure: bool,
) -> int:
    parent_size = get_parent_chunk_size()
    parent_overlap = min(200, parent_size // 10)
    child_size = get_child_chunk_size()
    child_overlap = min(50, child_size // 10)

    docstore = get_parent_docstore(persist_directory, collection_name)

    all_children: list[Document] = []
    parent_entries: list[tuple[str, Document]] = []

    for page_doc in load_result.documents:
        doc_bytes = page_doc.metadata.get("doc_bytes", 0)
        source_url = page_doc.metadata.get("source_url")
        doc_title = page_doc.metadata.get("doc_title")

        parent_chunks = chunk_documents(
            [page_doc],
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            document_id_key="document_id",
            respect_structure=respect_structure,
        )

        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())
            parent.metadata.update(
                {
                    "doc_id": parent_id,
                    "document_id": document_id,
                    "document_type": "web",
                    "upload_timestamp": upload_ts,
                    "doc_bytes": doc_bytes,
                }
            )
            if source_url:
                parent.metadata["source_url"] = source_url
                parent.metadata["source"] = source_url
            if doc_title:
                parent.metadata["doc_title"] = doc_title
            if tag is not None and str(tag).strip():
                parent.metadata["tag"] = str(tag).strip()

            child_chunks = chunk_documents(
                [parent],
                chunk_size=child_size,
                chunk_overlap=child_overlap,
                respect_structure=respect_structure,
            )
            for child in child_chunks:
                child.metadata.update(
                    {
                        "doc_id": parent_id,
                        "document_id": document_id,
                        "document_type": "web",
                        "upload_timestamp": upload_ts,
                        "doc_bytes": doc_bytes,
                    }
                )
                if source_url:
                    child.metadata["source_url"] = source_url
                    child.metadata["source"] = source_url
                if doc_title:
                    child.metadata["doc_title"] = doc_title
                if tag is not None and str(tag).strip():
                    child.metadata["tag"] = str(tag).strip()
                all_children.append(child)

            parent_entries.append((parent_id, parent))

    total_chunks = len(all_children)
    for child in all_children:
        child.metadata["doc_total_chunks"] = total_chunks
    for _, parent_doc in parent_entries:
        parent_doc.metadata["doc_total_chunks"] = total_chunks

    if parent_entries:
        docstore.mset(parent_entries)

    batch_size = 100
    for i in range(0, len(all_children), batch_size):
        store.add_documents(all_children[i : i + batch_size])
    return total_chunks
