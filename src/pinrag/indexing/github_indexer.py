"""Index GitHub repository contents into the Chroma vector store."""

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
    get_github_max_file_bytes,
    get_github_token,
    get_parent_chunk_size,
    get_structure_aware_chunking,
    get_use_parent_child,
)
from pinrag.indexing.github_loader import (
    GitHubLoadResult,
    load_github_repo_as_documents,
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
class GitHubIndexResult:
    """Summary of indexing a GitHub repo into Chroma."""

    owner: str
    repo: str
    branch: str
    files_indexed: int
    total_chunks: int
    persist_directory: Path
    collection_name: str
    failed_files: list[dict[str, str]] = field(default_factory=list)


def index_github(
    repo_url: str,
    *,
    persist_directory: PathLike = DEFAULT_PERSIST_DIR,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
    tag: str | None = None,
    branch: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_bytes: int | None = None,
) -> GitHubIndexResult:
    """Load, chunk, and index a GitHub repository into Chroma.

    Fetches files via GitHub API, chunks each file, and adds to the vector store.
    Replaces existing chunks for each file (upsert per file).

    Args:
        repo_url: GitHub URL (e.g. https://github.com/owner/repo).
        persist_directory: Chroma persistence directory.
        collection_name: Chroma collection name. If None, uses config default.
        embedding: Optional embedding model; if None, uses default.
        tag: Optional tag for all chunks; stored for filtering.
        branch: Override branch. If None, parsed from URL or defaults to main.
        include_patterns: Glob patterns for files to include.
        exclude_patterns: Glob patterns to exclude.
        max_file_bytes: Skip files larger than this. If None, uses config.

    Returns:
        GitHubIndexResult with owner, repo, branch, files_indexed, total_chunks,
        and per-file loader failures in ``failed_files`` when applicable.

    """
    if collection_name is None:
        collection_name = get_collection_name()
    respect_structure = get_structure_aware_chunking()
    token = get_github_token()
    max_bytes = (
        max_file_bytes if max_file_bytes is not None else get_github_max_file_bytes()
    )

    load_result = load_github_repo_as_documents(
        repo_url,
        token=token,
        branch=branch,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_bytes=max_bytes,
    )

    if not load_result.documents:
        store = get_chroma_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
        )
        repo_id = f"{load_result.owner}/{load_result.repo}"
        if get_use_parent_child():
            docstore = get_parent_docstore(persist_directory, collection_name)
            remove_parent_docs_for_document(
                store=store, docstore=docstore, document_id=repo_id
            )
        store._collection.delete(where={"document_id": repo_id})
        return GitHubIndexResult(
            owner=load_result.owner,
            repo=load_result.repo,
            branch=load_result.branch,
            files_indexed=0,
            total_chunks=0,
            persist_directory=Path(persist_directory).expanduser().resolve(),
            collection_name=collection_name,
            failed_files=list(load_result.failed_files),
        )

    store = get_chroma_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
    )

    repo_id = f"{load_result.owner}/{load_result.repo}"
    if get_use_parent_child():
        docstore = get_parent_docstore(persist_directory, collection_name)
        remove_parent_docs_for_document(
            store=store, docstore=docstore, document_id=repo_id
        )
    store._collection.delete(where={"document_id": repo_id})

    upload_ts = datetime.now(UTC).isoformat()

    if get_use_parent_child():
        total_chunks = _index_github_parent_child(
            load_result=load_result,
            store=store,
            persist_directory=persist_directory,
            collection_name=collection_name,
            repo_id=repo_id,
            upload_ts=upload_ts,
            tag=tag,
            respect_structure=respect_structure,
        )
    else:
        total_chunks = _index_github_flat(
            load_result=load_result,
            store=store,
            repo_id=repo_id,
            upload_ts=upload_ts,
            tag=tag,
            respect_structure=respect_structure,
        )

    return GitHubIndexResult(
        owner=load_result.owner,
        repo=load_result.repo,
        branch=load_result.branch,
        files_indexed=len(load_result.documents),
        total_chunks=total_chunks,
        persist_directory=Path(persist_directory).expanduser().resolve(),
        collection_name=collection_name,
        failed_files=list(load_result.failed_files),
    )


def _index_github_flat(
    *,
    load_result: GitHubLoadResult,
    store,
    repo_id: str,
    upload_ts: str,
    tag: str | None,
    respect_structure: bool,
) -> int:
    """Index GitHub files as flat chunks (no parent-child)."""
    size = get_chunk_size()
    overlap = get_chunk_overlap()
    all_chunks: list[Document] = []

    for file_doc in load_result.documents:
        doc_bytes = file_doc.metadata.get("doc_bytes", 0)

        chunk_docs = chunk_documents(
            [file_doc],
            chunk_size=size,
            chunk_overlap=overlap,
            document_id_key="document_id",
            respect_structure=respect_structure,
        )

        for doc in chunk_docs:
            doc.metadata["document_type"] = "github"
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
        batch = all_chunks[i : i + batch_size]
        store.add_documents(batch)

    return total_chunks


def _index_github_parent_child(
    *,
    load_result: GitHubLoadResult,
    store,
    persist_directory: PathLike,
    collection_name: str,
    repo_id: str,
    upload_ts: str,
    tag: str | None,
    respect_structure: bool,
) -> int:
    """Index GitHub files with parent-child retrieval: small child chunks in Chroma, large parents in docstore."""
    parent_size = get_parent_chunk_size()
    parent_overlap = min(200, parent_size // 10)
    child_size = get_child_chunk_size()
    child_overlap = min(50, child_size // 10)

    docstore = get_parent_docstore(persist_directory, collection_name)

    all_children: list[Document] = []
    parent_entries: list[tuple[str, Document]] = []

    for file_doc in load_result.documents:
        doc_bytes = file_doc.metadata.get("doc_bytes", 0)

        parent_chunks = chunk_documents(
            [file_doc],
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
                    "document_id": repo_id,
                    "document_type": "github",
                    "upload_timestamp": upload_ts,
                    "doc_bytes": doc_bytes,
                }
            )
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
                        "document_id": repo_id,
                        "document_type": "github",
                        "upload_timestamp": upload_ts,
                        "doc_bytes": doc_bytes,
                    }
                )
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
        batch = all_children[i : i + batch_size]
        store.add_documents(batch)

    return total_chunks
