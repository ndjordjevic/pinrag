"""Transport-agnostic query, add, list, and remove operations."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import chromadb
from langchain_core.documents import Document
from langsmith import traceable

from pinrag.config import get_collection_name, get_persist_dir, get_use_parent_child
from pinrag.core.format_detection import (
    categorize_failures,
    detect_file_format,
    detect_source_format,
    resolve_persist_dir_path,
    resolve_user_content_path,
)
from pinrag.embeddings import get_embedding_model
from pinrag.indexing import (
    DiscordIndexResult,
    GitHubIndexResult,
    IndexResult,
    PlaintextIndexResult,
    YouTubeIndexResult,
    index_discord,
    index_github,
    index_pdf,
    index_plaintext,
    index_youtube,
    index_youtube_playlist,
)
from pinrag.llm import get_chat_model
from pinrag.rag import build_retriever, generate_answer, run_rag
from pinrag.rag.chain import RAGResult, _apply_post_retrieval_doc_limit, _retrieve
from pinrag.rag.query_preprocess import preprocess_query
from pinrag.vectorstore import get_chroma_store
from pinrag.vectorstore.docstore import get_parent_docstore

logger = logging.getLogger(__name__)
VerboseSyncEmitter = Callable[[str, str], None]
PhaseCallback = Callable[[str], None]


def _chunk_metadata_looks_like_pdf(meta: dict[str, Any]) -> bool:
    if meta.get("document_type") == "pdf":
        return True
    for key in ("document_id", "file_name"):
        v = meta.get(key)
        if isinstance(v, str) and v.lower().endswith(".pdf"):
            return True
    return False


def _logical_doc_key(meta: dict[str, Any]) -> str:
    return str(
        meta.get("document_id")
        or meta.get("file_name")
        or meta.get("source")
        or ""
    )


def _ensure_list_document_title(doc_id: str, details: dict[str, Any]) -> None:
    """Set display title for list_documents when doc_title was never stored."""
    if str(details.get("title") or "").strip():
        return
    dtype = details.get("document_type")
    if dtype == "pdf" or str(doc_id).lower().endswith(".pdf"):
        details["title"] = Path(doc_id).stem
    else:
        details["title"] = doc_id


def _list_title_from_chunk_meta(meta: dict[str, Any], doc_ref: str) -> str:
    """Display title for one chunk (aligned with list_documents / doc_title fallbacks)."""
    t = str(meta.get("doc_title") or "").strip()
    if t:
        return t
    dtype = meta.get("document_type")
    if dtype == "pdf" or doc_ref.lower().endswith(".pdf"):
        return Path(doc_ref).stem if doc_ref else ""
    return doc_ref


def _resolve_remove_document_id(store: Any, user_input: str) -> str:
    """Map user input to canonical ``document_id`` stored in Chroma.

    Tries exact ``document_id`` first, then case-insensitive match on list title
    or PDF filename stem (when the stored ref ends with ``.pdf``).
    """
    key = user_input.strip()
    if not key:
        raise ValueError("document_id cannot be empty")

    direct = store.get(where={"document_id": key}, include=[])
    if direct.get("ids"):
        return key

    bulk = store.get(include=["metadatas"])
    metas = bulk.get("metadatas") or []
    titles_by_ref: dict[str, str] = {}
    for m in metas:
        if not isinstance(m, dict):
            continue
        ref = _logical_doc_key(m).strip()
        if not ref:
            continue
        title = _list_title_from_chunk_meta(m, ref)
        if ref not in titles_by_ref:
            titles_by_ref[ref] = title

    q = key.casefold()
    candidates: set[str] = set()
    for ref, title in titles_by_ref.items():
        if title.casefold() == q:
            candidates.add(ref)
        if ref.lower().endswith(".pdf") and Path(ref).stem.casefold() == q:
            candidates.add(ref)

    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        shortlist = ", ".join(sorted(candidates)[:8])
        raise ValueError(
            f"Ambiguous document {user_input!r}: matches multiple documents "
            f"({shortlist}). Use the exact ref from list_documents."
        )
    return key


def _emit_verbose(
    verbose_emitter: VerboseSyncEmitter | None, message: str, level: str = "info"
) -> None:
    """Best-effort sync verbose emitter for MCP notifications."""
    if verbose_emitter is None:
        return
    try:
        verbose_emitter(message, level)
    except Exception:
        logger.debug("verbose_emit_failed message=%s", message, exc_info=True)


@traceable(name="query", run_type="tool")
def query(
    user_query: str = "",
    document_id: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    tag: str | None = None,
    document_type: str | None = None,
    response_style: Literal["thorough", "concise"] = "thorough",
    persist_dir: str = "",
    collection: str | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
    phase_callback: PhaseCallback | None = None,
) -> dict[str, Any]:
    """Query indexed documents (PDF, Discord) and return an answer with citations.

    Retrieval is driven by .env: PINRAG_RETRIEVE_K, rerank, multi-query, etc.
    Persist dir and collection come from persist_dir/collection params or
    PINRAG_PERSIST_DIR and PINRAG_COLLECTION_NAME when not provided.

    Args:
        user_query: Natural language question to ask.
        document_id: Optional document selector to filter retrieval: exact list **ref**
            (Chroma ``document_id``), exact list **title**, or unique PDF **stem** — same
            resolution as ``remove_document`` / ``set_document_tag``.
        page_min: Optional start of page range (inclusive). Use with page_max. PDF only.
        page_max: Optional end of page range (inclusive). Single page: page_min=64, page_max=64. PDF only.
        tag: Optional tag to filter retrieval (e.g. from list_documents document_details).
        document_type: Optional type to filter: "pdf", "youtube", "discord", "github", or "plaintext".
        response_style: Answer style for generation ("thorough" or "concise").
        persist_dir: Chroma persistence directory (default: from PINRAG_PERSIST_DIR or chroma_db).
        collection: Chroma collection name (default: from PINRAG_COLLECTION_NAME or pinrag).
        verbose_emitter: Optional sync callback for progress messages (e.g. MCP verbose).
        phase_callback: Optional sync callback called with a phase label between retrieval
            and generation (e.g. ``"generating"``). Useful for emitting mid-query progress.

    Returns:
        Dictionary with "answer" (str) and "sources" (list of dicts with document_id and page).

    Raises:
        ValueError: If query is empty or invalid.
        FileNotFoundError: If persist dir doesn't exist.

    """
    if not user_query or not user_query.strip():
        raise ValueError("Query cannot be empty")
    if (page_min is not None) != (page_max is not None):
        raise ValueError(
            "page_min and page_max must be provided together for page range filter"
        )
    if page_min is not None and page_max is not None and page_min > page_max:
        raise ValueError("page_min must be <= page_max")
    if response_style not in ("thorough", "concise"):
        raise ValueError("response_style must be 'thorough' or 'concise'")

    _persist = (persist_dir or "").strip() or get_persist_dir()
    _collection = (collection or "").strip() or get_collection_name()
    persist_path = resolve_persist_dir_path(_persist)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Persistence directory does not exist: {_persist}. "
            "Index some documents first using add_document_tool."
        )

    embedding = get_embedding_model()
    llm = get_chat_model()

    doc_id_filter = (
        document_id.strip() if document_id and str(document_id).strip() else None
    )
    if doc_id_filter:
        store_resolve = get_chroma_store(
            persist_directory=str(persist_path),
            collection_name=_collection,
            embedding=embedding,
        )
        requested_doc = doc_id_filter
        doc_id_filter = _resolve_remove_document_id(store_resolve, requested_doc)
        if doc_id_filter != requested_doc:
            _emit_verbose(
                verbose_emitter,
                f"phase=query_document_resolved requested={requested_doc!r} "
                f"document_id={doc_id_filter!r}",
            )
    tag_filter = tag.strip() if tag and str(tag).strip() else None
    doc_type_filter = (
        document_type.strip() if document_type and str(document_type).strip() else None
    )

    _emit_verbose(verbose_emitter, "phase=retrieving")
    retriever, truncate_k = build_retriever(
        llm,
        persist_directory=str(persist_path),
        collection_name=_collection,
        embedding=embedding,
        document_id=doc_id_filter,
        page_min=page_min,
        page_max=page_max,
        tag=tag_filter,
        document_type=doc_type_filter,
    )
    query_for_retrieval = preprocess_query(user_query)
    docs = _retrieve(retriever, query_for_retrieval)
    docs = _apply_post_retrieval_doc_limit(docs, truncate_k)
    _emit_verbose(verbose_emitter, f"phase=retrieved doc_count={len(docs)}")

    if phase_callback is not None:
        try:
            phase_callback("generating")
        except Exception:
            pass

    _emit_verbose(verbose_emitter, "phase=generating")
    rag_result = generate_answer(user_query, docs, llm, response_style=response_style)
    _emit_verbose(
        verbose_emitter,
        f"phase=query_complete source_count={len(rag_result.sources)} response_style={response_style}",
    )

    sources_out: list[dict[str, Any]] = []
    for s in rag_result.sources:
        item: dict[str, Any] = {
            "document_id": str(s.get("document_id", "unknown")),
            "page": int(s.get("page", 0)),
        }
        if "start" in s:
            item["start"] = int(s["start"])
        if s.get("title"):
            item["title"] = str(s["title"])
        sources_out.append(item)
    return {"answer": rag_result.answer, "sources": sources_out}


@traceable(name="add_file", run_type="tool")
def add_file(
    path: str,
    persist_dir: str = "",
    collection: str | None = None,
    tag: str | None = None,
    branch: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
) -> dict[str, Any]:
    """Add a file, directory, YouTube video, or GitHub repo to the index.

    Automatically detects format: GitHub URL, YouTube URL/video ID, PDF (.pdf), or
    Discord export (.txt with DiscordChatExporter header). Indexes the item or
    all supported files in the directory.

    Args:
        path: Path to a file/directory, YouTube URL, or GitHub URL (e.g. https://github.com/owner/repo).
        persist_dir: Chroma persistence directory (default: "chroma_db").
        collection: Chroma collection name (default: "pinrag").
        tag: Optional tag for indexed documents; stored on all chunks for filtering.
        branch: For GitHub: override branch (default: main). Ignored for other formats.
        include_patterns: For GitHub: glob patterns for files to include (e.g. ["*.md", "src/**/*.py"]).
        exclude_patterns: For GitHub: glob patterns to exclude. Ignored for other formats.
        verbose_emitter: Optional sync callback for progress messages (e.g. MCP verbose).

    Returns:
        Dictionary with "indexed" (list of results), "failed" (errors),
        "total_indexed", "total_failed", "persist_directory", "collection_name".

    """
    if not path or not str(path).strip():
        raise ValueError("path cannot be empty")
    if collection is None or not str(collection).strip():
        collection = get_collection_name()
    else:
        collection = str(collection).strip()

    _persist = (persist_dir or "").strip() or get_persist_dir()
    tag_clean = tag.strip() if tag and str(tag).strip() else None

    fmt = detect_source_format(path)
    _emit_verbose(
        verbose_emitter,
        f"phase=detect_format path={path!r} format={(fmt or 'unsupported')!r}",
    )
    if fmt == "github":
        logger.info(
            "Indexing GitHub repo: %s", path[:80] + "..." if len(path) > 80 else path
        )
        try:
            _emit_verbose(verbose_emitter, f"phase=github_index_start path={path!r}")
            embedding = get_embedding_model()
            result_gh: GitHubIndexResult = index_github(
                path,
                persist_directory=_persist,
                collection_name=collection,
                embedding=embedding,
                tag=tag_clean,
                branch=branch.strip() if branch and str(branch).strip() else None,
                include_patterns=include_patterns if include_patterns else None,
                exclude_patterns=exclude_patterns if exclude_patterns else None,
            )
            logger.info(
                "GitHub indexed: %s (%d files, %d chunks)",
                f"{result_gh.owner}/{result_gh.repo}",
                result_gh.files_indexed,
                result_gh.total_chunks,
            )
            _emit_verbose(
                verbose_emitter,
                f"phase=github_index_done repo={result_gh.owner}/{result_gh.repo} files={result_gh.files_indexed} chunks={result_gh.total_chunks}",
            )
            gh_item: dict[str, Any] = {
                "path": path,
                "format": "github",
                "repo": f"{result_gh.owner}/{result_gh.repo}",
                "branch": result_gh.branch,
                "files_indexed": result_gh.files_indexed,
                "total_chunks": result_gh.total_chunks,
            }
            if result_gh.failed_files:
                gh_item["failed_files"] = result_gh.failed_files
            return {
                "indexed": [gh_item],
                "failed": [],
                "total_indexed": 1,
                "total_failed": 0,
                "persist_directory": str(resolve_persist_dir_path(_persist)),
                "collection_name": collection,
            }
        except Exception as e:
            logger.warning("GitHub indexing failed: %s - %s", path, e)
            _emit_verbose(
                verbose_emitter,
                f"phase=github_index_error path={path!r} error={str(e)!r}",
                level="warning",
            )
            return {
                "indexed": [],
                "failed": [{"path": path, "error": str(e)}],
                "total_indexed": 0,
                "total_failed": 1,
                "persist_directory": str(resolve_persist_dir_path(_persist)),
                "collection_name": collection,
            }
    if fmt == "youtube_playlist":
        logger.info(
            "Indexing YouTube playlist: %s",
            path[:80] + "..." if len(path) > 80 else path,
        )
        try:
            _emit_verbose(
                verbose_emitter, f"phase=youtube_playlist_start path={path!r}"
            )
            embedding = get_embedding_model()
            result_pl = index_youtube_playlist(
                path,
                persist_directory=_persist,
                collection_name=collection,
                embedding=embedding,
                tag=tag_clean,
                verbose_emitter=verbose_emitter,
            )
            indexed_items: list[dict[str, Any]] = []
            for r in result_pl.indexed:
                item: dict[str, Any] = {
                    "path": path,
                    "format": "youtube_playlist",
                    "video_id": r.video_id,
                    "source_url": r.source_url,
                    "total_segments": r.total_segments,
                    "total_chunks": r.total_chunks,
                }
                if r.title:
                    item["title"] = r.title
                indexed_items.append(item)
            failed_items = [
                {
                    "path": f"https://www.youtube.com/watch?v={f['video_id']}",
                    "error": f["error"],
                }
                for f in result_pl.failed
            ]
            logger.info(
                "YouTube playlist indexed: %d videos, %d failed",
                result_pl.total_indexed,
                result_pl.total_failed,
            )
            _emit_verbose(
                verbose_emitter,
                f"phase=youtube_playlist_done indexed={result_pl.total_indexed} failed={result_pl.total_failed}",
            )
            out: dict[str, Any] = {
                "indexed": indexed_items,
                "failed": failed_items,
                "total_indexed": result_pl.total_indexed,
                "total_failed": result_pl.total_failed,
                "persist_directory": str(resolve_persist_dir_path(_persist)),
                "collection_name": collection,
            }
            if failed_items:
                out["fail_summary"] = categorize_failures(failed_items)
            return out
        except Exception as e:
            logger.warning("YouTube playlist indexing failed: %s - %s", path, e)
            _emit_verbose(
                verbose_emitter,
                f"phase=youtube_playlist_error path={path!r} error={str(e)!r}",
                level="warning",
            )
            return {
                "indexed": [],
                "failed": [{"path": path, "error": str(e)}],
                "total_indexed": 0,
                "total_failed": 1,
                "persist_directory": str(resolve_persist_dir_path(_persist)),
                "collection_name": collection,
            }
    if fmt == "youtube":
        logger.info(
            "Indexing YouTube video: %s", path[:80] + "..." if len(path) > 80 else path
        )
        try:
            _emit_verbose(verbose_emitter, f"phase=youtube_index_start path={path!r}")
            embedding = get_embedding_model()
            result_yt: YouTubeIndexResult = index_youtube(
                path,
                persist_directory=_persist,
                collection_name=collection,
                embedding=embedding,
                tag=tag_clean,
                verbose_emitter=verbose_emitter,
            )
            indexed_item: dict[str, Any] = {
                "path": path,
                "format": "youtube",
                "video_id": result_yt.video_id,
                "source_url": result_yt.source_url,
                "total_segments": result_yt.total_segments,
                "total_chunks": result_yt.total_chunks,
            }
            if result_yt.title:
                indexed_item["title"] = result_yt.title
            logger.info(
                "YouTube video indexed: %s (%d chunks)",
                result_yt.video_id,
                result_yt.total_chunks,
            )
            _emit_verbose(
                verbose_emitter,
                f"phase=youtube_index_done video_id={result_yt.video_id!r} segments={result_yt.total_segments} chunks={result_yt.total_chunks}",
            )
            return {
                "indexed": [indexed_item],
                "failed": [],
                "total_indexed": 1,
                "total_failed": 0,
                "persist_directory": str(resolve_persist_dir_path(_persist)),
                "collection_name": collection,
            }
        except Exception as e:
            logger.warning("YouTube video indexing failed: %s - %s", path, e)
            _emit_verbose(
                verbose_emitter,
                f"phase=youtube_index_error path={path!r} error={str(e)!r}",
                level="warning",
            )
            return {
                "indexed": [],
                "failed": [{"path": path, "error": str(e)}],
                "total_indexed": 0,
                "total_failed": 1,
                "persist_directory": str(resolve_persist_dir_path(_persist)),
                "collection_name": collection,
            }
    if fmt == "directory":
        pass
    elif fmt is None:
        base = resolve_user_content_path(path)
        if not base.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        raise ValueError(
            f"Unsupported format: {path}. "
            "Supported: GitHub URL, YouTube URL/video ID, YouTube playlist URL, .pdf, .txt (Discord or plain text)."
        )

    base = resolve_user_content_path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    files_to_index: list[Path] = []
    if base.is_file():
        if detect_file_format(base):
            files_to_index.append(base)
        else:
            raise ValueError(
                f"Unsupported file format: {base.name}. "
                "Supported: .pdf, .txt (Discord or plain text)."
            )
    else:
        for p in sorted(base.rglob("*")):
            if p.is_file() and detect_file_format(p):
                files_to_index.append(p)

    if not files_to_index:
        return {
            "indexed": [],
            "failed": [],
            "total_indexed": 0,
            "total_failed": 0,
            "persist_directory": str(resolve_persist_dir_path(_persist)),
            "collection_name": collection,
        }

    embedding = get_embedding_model()
    indexed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    logger.info("Indexing %d file(s) from %s", len(files_to_index), path)
    _emit_verbose(
        verbose_emitter,
        f"phase=file_index_batch_start root={path!r} files={len(files_to_index)}",
    )
    for f in files_to_index:
        try:
            file_fmt = detect_file_format(f)
            _emit_verbose(
                verbose_emitter,
                f"phase=file_index_item path={str(f)!r} format={(file_fmt or 'unsupported')!r}",
            )
            if file_fmt == "pdf":
                result: IndexResult = index_pdf(
                    f,
                    persist_directory=_persist,
                    collection_name=collection,
                    embedding=embedding,
                    tag=tag_clean,
                )
                logger.info(
                    "PDF indexed: %s (%d pages, %d chunks)",
                    f.name,
                    result.total_pages,
                    result.total_chunks,
                )
                indexed.append(
                    {
                        "path": str(f),
                        "format": "pdf",
                        "source_path": str(result.source_path),
                        "total_pages": result.total_pages,
                        "total_chunks": result.total_chunks,
                    }
                )
            elif file_fmt == "discord":
                result_d: DiscordIndexResult = index_discord(
                    f,
                    persist_directory=_persist,
                    collection_name=collection,
                    embedding=embedding,
                    tag=tag_clean,
                )
                logger.info(
                    "Discord indexed: %s (%d messages, %d chunks)",
                    result_d.document_id,
                    result_d.total_messages,
                    result_d.total_chunks,
                )
                indexed.append(
                    {
                        "path": str(f),
                        "format": "discord",
                        "source_path": str(result_d.source_path),
                        "document_id": result_d.document_id,
                        "channel": result_d.channel,
                        "guild": result_d.guild,
                        "total_messages": result_d.total_messages,
                        "total_chunks": result_d.total_chunks,
                    }
                )
            elif file_fmt == "plaintext":
                result_pt: PlaintextIndexResult = index_plaintext(
                    f,
                    persist_directory=_persist,
                    collection_name=collection,
                    embedding=embedding,
                    tag=tag_clean,
                )
                logger.info(
                    "Plaintext indexed: %s (%d chunks)",
                    result_pt.document_id,
                    result_pt.total_chunks,
                )
                indexed.append(
                    {
                        "path": str(f),
                        "format": "plaintext",
                        "source_path": str(result_pt.source_path),
                        "document_id": result_pt.document_id,
                        "total_chunks": result_pt.total_chunks,
                    }
                )
            else:
                failed.append({"path": str(f), "error": "Unsupported format"})
        except Exception as e:
            logger.warning("File indexing failed: %s - %s", f, e)
            _emit_verbose(
                verbose_emitter,
                f"phase=file_index_error path={str(f)!r} error={str(e)!r}",
                level="warning",
            )
            failed.append({"path": str(f), "error": str(e)})

    return {
        "indexed": indexed,
        "failed": failed,
        "total_indexed": len(indexed),
        "total_failed": len(failed),
        "persist_directory": str(resolve_persist_dir_path(_persist)),
        "collection_name": collection,
    }


@traceable(name="add_files", run_type="tool")
def add_files(
    paths: list[str],
    persist_dir: str = "",
    collection: str | None = None,
    tags: list[str] | None = None,
    branch: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
) -> dict[str, Any]:
    """Add multiple files, directories, or URLs to the index in one call.

    Automatically detects format per path (PDF, Discord export, YouTube, GitHub). Continues
    indexing even if some paths fail.

    Args:
        paths: List of file or directory paths to index.
        persist_dir: Chroma persistence directory (default: "chroma_db").
        collection: Chroma collection name (default: "pinrag").
        tags: Optional list of tags, one per path (same order as paths). Empty string = no tag.
        branch: For GitHub URLs: override branch (default: main). Ignored for other formats.
        include_patterns: For GitHub URLs: glob patterns for files to include (e.g. ["*.md", "src/**/*.py"]).
        exclude_patterns: For GitHub URLs: glob patterns to exclude. Ignored for other formats.
        verbose_emitter: Optional sync callback for progress messages (e.g. MCP verbose).

    Returns:
        Dictionary containing indexed file results, failed file errors, and totals.

    """
    if not paths:
        raise ValueError("paths cannot be empty")
    if collection is None or not str(collection).strip():
        collection = get_collection_name()
    else:
        collection = str(collection).strip()
    if tags is not None and len(tags) != len(paths):
        raise ValueError("tags must have same length as paths when provided")

    _persist = (persist_dir or "").strip() or get_persist_dir()
    all_indexed: list[dict[str, Any]] = []
    all_failed: list[dict[str, str]] = []

    n_paths = len(paths)
    _emit_verbose(verbose_emitter, f"phase=add_files_start paths={n_paths}")
    for i, raw_path in enumerate(paths):
        if not raw_path or not str(raw_path).strip():
            all_failed.append({"path": str(raw_path), "error": "path cannot be empty"})
            continue
        doc_tag: str | None = None
        if tags is not None and i < len(tags) and tags[i] and str(tags[i]).strip():
            doc_tag = str(tags[i]).strip()
        if n_paths > 1:
            logger.info(
                "Processing path %d/%d: %s",
                i + 1,
                n_paths,
                raw_path[:60] + "..." if len(raw_path) > 60 else raw_path,
            )
        _emit_verbose(
            verbose_emitter,
            f"phase=add_files_path_start index={i + 1} total={n_paths} path={raw_path!r}",
        )
        try:
            r = add_file(
                path=raw_path,
                persist_dir=_persist,
                collection=collection,
                tag=doc_tag,
                branch=branch.strip() if branch and str(branch).strip() else None,
                include_patterns=include_patterns if include_patterns else None,
                exclude_patterns=exclude_patterns if exclude_patterns else None,
                verbose_emitter=verbose_emitter,
            )
            all_indexed.extend(r["indexed"])
            all_failed.extend(r["failed"])
            _emit_verbose(
                verbose_emitter,
                f"phase=add_files_path_done index={i + 1} indexed={len(r['indexed'])} failed={len(r['failed'])}",
            )
        except Exception as e:
            logger.warning("Path failed: %s - %s", raw_path, e)
            _emit_verbose(
                verbose_emitter,
                f"phase=add_files_path_error index={i + 1} error={str(e)!r}",
                level="warning",
            )
            all_failed.append({"path": str(raw_path), "error": str(e)})

    logger.info(
        "add_files done: %d indexed, %d failed", len(all_indexed), len(all_failed)
    )
    _emit_verbose(
        verbose_emitter,
        f"phase=add_files_done indexed={len(all_indexed)} failed={len(all_failed)}",
    )
    result: dict[str, Any] = {
        "indexed": all_indexed,
        "failed": all_failed,
        "total_indexed": len(all_indexed),
        "total_failed": len(all_failed),
        "persist_directory": str(resolve_persist_dir_path(_persist)),
        "collection_name": collection,
    }
    if all_failed:
        fail_summary = categorize_failures(all_failed)
        result["fail_summary"] = fail_summary
        logger.info(
            "Fail summary: blocked=%d, disabled=%d, missing_transcript=%d, other=%d",
            fail_summary["blocked"],
            fail_summary["disabled"],
            fail_summary["missing_transcript"],
            fail_summary["other"],
        )
    return result


@traceable(name="list_collections", run_type="tool")
def list_collections(
    persist_dir: str | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
) -> dict[str, Any]:
    """List Chroma collection names in the persist directory.

    Args:
        persist_dir: Chroma persistence directory (default from env / chroma_db).
        verbose_emitter: Optional sync callback for progress messages.

    Returns:
        Dictionary with ``collections`` (sorted names) and ``persist_directory``.
    """
    _persist = (persist_dir or "").strip() or get_persist_dir()
    persist_path = resolve_persist_dir_path(_persist)
    _emit_verbose(
        verbose_emitter,
        f"phase=list_collections path={persist_path!r}",
    )
    if not persist_path.exists():
        return {
            "collections": [],
            "persist_directory": str(persist_path),
        }
    client = chromadb.PersistentClient(path=str(persist_path))
    cols = client.list_collections()
    names = sorted(c.name for c in cols)
    return {
        "collections": names,
        "persist_directory": str(persist_path),
    }


def backfill_pdf_doc_titles(
    persist_dir: str | None = None,
    collection: str | None = None,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Set ``doc_title`` on PDF chunks that are missing it (metadata-only).

    Uses the PDF /Title field from chunk metadata when present, otherwise the
    filename stem (``document_id`` or ``file_name``). Does not load embedding models.
    """
    if collection is None or not str(collection).strip():
        collection = get_collection_name()
    else:
        collection = str(collection).strip()

    _persist = (persist_dir or "").strip() or get_persist_dir()
    persist_path = resolve_persist_dir_path(_persist)
    if not persist_path.exists():
        raise FileNotFoundError(f"Persistence directory does not exist: {_persist}")

    client = chromadb.PersistentClient(path=str(persist_path))
    col = client.get_collection(name=collection)
    data = col.get(include=["metadatas"])
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []

    pending: list[tuple[str, dict[str, Any]]] = []
    for i, chunk_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else None
        if not isinstance(meta, dict):
            continue
        if not _chunk_metadata_looks_like_pdf(meta):
            continue
        if str(meta.get("doc_title") or "").strip():
            continue
        pending.append((str(chunk_id), meta))

    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for chunk_id, meta in pending:
        groups[_logical_doc_key(meta)].append((chunk_id, meta))

    update_ids: list[str] = []
    update_metas: list[dict[str, Any]] = []
    for doc_key, rows in groups.items():
        title: str | None = None
        for _, m in rows:
            embedded = m.get("document_title")
            if embedded is not None and str(embedded).strip():
                title = str(embedded).strip()
                break
        if not title:
            title = Path(doc_key).stem if doc_key else "document"

        for chunk_id, m in rows:
            merged = dict(m)
            merged["doc_title"] = title
            update_ids.append(chunk_id)
            update_metas.append(merged)

    for start in range(0, len(update_ids), max(1, batch_size)):
        batch_ids = update_ids[start : start + batch_size]
        batch_meta = update_metas[start : start + batch_size]
        col.update(ids=batch_ids, metadatas=batch_meta)

    return {
        "updated_chunks": len(update_ids),
        "persist_directory": str(persist_path),
        "collection_name": collection,
    }


@traceable(name="list_documents", run_type="tool")
def list_documents(
    persist_dir: str | None = None,
    collection: str | None = None,
    tag: str | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
) -> dict[str, Any]:
    """List all indexed documents (PDF, Discord, etc.) in the PinRAG index.

    Args:
        persist_dir: Chroma persistence directory (default: from PINRAG_PERSIST_DIR or chroma_db).
        collection: Chroma collection name (default: "pinrag").
        tag: Optional tag to filter: only list documents that have this tag.
        verbose_emitter: Optional sync callback for progress messages (e.g. MCP verbose).

    Returns:
        Dictionary with "documents" (list of unique document IDs)
        and "total_chunks" (total number of chunks in the index).

    """
    if collection is None or not str(collection).strip():
        collection = get_collection_name()
    else:
        collection = str(collection).strip()

    _persist = (persist_dir or "").strip() or get_persist_dir()
    persist_path = resolve_persist_dir_path(_persist)
    if not persist_path.exists():
        return {
            "documents": [],
            "total_chunks": 0,
            "persist_directory": str(persist_path),
            "collection_name": collection,
            "document_details": {},
        }

    store = get_chroma_store(
        persist_directory=_persist,
        collection_name=collection,
    )
    tag_filter = tag.strip() if tag and str(tag).strip() else None
    _emit_verbose(
        verbose_emitter,
        f"phase=list_documents_store_loaded collection={collection!r} tag={tag_filter!r}",
    )
    data = store.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    doc_ids: set[str] = set()
    document_details: dict[str, dict[str, Any]] = {}
    doc_bytes_by_key: dict[str, dict[str, int]] = {}  # doc_id -> {source or file_name: bytes}
    chunk_count = 0
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        if tag_filter is not None:
            meta_tag = meta.get("tag")
            if meta_tag is None or str(meta_tag).strip() != tag_filter:
                continue
        chunk_count += 1
        doc_id = str(
            meta.get("document_id")
            or meta.get("file_name")
            or meta.get("source")
            or "unknown"
        )
        doc_ids.add(doc_id)
        # Aggregate doc_bytes per unique source object (GitHub: one blob URL per file)
        doc_bytes = meta.get("doc_bytes")
        if doc_bytes is not None:
            dedup = str(meta.get("source") or meta.get("file_name") or "_")
            doc_bytes_by_key.setdefault(doc_id, {})[dedup] = int(doc_bytes)
        if doc_id not in document_details:
            details: dict[str, Any] = {}
            if meta.get("document_type") is not None:
                details["document_type"] = meta["document_type"]
            if meta.get("upload_timestamp") is not None:
                details["upload_timestamp"] = meta["upload_timestamp"]
            if meta.get("doc_pages") is not None:
                details["pages"] = meta["doc_pages"]
            if meta.get("doc_messages") is not None:
                details["messages"] = meta["doc_messages"]
            if meta.get("doc_segments") is not None:
                details["segments"] = meta["doc_segments"]
            if meta.get("doc_title") is not None:
                details["title"] = meta["doc_title"]
            if meta.get("doc_total_chunks") is not None:
                details["chunks"] = meta["doc_total_chunks"]
            if meta.get("tag") is not None and str(meta.get("tag", "")).strip():
                details["tag"] = str(meta["tag"]).strip()
            if details:
                document_details[doc_id] = details

    # Set bytes from aggregated sum (total across distinct sources for multi-file docs)
    for doc_id, by_key in doc_bytes_by_key.items():
        if doc_id in document_details:
            document_details[doc_id]["bytes"] = sum(by_key.values())

    for doc_id, det in document_details.items():
        det["ref"] = doc_id
        _ensure_list_document_title(doc_id, det)

    return {
        "documents": sorted(doc_ids),
        "total_chunks": chunk_count,
        "persist_directory": str(persist_path),
        "collection_name": collection,
        "document_details": {k: document_details[k] for k in sorted(document_details)},
    }


@traceable(name="set_document_tag", run_type="tool")
def set_document_tag(
    document_id: str,
    tag: str,
    persist_dir: str = "",
    collection: str | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Set or replace the ``tag`` metadata on all chunks for one document.

    The document is resolved like ``remove_document``: exact Chroma ``document_id``,
    list title (case-insensitive), or unique PDF filename stem.

    Args:
        document_id: Ref, title, or stem identifying the document.
        tag: Tag value (non-empty after strip).
        persist_dir: Chroma persistence directory (default from env).
        collection: Chroma collection name (default: pinrag).
        verbose_emitter: Optional MCP verbose callback.
        batch_size: Chroma ``update`` batch size.

    Returns:
        Dictionary with ``document_id`` (resolved), ``tag``, ``updated_chunks``,
        ``parents_updated`` (when parent-child mode updates the docstore),
        ``persist_directory``, ``collection_name``.

    Raises:
        ValueError: If ``document_id`` or ``tag`` is empty, or resolution is ambiguous.
        FileNotFoundError: If persist directory does not exist.
    """
    if not document_id or not str(document_id).strip():
        raise ValueError("document_id cannot be empty")
    tag_norm = str(tag).strip() if tag is not None else ""
    if not tag_norm:
        raise ValueError("tag cannot be empty")

    if collection is None or not str(collection).strip():
        collection = get_collection_name()
    else:
        collection = str(collection).strip()

    _persist = (persist_dir or "").strip() or get_persist_dir()
    persist_path = resolve_persist_dir_path(_persist)
    if not persist_path.exists():
        raise FileNotFoundError(f"Persistence directory does not exist: {_persist}")

    store = get_chroma_store(
        persist_directory=_persist,
        collection_name=collection,
    )
    requested = document_id.strip()
    resolved = _resolve_remove_document_id(store, requested)
    if resolved != requested:
        _emit_verbose(
            verbose_emitter,
            f"phase=set_document_tag_resolved requested={requested!r} document_id={resolved!r}",
        )
    _emit_verbose(
        verbose_emitter,
        f"phase=set_document_tag_start document_id={resolved!r} tag={tag_norm!r} collection={collection!r}",
    )

    data = store.get(
        where={"document_id": resolved},
        include=["metadatas"],
    )
    chunk_ids: list[str] = [str(x) for x in (data.get("ids") or [])]
    metas_raw = data.get("metadatas") or []

    if not chunk_ids:
        _emit_verbose(
            verbose_emitter,
            f"phase=set_document_tag_done document_id={resolved!r} updated_chunks=0",
        )
        return {
            "document_id": resolved,
            "tag": tag_norm,
            "updated_chunks": 0,
            "parents_updated": 0,
            "persist_directory": str(persist_path),
            "collection_name": collection,
        }

    update_ids: list[str] = []
    update_metas: list[dict[str, Any]] = []
    for i, chunk_id in enumerate(chunk_ids):
        raw = metas_raw[i] if i < len(metas_raw) else None
        if not isinstance(raw, dict):
            raw = {}
        merged: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
        merged["tag"] = tag_norm
        update_ids.append(chunk_id)
        update_metas.append(merged)

    col = store._collection
    for start in range(0, len(update_ids), max(1, batch_size)):
        batch_ids = update_ids[start : start + batch_size]
        batch_meta = update_metas[start : start + batch_size]
        col.update(ids=batch_ids, metadatas=batch_meta)

    parents_updated = 0
    if get_use_parent_child():
        parent_ids = sorted(
            {
                str(m["doc_id"])
                for m in metas_raw
                if isinstance(m, dict) and m.get("doc_id")
            }
        )
        if parent_ids:
            docstore = get_parent_docstore(_persist, collection)
            loaded = docstore.mget(parent_ids)
            pairs: list[tuple[str, Any]] = []
            for pid, doc in zip(parent_ids, loaded, strict=True):
                if doc is None:
                    continue
                new_meta = dict(doc.metadata)
                new_meta["tag"] = tag_norm
                pairs.append(
                    (pid, Document(page_content=doc.page_content, metadata=new_meta))
                )
            if pairs:
                docstore.mset(pairs)
                parents_updated = len(pairs)

    _emit_verbose(
        verbose_emitter,
        f"phase=set_document_tag_done document_id={resolved!r} "
        f"updated_chunks={len(chunk_ids)} parents_updated={parents_updated}",
    )

    return {
        "document_id": resolved,
        "tag": tag_norm,
        "updated_chunks": len(chunk_ids),
        "parents_updated": parents_updated,
        "persist_directory": str(persist_path),
        "collection_name": collection,
    }


@traceable(name="remove_document", run_type="tool")
def remove_document(
    document_id: str,
    persist_dir: str = "",
    collection: str | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
) -> dict[str, Any]:
    """Remove a document and all its chunks and embeddings from the Chroma index.

    The argument may be ``document_details.ref`` (exact Chroma ``document_id``) or,
    if unique, the same **title** shown in list_documents (case-insensitive), or the
    **stem** of a PDF filename (e.g. ``Handbook`` for ``Handbook.pdf``). Deletes
    child chunks from Chroma and, when parent-child retrieval is enabled, parent
    chunks from the docstore.

    Args:
        document_id: Ref and/or title to remove (see above).
        persist_dir: Chroma persistence directory (default: "chroma_db").
        collection: Chroma collection name (default: "pinrag").
        verbose_emitter: Optional sync callback for progress messages (e.g. MCP verbose).

    Returns:
        Dictionary with "deleted_chunks" (int), "document_id" (str),
        "persist_directory", "collection_name".

    Raises:
        ValueError: If document_id is empty or collection is empty.
        FileNotFoundError: If persist_dir does not exist.

    """
    if not document_id or not str(document_id).strip():
        raise ValueError("document_id cannot be empty")
    if collection is None or not str(collection).strip():
        collection = get_collection_name()
    else:
        collection = str(collection).strip()

    _persist = (persist_dir or "").strip() or get_persist_dir()
    persist_path = resolve_persist_dir_path(_persist)
    if not persist_path.exists():
        raise FileNotFoundError(f"Persistence directory does not exist: {_persist}")

    store = get_chroma_store(
        persist_directory=_persist,
        collection_name=collection,
    )
    requested = document_id.strip()
    resolved = _resolve_remove_document_id(store, requested)
    if resolved != requested:
        _emit_verbose(
            verbose_emitter,
            f"phase=remove_document_resolved requested={requested!r} document_id={resolved!r}",
        )
    _emit_verbose(
        verbose_emitter,
        f"phase=remove_document_start document_id={resolved!r} collection={collection!r}",
    )

    # Get chunks matching this document_id (need metadatas for parent doc_ids when parent-child)
    data = store.get(
        where={"document_id": resolved},
        include=["metadatas"] if get_use_parent_child() else [],
    )
    ids = data.get("ids") or []
    deleted_count = len(ids)

    # When parent-child is enabled, also delete parent chunks from docstore
    if get_use_parent_child() and ids:
        metadatas = data.get("metadatas") or []
        parent_ids = set()
        for meta in metadatas:
            if isinstance(meta, dict) and meta.get("doc_id"):
                parent_ids.add(str(meta["doc_id"]))
        if parent_ids:
            # Defensive check: keep parent docs still referenced by other documents.
            safe_to_delete: list[str] = []
            target_doc = resolved
            for pid in parent_ids:
                refs = store.get(where={"doc_id": pid}, include=["metadatas"])
                ref_metas = refs.get("metadatas") or []
                referenced_elsewhere = any(
                    isinstance(m, dict) and str(m.get("document_id", "")) != target_doc
                    for m in ref_metas
                )
                if not referenced_elsewhere:
                    safe_to_delete.append(pid)
            docstore = get_parent_docstore(_persist, collection)
            if safe_to_delete:
                docstore.mdelete(safe_to_delete)

    if ids:
        store.delete(where={"document_id": resolved})
    _emit_verbose(
        verbose_emitter,
        f"phase=remove_document_done document_id={resolved!r} deleted_chunks={deleted_count}",
    )

    return {
        "deleted_chunks": deleted_count,
        "document_id": resolved,
        "persist_directory": str(persist_path),
        "collection_name": collection,
    }
