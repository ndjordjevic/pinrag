"""Transport-agnostic query, add, list, and remove operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

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
from pinrag.rag import run_rag
from pinrag.vectorstore import get_chroma_store
from pinrag.vectorstore.docstore import get_parent_docstore

logger = logging.getLogger(__name__)
VerboseSyncEmitter = Callable[[str, str], None]


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
) -> dict[str, Any]:
    """Query indexed documents (PDF, Discord) and return an answer with citations.

    Retrieval is driven by .env: PINRAG_RETRIEVE_K, rerank, multi-query, etc.
    Persist dir and collection come from persist_dir/collection params or
    PINRAG_PERSIST_DIR and PINRAG_COLLECTION_NAME when not provided.

    Args:
        user_query: Natural language question to ask.
        document_id: Optional document ID to filter retrieval (e.g. from list_documents).
        page_min: Optional start of page range (inclusive). Use with page_max. PDF only.
        page_max: Optional end of page range (inclusive). Single page: page_min=64, page_max=64. PDF only.
        tag: Optional tag to filter retrieval (e.g. from list_documents document_details).
        document_type: Optional type to filter: "pdf", "youtube", "discord", "github", or "plaintext".
        response_style: Answer style for generation ("thorough" or "concise").
        persist_dir: Chroma persistence directory (default: from PINRAG_PERSIST_DIR or chroma_db).
        collection: Chroma collection name (default: from PINRAG_COLLECTION_NAME or pinrag).
        verbose_emitter: Optional sync callback for progress messages (e.g. MCP verbose).

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
    tag_filter = tag.strip() if tag and str(tag).strip() else None
    doc_type_filter = (
        document_type.strip() if document_type and str(document_type).strip() else None
    )
    rag_result = run_rag(
        user_query,
        llm,
        k=None,
        persist_directory=str(persist_path),
        collection_name=_collection,
        embedding=embedding,
        document_id=doc_id_filter,
        page_min=page_min,
        page_max=page_max,
        tag=tag_filter,
        document_type=doc_type_filter,
        response_style=response_style,
    )
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

    return {
        "documents": sorted(doc_ids),
        "total_chunks": chunk_count,
        "persist_directory": str(persist_path),
        "collection_name": collection,
        "document_details": {k: document_details[k] for k in sorted(document_details)},
    }


@traceable(name="remove_document", run_type="tool")
def remove_document(
    document_id: str,
    persist_dir: str = "",
    collection: str | None = None,
    verbose_emitter: VerboseSyncEmitter | None = None,
) -> dict[str, Any]:
    """Remove a document and all its chunks and embeddings from the Chroma index.

    The document_id must match exactly the name shown in list_documents (e.g.
    "mybook.pdf" or "discord-alicia-1200-pcb"). Deletes child chunks from Chroma
    and, when parent-child retrieval is enabled, parent chunks from the docstore.

    Args:
        document_id: Document identifier to remove (same as in list_documents).
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
    _emit_verbose(
        verbose_emitter,
        f"phase=remove_document_start document_id={document_id.strip()!r} collection={collection!r}",
    )

    # Get chunks matching this document_id (need metadatas for parent doc_ids when parent-child)
    data = store.get(
        where={"document_id": document_id.strip()},
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
            target_doc = document_id.strip()
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
        store.delete(where={"document_id": document_id.strip()})
    _emit_verbose(
        verbose_emitter,
        f"phase=remove_document_done document_id={document_id.strip()!r} deleted_chunks={deleted_count}",
    )

    return {
        "deleted_chunks": deleted_count,
        "document_id": document_id.strip(),
        "persist_directory": str(persist_path),
        "collection_name": collection,
    }
