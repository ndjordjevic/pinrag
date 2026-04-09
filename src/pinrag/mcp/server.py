"""MCP server setup using FastMCP."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

import anyio
from mcp import types
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from pinrag import __version__
from pinrag import config
from pinrag.mcp.logging_utils import (
    _emit_client_log,
    _log_tool_errors,
    configure_logging,
    make_verbose_emitter,
    mirror_verbose_to_output_panel,
)
from pinrag.mcp.resource_text import format_documents_list, format_server_config
from pinrag.mcp.tools import (
    add_files,
    list_collections,
    list_documents,
    remove_document,
    set_document_tag,
)
from pinrag.mcp.tools import (
    query as query_index,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("PinRAG", json_response=True)


def emit_server_runtime_stderr() -> None:
    """Print resolved Chroma path and LLM settings to stderr at process start.

    Uses stderr so output appears in editors and next to uvicorn logs; the
    ``pinrag`` logger is intentionally quiet after :func:`configure_logging`.
    """
    persist = Path(config.get_persist_dir()).expanduser()
    try:
        persist_display = str(persist.resolve())
    except OSError:
        persist_display = str(persist)
    collection = config.get_collection_name()
    provider = config.get_llm_provider()
    model = config.get_llm_model()
    sys.stderr.write(
        f"PinRAG store: {persist_display}  collection: {collection}\n"
        f"PinRAG LLM: {provider}  model: {model}\n"
    )
    sys.stderr.flush()


def _attach_server_metadata(
    result: dict, *, verbose_lines: list[str] | None = None
) -> dict:
    """Attach stable server metadata (and optional verbose lines) to tool results."""
    result["_server_version"] = __version__
    if verbose_lines:
        result["_verbose_log"] = verbose_lines
    return result


def _drain_verbose_lines(verbose_emitter: object) -> list[str]:
    """Drain buffered verbose lines from emitter collector (if present)."""
    collector = getattr(verbose_emitter, "_collector", None)
    if collector is None:
        return []
    return collector.drain()


def _collection_kwarg(collection: str) -> str | None:
    """Return explicit collection name or None to use PINRAG_COLLECTION_NAME."""
    c = (collection or "").strip()
    return c if c else None


@mcp._mcp_server.set_logging_level()
async def _handle_set_logging_level(level: types.LoggingLevel) -> None:
    """Handle MCP ``logging/setLevel`` to declare logging capability."""
    _ = level


# Re-export for pinrag.cli (F401: name used via import from this module).
__all__ = ["configure_logging", "emit_server_runtime_stderr", "mcp"]


@mcp.tool()
@_log_tool_errors
async def query_tool(
    query: Annotated[str, Field(description="Natural language question to ask.")],
    document_id: Annotated[
        str,
        Field(
            description=(
                "Optional document filter: exact ref from list_documents, exact list title, "
                "or unique PDF stem (same rules as remove_document)."
            ),
        ),
    ] = "",
    page_min: Annotated[
        int | None,
        Field(description="Optional start of page range (inclusive). PDF only."),
    ] = None,
    page_max: Annotated[
        int | None,
        Field(description="Optional end of page range (inclusive). PDF only."),
    ] = None,
    tag: Annotated[
        str,
        Field(description="Optional tag to filter retrieval (from list_documents)."),
    ] = "",
    document_type: Annotated[
        str,
        Field(
            description="Optional type to filter: 'pdf', 'youtube', 'discord', 'github', or 'plaintext'."
        ),
    ] = "",
    response_style: Annotated[
        str,
        Field(
            description=(
                "Answer style: 'thorough' (detailed) or 'concise'. "
                "Leave empty to use PINRAG_RESPONSE_STYLE."
            ),
        ),
    ] = "",
    collection: Annotated[
        str,
        Field(
            description=(
                "Optional Chroma collection name (default: PINRAG_COLLECTION_NAME)."
            ),
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """Query indexed documents and return an answer with citations.

    Searches through all indexed documents (PDF, Discord, etc.) and uses RAG
    to provide an answer based on retrieved context, with source citations.

    Args:
        query: Natural language question to ask.
        document_id: Optional document selector (ref, title, or unique PDF stem).
        page_min: Optional start of page range (inclusive). PDF only.
        page_max: Optional end of page range (inclusive). PDF only.
        tag: Optional tag to filter retrieval (from list_documents).
        document_type: Optional type to filter: "pdf", "youtube", "discord", "github", or "plaintext".
        response_style: Answer style; omit or empty string uses PINRAG_RESPONSE_STYLE.
        ctx: MCP request context (injected by the server).

    Returns:
        Dictionary containing answer and sources (document_id, page).

    """
    style_input = (response_style or "").strip().lower()
    if style_input in ("thorough", "concise"):
        style = style_input
    else:
        style = config.get_response_style()

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="query_tool")

    if ctx is not None:
        await ctx.report_progress(0, 100, message="retrieving…")

    # phase_callback is called from the worker thread after retrieval completes.
    # We schedule report_progress on the running event loop via run_coroutine_threadsafe.
    _loop = asyncio.get_running_loop()

    def _phase_callback(phase: str) -> None:
        if ctx is None:
            return
        if phase == "generating":
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(50, 100, message="generating answer…"),
                _loop,
            )

    def _run() -> dict:
        return query_index(
            user_query=query,
            document_id=document_id or None,
            page_min=page_min,
            page_max=page_max,
            tag=tag or None,
            document_type=document_type or None,
            response_style=style,
            collection=_collection_kwarg(collection),
            verbose_emitter=verbose_emitter,
            phase_callback=_phase_callback,
        )

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception:
        mirror_verbose_to_output_panel(_drain_verbose_lines(verbose_emitter))
        raise

    if ctx is not None:
        await ctx.report_progress(100, 100, message="done")

    verbose_lines = _drain_verbose_lines(verbose_emitter)
    mirror_verbose_to_output_panel(verbose_lines)
    if isinstance(result, dict):
        return _attach_server_metadata(result, verbose_lines=verbose_lines)
    return result


@mcp.tool()
@_log_tool_errors
async def add_document_tool(
    paths: Annotated[
        list[str],
        Field(
            description='Paths to index: file, directory, YouTube URL, or GitHub URL (e.g. https://github.com/owner/repo). Single path: ["/path/to/file.pdf"] or ["https://github.com/owner/repo"].'
        ),
    ],
    tags: Annotated[
        list[str] | None,
        Field(description="Optional list of tags, one per path (same order as paths)."),
    ] = None,
    branch: Annotated[
        str | None,
        Field(
            description="For GitHub URLs: override branch (default: main). Ignored for other formats."
        ),
    ] = None,
    include_patterns: Annotated[
        list[str] | None,
        Field(
            description='For GitHub URLs: glob patterns for files to include (e.g. ["*.md", "src/**/*.py"]). Ignored for other formats.'
        ),
    ] = None,
    exclude_patterns: Annotated[
        list[str] | None,
        Field(
            description="For GitHub URLs: glob patterns to exclude. Ignored for other formats."
        ),
    ] = None,
    collection: Annotated[
        str,
        Field(
            description=(
                "Optional Chroma collection name (default: PINRAG_COLLECTION_NAME)."
            ),
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    r"""Add files, directories, YouTube videos, or GitHub repos to the index.

    Automatically detects format per path and indexes:
    - GitHub (URL, e.g. https://github.com/owner/repo or github.com/owner/repo/tree/branch)
    - YouTube (URL or video ID, e.g. https://youtu.be/xyz)
    - PDF (.pdf)
    - Discord export (.txt with DiscordChatExporter Guild:/Channel: header)

    Pass one or more paths. Single path: paths=[\"/path/to/file.pdf\"]. Uses
    server config for vector store location and collection. Returns both
    successful and failed entries so one bad path does not fail the batch.

    Args:
        paths: List of file paths, directory paths, YouTube URLs, or GitHub URLs to index.
        tags: Optional list of tags, one per path (same order as paths).
        branch: For GitHub URLs: override branch (default: main). Ignored for other formats.
        include_patterns: For GitHub URLs: glob patterns for files to include (e.g. ["*.md", "src/**/*.py"]).
        exclude_patterns: For GitHub URLs: glob patterns to exclude. Ignored for other formats.
        ctx: MCP request context (injected by the server; unused).

    Returns:
        Dictionary containing indexed entries, failed entries, and totals.

    """

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="add_document_tool")

    def _run() -> dict:
        return add_files(
            paths=paths,
            persist_dir=config.get_persist_dir(),
            collection=_collection_kwarg(collection),
            tags=tags,
            branch=branch,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            verbose_emitter=verbose_emitter,
        )

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception:
        mirror_verbose_to_output_panel(_drain_verbose_lines(verbose_emitter))
        raise

    verbose_lines = _drain_verbose_lines(verbose_emitter)
    mirror_verbose_to_output_panel(verbose_lines)
    if isinstance(result, dict):
        return _attach_server_metadata(result, verbose_lines=verbose_lines)
    return result


@mcp.tool()
@_log_tool_errors
async def list_documents_tool(
    tag: Annotated[
        str,
        Field(
            description="Optional tag to filter: only list documents that have this tag."
        ),
    ] = "",
    collection: Annotated[
        str,
        Field(
            description=(
                "Optional Chroma collection name (default: PINRAG_COLLECTION_NAME)."
            ),
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """List all indexed documents in the PinRAG index.

    Returns unique document IDs (PDF file names, video IDs, discord-alicia-1200-pcb, owner/repo/path for GitHub, etc.)
    currently in the vector store, plus total chunk count. Uses server config
    for vector store location and collection.

    Args:
        tag: Optional tag to filter: only list documents that have this tag.
        ctx: MCP request context (injected by the server; unused).

    Returns:
        Dictionary containing documents, total_chunks, persist_directory,
        collection_name, document_details.

    """

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="list_documents_tool")

    def _run() -> dict:
        return list_documents(
            persist_dir=config.get_persist_dir(),
            collection=_collection_kwarg(collection),
            tag=tag or None,
            verbose_emitter=verbose_emitter,
        )

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception:
        mirror_verbose_to_output_panel(_drain_verbose_lines(verbose_emitter))
        raise

    verbose_lines = _drain_verbose_lines(verbose_emitter)
    mirror_verbose_to_output_panel(verbose_lines)
    if isinstance(result, dict):
        return _attach_server_metadata(result, verbose_lines=verbose_lines)
    return result


@mcp.tool()
@_log_tool_errors
async def list_collections_tool(
    persist_dir: Annotated[
        str,
        Field(
            description=(
                "Optional persist directory override (default: PINRAG_PERSIST_DIR)."
            ),
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """List Chroma collection names in the configured (or overridden) persist directory."""

    verbose_emitter = make_verbose_emitter(
        ctx, scope="tool", name="list_collections_tool"
    )

    def _run() -> dict:
        pd = (persist_dir or "").strip() or None
        return list_collections(persist_dir=pd, verbose_emitter=verbose_emitter)

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception:
        mirror_verbose_to_output_panel(_drain_verbose_lines(verbose_emitter))
        raise

    verbose_lines = _drain_verbose_lines(verbose_emitter)
    mirror_verbose_to_output_panel(verbose_lines)
    if isinstance(result, dict):
        return _attach_server_metadata(result, verbose_lines=verbose_lines)
    return result


@mcp.tool()
@_log_tool_errors
async def remove_document_tool(
    document_id: Annotated[
        str,
        Field(
            description=(
                "Document ref from list_documents_tool, or the list title / PDF filename stem "
                "if it uniquely identifies one document."
            ),
        ),
    ],
    collection: Annotated[
        str,
        Field(
            description=(
                "Optional Chroma collection name (default: PINRAG_COLLECTION_NAME)."
            ),
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """Remove a document and all its chunks from the PinRAG index.

    Deletes all chunks and embeddings for the given document. Use
    list_documents_tool to see document_ids (e.g. "mybook.pdf", "bwgLXEQdq20", "discord-alicia-1200-pcb", "owner/repo/path" for GitHub).
    Uses server config for vector store location and collection.

    Args:
        document_id: Document identifier to remove (from list_documents_tool).
        ctx: MCP request context (injected by the server; unused).

    Returns:
        Dictionary containing deleted_chunks, document_id (canonical ref after any title match),
        persist_directory, collection_name.

    """

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="remove_document_tool")

    def _run() -> dict:
        return remove_document(
            document_id=document_id,
            persist_dir=config.get_persist_dir(),
            collection=_collection_kwarg(collection),
            verbose_emitter=verbose_emitter,
        )

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception:
        mirror_verbose_to_output_panel(_drain_verbose_lines(verbose_emitter))
        raise

    verbose_lines = _drain_verbose_lines(verbose_emitter)
    mirror_verbose_to_output_panel(verbose_lines)
    if isinstance(result, dict):
        return _attach_server_metadata(result, verbose_lines=verbose_lines)
    return result


@mcp.tool()
@_log_tool_errors
async def set_document_tag_tool(
    document_id: Annotated[
        str,
        Field(
            description=(
                "Document ref from list_documents_tool, or list title / PDF stem "
                "if it uniquely identifies one document."
            ),
        ),
    ],
    tag: Annotated[
        str,
        Field(description="Tag to set on all chunks (and parent docs when parent-child is enabled)."),
    ],
    collection: Annotated[
        str,
        Field(
            description=(
                "Optional Chroma collection name (default: PINRAG_COLLECTION_NAME)."
            ),
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """Set or replace the tag on every indexed chunk for one document."""

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="set_document_tag_tool")

    def _run() -> dict:
        return set_document_tag(
            document_id=document_id,
            tag=tag,
            persist_dir=config.get_persist_dir(),
            collection=_collection_kwarg(collection),
            verbose_emitter=verbose_emitter,
        )

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception:
        mirror_verbose_to_output_panel(_drain_verbose_lines(verbose_emitter))
        raise

    verbose_lines = _drain_verbose_lines(verbose_emitter)
    mirror_verbose_to_output_panel(verbose_lines)
    if isinstance(result, dict):
        return _attach_server_metadata(result, verbose_lines=verbose_lines)
    return result


@mcp.resource(
    "pinrag://documents",
    title="Indexed documents list",
    description="Read-only list of documents currently indexed in PinRAG (default collection).",
)
async def documents_resource() -> str:
    """Return a plain-text list of indexed documents for the default collection.

    For PDFs: shows page count. For Discord: shows size (messages or bytes).
    For YouTube: shows video title when available, with video ID in parentheses.
    Shows tag when attached to a document.

    Must remain parameterless so FastMCP registers a static resource (``resources/list``).
    A ``ctx: Context`` parameter would register only a template, which Cursor does not list.
    """
    ctx = mcp.get_context()
    await _emit_client_log(ctx, "resource=pinrag://documents status=start")
    logger.debug("Resource pinrag://documents read")
    try:
        body = await anyio.to_thread.run_sync(format_documents_list)
        await _emit_client_log(ctx, "resource=pinrag://documents status=ok")
        return body
    except Exception as e:
        await _emit_client_log(ctx, "resource=pinrag://documents status=error", "error")
        logger.exception("Resource pinrag://documents failed")
        return f"Error listing documents: {e}"


@mcp.resource(
    "pinrag://server-config",
    title="Server configuration",
    description="Environment variables and config params used by the PinRAG MCP server.",
)
async def server_config_resource() -> str:
    """Return plain-text summary: env vars that are set (top), defaults (bottom)."""
    ctx = mcp.get_context()
    await _emit_client_log(ctx, "resource=pinrag://server-config status=start")
    logger.debug("Resource pinrag://server-config read")
    body = await anyio.to_thread.run_sync(format_server_config)
    await _emit_client_log(ctx, "resource=pinrag://server-config status=ok")
    return body


@mcp.prompt()
def use_pinrag(request: str = "") -> str:
    """Use PinRAG to query, index, list, or remove documents.

    Routes to the correct tool based on the request:
    - Query / question  → query_tool
    - Index / add       → add_document_tool
    - List / show       → list_documents_tool
    - Collections       → list_collections_tool
    - Remove / delete   → remove_document_tool
    - Tag / label       → set_document_tag_tool
    """
    return (
        f"Use PinRAG to fulfil this request: {request or 'not specified'}.\n\n"
        "Available tools and when to use them:\n\n"
        "query_tool — answer a question from indexed documents.\n"
        "  Required: query (str). "
        "  Optional: document_id (ref, exact list title, or unique PDF stem, like remove_document_tool), "
        "page_min/page_max (PDF page range), tag (filter by tag), "
        "document_type ('pdf', 'youtube', 'discord', 'github', 'plaintext'), "
        "response_style ('thorough' or 'concise'), "
        "collection (Chroma collection; default from PINRAG_COLLECTION_NAME).\n\n"
        "add_document_tool — index local files, directories, or remote URLs.\n"
        "  Required: paths (list of str: file paths, dir paths, YouTube URLs/IDs, GitHub URLs). "
        "  Optional: tags (list, one per path), branch (GitHub only), "
        "include_patterns / exclude_patterns (GitHub only), collection.\n\n"
        "list_documents_tool — list all indexed documents and chunk counts.\n"
        "  Optional: tag (filter by tag), collection.\n\n"
        "list_collections_tool — list Chroma collection names in the persist directory.\n"
        "  Optional: persist_dir (default from PINRAG_PERSIST_DIR).\n\n"
        "remove_document_tool — remove a document and all its chunks.\n"
        "  Required: document_id (ref from list_documents, exact list title, or unique PDF stem).\n"
        "  Optional: collection.\n\n"
        "set_document_tag_tool — set or replace the tag for one document (all chunks).\n"
        "  Required: document_id, tag (non-empty).\n"
        "  Optional: collection.\n\n"
        "If the request is ambiguous, call list_documents_tool first to show what is indexed, "
        "then decide which tool to use."
    )
