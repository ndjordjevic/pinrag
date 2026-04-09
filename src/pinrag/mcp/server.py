"""MCP server setup using FastMCP."""

from __future__ import annotations

import logging
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
    list_documents,
    remove_document,
)
from pinrag.mcp.tools import (
    query as query_index,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("PinRAG", json_response=True)


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


@mcp._mcp_server.set_logging_level()
async def _handle_set_logging_level(level: types.LoggingLevel) -> None:
    """Handle MCP ``logging/setLevel`` to declare logging capability."""
    _ = level


# Re-export for pinrag.cli (F401: name used via import from this module).
__all__ = ["configure_logging", "mcp"]


@mcp.tool()
@_log_tool_errors
async def query_tool(
    query: Annotated[str, Field(description="Natural language question to ask.")],
    document_id: Annotated[
        str,
        Field(
            description="Optional document ID to filter retrieval (from list_documents)."
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
    ctx: Context | None = None,
) -> dict:
    """Query indexed documents and return an answer with citations.

    Searches through all indexed documents (PDF, Discord, etc.) and uses RAG
    to provide an answer based on retrieved context, with source citations.

    Args:
        query: Natural language question to ask.
        document_id: Optional document ID to filter retrieval (from list_documents).
        page_min: Optional start of page range (inclusive). PDF only.
        page_max: Optional end of page range (inclusive). PDF only.
        tag: Optional tag to filter retrieval (from list_documents).
        document_type: Optional type to filter: "pdf", "youtube", "discord", "github", or "plaintext".
        response_style: Answer style; omit or empty string uses PINRAG_RESPONSE_STYLE.
        ctx: MCP request context (injected by the server; unused).

    Returns:
        Dictionary containing answer and sources (document_id, page).

    """
    style_input = (response_style or "").strip().lower()
    if style_input in ("thorough", "concise"):
        style = style_input
    else:
        style = config.get_response_style()

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="query_tool")

    def _run() -> dict:
        return query_index(
            user_query=query,
            document_id=document_id or None,
            page_min=page_min,
            page_max=page_max,
            tag=tag or None,
            document_type=document_type or None,
            response_style=style,
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
            collection=config.get_collection_name(),
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
            collection=config.get_collection_name(),
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
async def remove_document_tool(
    document_id: Annotated[
        str,
        Field(description="Document identifier to remove (from list_documents_tool)."),
    ],
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
        Dictionary containing deleted_chunks, document_id, persist_directory, collection_name.

    """

    verbose_emitter = make_verbose_emitter(ctx, scope="tool", name="remove_document_tool")

    def _run() -> dict:
        return remove_document(
            document_id=document_id,
            persist_dir=config.get_persist_dir(),
            collection=config.get_collection_name(),
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
    - Remove / delete   → remove_document_tool
    """
    return (
        f"Use PinRAG to fulfil this request: {request or 'not specified'}.\n\n"
        "Available tools and when to use them:\n\n"
        "query_tool — answer a question from indexed documents.\n"
        "  Required: query (str). "
        "  Optional: document_id (filter to one doc, from list_documents_tool), "
        "page_min/page_max (PDF page range), tag (filter by tag), "
        "document_type ('pdf', 'youtube', 'discord', 'github', 'plaintext'), "
        "response_style ('thorough' or 'concise').\n\n"
        "add_document_tool — index local files, directories, or remote URLs.\n"
        "  Required: paths (list of str: file paths, dir paths, YouTube URLs/IDs, GitHub URLs). "
        "  Optional: tags (list, one per path), branch (GitHub only), "
        "include_patterns / exclude_patterns (GitHub only).\n\n"
        "list_documents_tool — list all indexed documents and chunk counts.\n"
        "  Optional: tag (filter by tag).\n\n"
        "remove_document_tool — remove a document and all its chunks.\n"
        "  Required: document_id (get exact ID from list_documents_tool first).\n\n"
        "If the request is ambiguous, call list_documents_tool first to show what is indexed, "
        "then decide which tool to use."
    )
