"""CLI entry point for PinRAG MCP server."""

from __future__ import annotations

import sys

import typer

from pinrag import __version__
from pinrag.core.operations import backfill_pdf_doc_titles
from pinrag.env_validation import require_api_keys_for_server
from pinrag.mcp.server import configure_logging, emit_server_runtime_stderr, mcp

app = typer.Typer(
    help="PinRAG MCP: default runs stdio transport; use `server` for HTTP.",
    invoke_without_command=True,
)


def _run_stdio() -> None:
    """Run the MCP server with stdio transport (default for editors)."""
    configure_logging()
    require_api_keys_for_server()
    sys.stderr.write(f"PinRAG MCP v{__version__}\n")
    sys.stderr.flush()
    emit_server_runtime_stderr()
    mcp.run(transport="stdio")


@app.callback(invoke_without_command=True)
def _callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _run_stdio()


@app.command("backfill-pdf-titles")
def _backfill_pdf_titles(
    persist_dir: str | None = typer.Option(
        None,
        "--persist-dir",
        envvar="PINRAG_PERSIST_DIR",
        help="Chroma persistence directory (default: PINRAG_PERSIST_DIR or chroma_db).",
    ),
    collection: str | None = typer.Option(
        None,
        "--collection",
        envvar="PINRAG_COLLECTION_NAME",
        help="Collection name (default: PINRAG_COLLECTION_NAME or pinrag).",
    ),
) -> None:
    """Set missing doc_title on PDF chunks (filename stem or PDF /Title metadata)."""
    try:
        out = backfill_pdf_doc_titles(persist_dir=persist_dir or "", collection=collection)
    except FileNotFoundError as e:
        sys.stderr.write(f"pinrag backfill-pdf-titles: {e}\n")
        raise typer.Exit(1) from e
    except Exception as e:
        sys.stderr.write(f"pinrag backfill-pdf-titles: {e}\n")
        raise typer.Exit(1) from e
    sys.stderr.write(
        f"Updated {out['updated_chunks']} PDF chunk(s) with doc_title — "
        f"{out['persist_directory']} collection={out['collection_name']!r}\n"
    )


@app.command("server")
def _server(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address for streamable-http.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        help="Listen port for streamable-http.",
    ),
) -> None:
    """Run MCP server with streamable-http (e.g. for pinrag-cli).

    Endpoint: http://<host>:<port>/mcp
    """
    configure_logging()
    require_api_keys_for_server()
    sys.stderr.write(
        f"PinRAG MCP v{__version__} streamable-http http://{host}:{port}/mcp\n"
    )
    sys.stderr.flush()
    emit_server_runtime_stderr()
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


def main() -> None:
    """Console script entry: ``pinrag`` or ``uv run pinrag``."""
    app()
