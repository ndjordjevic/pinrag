"""Unit tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pinrag.cli import app


def test_main_exits_without_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI exits with code 1 when no LLM API key (default OpenRouter)."""
    monkeypatch.delenv("PINRAG_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, [], catch_exceptions=False)
    assert result.exit_code == 1


def test_main_runs_mcp_stdio_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default invocation runs MCP server with stdio transport."""
    monkeypatch.delenv("PINRAG_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mock_mcp = MagicMock()
    with patch("pinrag.cli.configure_logging"):
        with patch("pinrag.cli.mcp", mock_mcp):
            with patch("pinrag.cli.__version__", "9.9.9-test"):
                runner = CliRunner()
                result = runner.invoke(app, [], catch_exceptions=False)
    assert result.exit_code == 0
    mock_mcp.run.assert_called_once_with(transport="stdio")


def test_server_subcommand_runs_streamable_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """`pinrag server` uses streamable-http and applies host/port to FastMCP settings."""
    monkeypatch.delenv("PINRAG_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    mock_mcp = MagicMock()
    mock_mcp.settings = MagicMock()
    with patch("pinrag.cli.configure_logging"):
        with patch("pinrag.cli.mcp", mock_mcp):
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["server", "--host", "0.0.0.0", "--port", "9999"],
                catch_exceptions=False,
            )
    assert result.exit_code == 0
    assert mock_mcp.settings.host == "0.0.0.0"
    assert mock_mcp.settings.port == 9999
    mock_mcp.run.assert_called_once_with(transport="streamable-http")


def test_backfill_pdf_titles_subcommand_skips_llm_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """backfill-pdf-titles does not require LLM API keys."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    client.get_or_create_collection("pinrag")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["backfill-pdf-titles", "--persist-dir", str(tmp_path), "--collection", "pinrag"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Updated" in (result.stderr or "")
