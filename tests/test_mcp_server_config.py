"""Tests for MCP server config resource, logging, remove_document, and schemas."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pinrag.mcp import resource_text
from pinrag.mcp.tools import remove_document
from tests.helpers.mcp_patched_server import mcp_logging, mcp_server

# --- server_config_resource ---


def test_server_config_resource_includes_api_key_status() -> None:
    """server_config_resource shows API key set/not set, never the values."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-secret"}, clear=False):
        out = asyncio.run(mcp_server.server_config_resource())
    assert "OPENAI_API_KEY: set" in out
    assert "sk-secret" not in out
    assert "OPENROUTER_API_KEY:" in out
    assert "CEREBRAS_API_KEY:" in out
    assert "ANTHROPIC_API_KEY:" in out


def test_server_config_resource_includes_effective_config() -> None:
    """server_config_resource includes effective config from get_* functions."""
    with patch("pinrag.config.get_persist_dir", return_value="/my/chroma"):
        with patch("pinrag.config.get_collection_name", return_value="my_coll"):
            out = asyncio.run(mcp_server.server_config_resource())
    assert "PINRAG_VERSION:" in out
    assert "PINRAG_PERSIST_DIR: /my/chroma" in out
    assert "PINRAG_COLLECTION_NAME: my_coll" in out
    assert "PINRAG_VERBOSE_LOGGING:" in out
    assert "--- Explicitly set (runtime env) ---" in out
    assert "--- Defaults (not set in env) ---" in out
    assert "--- API keys (status only) ---" in out
    assert "--- Optional: OpenRouter attribution & OpenAI / Cerebras client ---" in out
    assert "PINRAG_CEREBRAS_BASE_URL (effective):" in out
    assert "OPENROUTER_APP_URL (effective):" in out
    assert "PINRAG_EVALUATOR_PROVIDER:" in out


def test_server_config_resource_explicitly_set_in_runtime_section() -> None:
    """Vars set in os.environ appear under Explicitly set (runtime env)."""
    with patch.dict(
        "os.environ",
        {"PINRAG_PERSIST_DIR": "/tmp/pinrag-db", "PINRAG_VERBOSE_LOGGING": "true"},
        clear=False,
    ):
        out = asyncio.run(mcp_server.server_config_resource())
    assert "PINRAG_PERSIST_DIR: /tmp/pinrag-db" in out
    assert "PINRAG_VERBOSE_LOGGING: true" in out
    assert "--- Explicitly set (runtime env) ---" in out
    # Set var appears before Defaults section
    set_section = out.split("--- Defaults (not set in env) ---")[0]
    assert "PINRAG_PERSIST_DIR: /tmp/pinrag-db" in set_section
    assert "PINRAG_VERBOSE_LOGGING: true" in set_section


def test_server_config_resource_shows_set_vs_default() -> None:
    """server_config_resource splits set vars (Explicitly set) vs unset (Defaults)."""
    with patch.dict(
        "os.environ",
        {"PINRAG_PERSIST_DIR": "/custom/db"},
        clear=False,
    ):
        out = asyncio.run(mcp_server.server_config_resource())
    assert "PINRAG_PERSIST_DIR: /custom/db" in out
    assert "--- Explicitly set (runtime env) ---" in out
    assert "--- Defaults (not set in env) ---" in out
    assert "PINRAG_COLLECTION_NAME: pinrag" in out  # unset, so in Defaults


def test_server_config_resource_unset_var_in_defaults_section() -> None:
    """server_config_resource lists unset vars under Defaults (not set in env)."""
    env_no_pinrag = {
        k: v for k, v in __import__("os").environ.items() if not k.startswith("PINRAG_")
    }
    with patch.dict("os.environ", env_no_pinrag, clear=True):
        out = asyncio.run(mcp_server.server_config_resource())
    assert "--- Defaults (not set in env) ---" in out
    assert "PINRAG_COLLECTION_NAME:" in out


def test_server_config_resource_includes_langsmith_section() -> None:
    """server_config_resource shows LangSmith vars when set in runtime env."""
    extra = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com",
        "LANGSMITH_API_KEY": "lsv2_test_key",
        "LANGSMITH_PROJECT": "pinrag",
    }
    with patch.dict("os.environ", extra, clear=False):
        out = asyncio.run(mcp_server.server_config_resource())
    assert "--- LangSmith observability ---" in out
    assert "LANGSMITH_TRACING: true" in out
    assert "LANGSMITH_ENDPOINT: https://eu.api.smith.langchain.com" in out
    assert "LANGSMITH_PROJECT: pinrag" in out
    assert "LANGSMITH_API_KEY: set" in out
    assert "lsv2_test_key" not in out


def test_server_config_resource_langsmith_not_set() -> None:
    """server_config_resource shows 'not set' for LangSmith vars when absent."""
    env_no_ls = {
        k: v
        for k, v in __import__("os").environ.items()
        if not k.startswith("LANGSMITH_")
    }
    with patch.dict("os.environ", env_no_ls, clear=True):
        out = asyncio.run(mcp_server.server_config_resource())
    assert "LANGSMITH_TRACING: not set" in out
    assert "LANGSMITH_API_KEY: not set" in out


# --- configure_logging ---


def _reset_pinrag_logger() -> None:
    """Reset pinrag logger to clean state for isolated tests."""
    root = logging.getLogger("pinrag")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    root.propagate = True


def test_configure_logging_suppresses_pinrag_logger() -> None:
    """configure_logging always suppresses pinrag logger output."""
    _reset_pinrag_logger()
    mcp_server.configure_logging()
    root = logging.getLogger("pinrag")
    assert root.level == logging.CRITICAL + 1
    assert root.handlers == []
    assert root.propagate is False


def test_configure_logging_suppresses_tqdm_without_tqdm_disable_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset TQDM_DISABLE: tqdm is silenced via patch, not by mutating os.environ."""
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.setattr(mcp_logging, "_PINRAG_TQDM_PATCHED", False)
    _reset_pinrag_logger()
    mcp_server.configure_logging()
    assert "TQDM_DISABLE" not in os.environ
    from io import StringIO

    from tqdm import tqdm

    buf = StringIO()
    list(tqdm(range(2), file=buf, mininterval=0))
    assert buf.getvalue() == ""


# --- remove_document ---


def test_remove_document_deletes_parent_chunks_when_parent_child_enabled(
    tmp_path: Path,
) -> None:
    """remove_document deletes both Chroma children and docstore parents when parent-child is on."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    # First call: get chunks by document_id; second: check if parent is referenced elsewhere
    mock_store.get.side_effect = [
        {
            "ids": ["child1", "child2"],
            "metadatas": [
                {"doc_id": "parent-uuid-1", "document_id": "doc.pdf"},
                {"doc_id": "parent-uuid-1", "document_id": "doc.pdf"},
            ],
        },
        {"metadatas": [{"document_id": "doc.pdf"}]},  # parent only ref'd by doc.pdf
    ]
    mock_docstore = MagicMock()

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch("pinrag.core.operations.get_use_parent_child", return_value=True):
            with patch(
                "pinrag.core.operations.get_parent_docstore", return_value=mock_docstore
            ):
                result = remove_document(
                    document_id="doc.pdf",
                    persist_dir=str(tmp_path),
                    collection="test_coll",
                )

    assert result["deleted_chunks"] == 2
    assert result["document_id"] == "doc.pdf"
    mock_docstore.mdelete.assert_called_once()
    assert set(mock_docstore.mdelete.call_args[0][0]) == {"parent-uuid-1"}
    mock_store.delete.assert_called_once_with(where={"document_id": "doc.pdf"})


def test_remove_document_skips_docstore_when_parent_child_disabled(
    tmp_path: Path,
) -> None:
    """remove_document does not touch docstore when parent-child is off."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["c1"], "metadatas": [{}]}

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch("pinrag.core.operations.get_use_parent_child", return_value=False):
            with patch("pinrag.core.operations.get_parent_docstore") as mock_get_docstore:
                remove_document(
                    document_id="doc.pdf", persist_dir=str(tmp_path), collection="c"
                )
    mock_get_docstore.assert_not_called()


# --- Error handling: propagate ---


def test_server_tool_propagates_on_failure() -> None:
    """When a tool raises, the server decorator re-raises the exception."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        asyncio.run(mcp_server.query_tool(query=""))


# --- _log_tool_errors: context logging, schema stability, redaction ---


def test_build_tool_summary_query_redacts() -> None:
    """query_tool summary never exposes full raw query text."""
    s = mcp_logging._build_tool_summary(
        "query_tool", {"query": "my secret search term"}
    )
    assert "my secret search term" not in s
    assert "query=<" in s and "chars>" in s


def test_resolve_context_from_kwargs_uses_annotation() -> None:
    """Context is resolved by annotation (find_context_parameter), not by hardcoded names."""
    from mcp.server.fastmcp.utilities.context_injection import find_context_parameter

    # query_tool has ctx: Context | None - find_context_parameter finds it by annotation
    param = find_context_parameter(mcp_server.query_tool)
    assert param == "ctx"

    # When kwargs has wrong type (e.g. string), returns None (isinstance guard)
    result = mcp_logging._resolve_context_from_kwargs(
        mcp_server.query_tool, {"query": "x", "ctx": "not a Context"}
    )
    assert result is None

    # When ctx not in kwargs, returns None
    result = mcp_logging._resolve_context_from_kwargs(
        mcp_server.query_tool, {"query": "x"}
    )
    assert result is None


def test_query_tool_schema_has_expected_parameters() -> None:
    """query_tool MCP schema includes expected parameters (no regression from decorator)."""
    import inspect

    sig = inspect.signature(mcp_server.query_tool)
    params = list(sig.parameters)
    assert "query" in params
    assert "document_id" in params
    assert "page_min" in params
    assert "page_max" in params
    assert "tag" in params
    assert "document_type" in params
    assert "response_style" in params
    assert "ctx" in params


def test_query_tool_uses_pinrag_response_style_when_response_style_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted or empty response_style forwards get_response_style() (e.g. PINRAG_RESPONSE_STYLE)."""
    captured: dict[str, str] = {}

    def fake_query(**kwargs: object) -> dict[str, str]:
        captured["response_style"] = str(kwargs.get("response_style", ""))
        return {"answer": "ok", "sources": []}

    monkeypatch.setenv("PINRAG_RESPONSE_STYLE", "concise")
    with patch("pinrag.mcp.server.query_index", side_effect=fake_query):
        asyncio.run(mcp_server.query_tool(query="hello"))

    assert captured.get("response_style") == "concise"


def test_add_document_tool_schema_has_expected_parameters() -> None:
    """add_document_tool MCP schema includes expected parameters."""
    import inspect

    sig = inspect.signature(mcp_server.add_document_tool)
    params = list(sig.parameters)
    assert "paths" in params
    assert "tags" in params
    assert "branch" in params
    assert "include_patterns" in params
    assert "exclude_patterns" in params
    assert "ctx" in params


def test_context_logging_emits_for_async_tool() -> None:
    """When ctx is provided, _emit_client_log is called (async tool path)."""
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock(return_value=None)

    asyncio.run(mcp_logging._emit_client_log(mock_ctx, "test message"))

    mock_ctx.log.assert_called_once_with(
        "info", "test message", logger_name="pinrag"
    )


def test_mcp_list_resources_includes_pinrag_uris() -> None:
    """PinRAG resources must be static entries in resources/list for clients like Cursor."""

    async def _uris() -> list[str]:
        resources = await mcp_server.mcp.list_resources()
        return [str(r.uri) for r in resources]

    uris = asyncio.run(_uris())
    assert "pinrag://documents" in uris
    assert "pinrag://server-config" in uris


def test_format_documents_list_sorts_case_insensitively() -> None:
    """Documents resource lists entries sorted by YouTube title or file id."""
    fake = {
        "documents": ["zzz.pdf", "vid_b", "vid_a"],
        "total_chunks": 3,
        "document_details": {
            "zzz.pdf": {"pages": 10},
            "vid_b": {
                "document_type": "youtube",
                "title": "Zebra Episode",
                "segments": 1,
            },
            "vid_a": {
                "document_type": "youtube",
                "title": "alpha Episode",
                "segments": 1,
            },
        },
    }
    with patch.object(resource_text, "list_documents", return_value=fake):
        out = resource_text.format_documents_list()
    assert out.index("alpha Episode") < out.index("Zebra Episode")
    assert out.index("Zebra Episode") < out.index("zzz.pdf")
