"""Tests for MCP query tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tests.helpers.mcp_patched_server  # noqa: F401 — patch server import-time validation
from pinrag.mcp.tools import query


def test_query_empty_query_raises() -> None:
    """Query raises ValueError when query is empty."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        query(user_query="")
    with pytest.raises(ValueError, match="Query cannot be empty"):
        query(user_query="   ")


def test_query_uses_config_for_persist_and_collection(tmp_path: Path) -> None:
    """Query uses get_persist_dir and get_collection_name from config."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="pinrag"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag") as mock_run:
                        mock_run.return_value = RAGResult(answer="ok", sources=[])
                        query(user_query="test")
    mock_run.assert_called_once()


def test_query_missing_persist_dir_raises() -> None:
    """Query raises FileNotFoundError when persist dir does not exist."""
    with patch(
        "pinrag.core.operations.get_persist_dir", return_value="/nonexistent/chroma_db"
    ):
        with pytest.raises(
            FileNotFoundError, match="Persistence directory does not exist"
        ):
            query(user_query="test")


def test_query_chain_error_propagates(tmp_path: Path) -> None:
    """Query propagates retrieval/LLM errors from run_rag."""
    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_embedding_model"):
            with patch("pinrag.core.operations.get_chat_model"):
                with patch(
                    "pinrag.core.operations.run_rag",
                    side_effect=RuntimeError("OpenAI API rate limit"),
                ):
                    with pytest.raises(RuntimeError, match="OpenAI API rate limit"):
                        query(user_query="test")


def test_query_success(tmp_path: Path) -> None:
    """Query returns answer and sources when run_rag succeeds."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    mock_run_rag = MagicMock(
        return_value=RAGResult(
            answer="The answer is 42.",
            sources=[
                {"document_id": "doc.pdf", "page": 1},
                {"document_id": "doc.pdf", "page": 2},
            ],
        )
    )

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="test_coll"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag", mock_run_rag):
                        result = query(user_query="What is the answer?")

    assert result["answer"] == "The answer is 42."
    assert len(result["sources"]) == 2
    assert result["sources"][0] == {"document_id": "doc.pdf", "page": 1}
    assert result["sources"][1] == {"document_id": "doc.pdf", "page": 2}
    mock_run_rag.assert_called_once()


def test_query_sources_include_start_for_youtube(tmp_path: Path) -> None:
    """Query returns start (timestamp) in sources when present (YouTube)."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    mock_run_rag = MagicMock(
        return_value=RAGResult(
            answer="From the video.",
            sources=[
                {"document_id": "dQw4w9WgXcQ", "page": 0, "start": 83},
                {"document_id": "dQw4w9WgXcQ", "page": 0, "start": 120},
            ],
        )
    )

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="test_coll"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag", mock_run_rag):
                        result = query(user_query="What does the video say?")

    assert result["sources"][0] == {
        "document_id": "dQw4w9WgXcQ",
        "page": 0,
        "start": 83,
    }
    assert result["sources"][1] == {
        "document_id": "dQw4w9WgXcQ",
        "page": 0,
        "start": 120,
    }
    assert mock_run_rag.call_args[0][0] == "What does the video say?"


def test_query_page_range_validation() -> None:
    """Query raises when page_min or page_max is provided without the other."""
    with pytest.raises(
        ValueError, match="page_min and page_max must be provided together"
    ):
        query(user_query="test", page_min=1)
    with pytest.raises(
        ValueError, match="page_min and page_max must be provided together"
    ):
        query(user_query="test", page_max=10)
    with pytest.raises(ValueError, match="page_min must be <= page_max"):
        query(user_query="test", page_min=10, page_max=1)


def test_query_with_page_range(tmp_path: Path) -> None:
    """Query passes page_min and page_max to run_rag when provided."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    mock_run_rag = MagicMock(
        return_value=RAGResult(
            answer="Page 16 content.",
            sources=[{"document_id": "pico.pdf", "page": 16}],
        )
    )

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="test_coll"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag", mock_run_rag):
                        query(
                            user_query="OpenOCD?",
                            document_id="pico.pdf",
                            page_min=16,
                            page_max=16,
                        )

    mock_run_rag.assert_called_once()
    assert mock_run_rag.call_args[1]["page_min"] == 16
    assert mock_run_rag.call_args[1]["page_max"] == 16


def test_query_with_document_type_filter(tmp_path: Path) -> None:
    """Query passes document_type to run_rag when provided."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    mock_run_rag = MagicMock(
        return_value=RAGResult(
            answer="From YouTube.",
            sources=[{"document_id": "bwgLXEQdq20", "page": 0, "start": 664}],
        )
    )

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="test_coll"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag", mock_run_rag):
                        query(user_query="OTG?", document_type="youtube")

    mock_run_rag.assert_called_once()
    assert mock_run_rag.call_args[1]["document_type"] == "youtube"


def test_query_with_tag_filter(tmp_path: Path) -> None:
    """Query passes tag to run_rag when provided."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    mock_run_rag = MagicMock(
        return_value=RAGResult(
            answer="From PI_PICO docs.",
            sources=[{"document_id": "pico.pdf", "page": 1}],
        )
    )

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="test_coll"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag", mock_run_rag):
                        query(user_query="GPIO?", tag="PI_PICO")

    mock_run_rag.assert_called_once()
    assert mock_run_rag.call_args[1]["tag"] == "PI_PICO"


def test_query_with_document_id_filter(tmp_path: Path) -> None:
    """Query passes document_id to run_rag when provided."""
    from pinrag.rag import RAGResult

    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    mock_run_rag = MagicMock(
        return_value=RAGResult(
            answer="Filtered answer.",
            sources=[{"document_id": "pico.pdf", "page": 1}],
        )
    )

    with patch("pinrag.core.operations.get_persist_dir", return_value=str(tmp_path)):
        with patch("pinrag.core.operations.get_collection_name", return_value="test_coll"):
            with patch("pinrag.core.operations.get_embedding_model"):
                with patch("pinrag.core.operations.get_chat_model"):
                    with patch("pinrag.core.operations.run_rag", mock_run_rag):
                        query(
                            user_query="GPIO?",
                            document_id="RP-008276-DS-1-getting-started-with-pico.pdf",
                        )

    mock_run_rag.assert_called_once()
    assert (
        mock_run_rag.call_args[1]["document_id"]
        == "RP-008276-DS-1-getting-started-with-pico.pdf"
    )

