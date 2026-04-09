"""Tests for MCP list_documents."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import tests.helpers.mcp_patched_server  # noqa: F401 — patch server import-time validation
from pinrag.mcp.tools import list_documents


def test_list_documents_missing_persist_dir_returns_empty() -> None:
    """list_documents returns empty when persist dir does not exist."""
    result = list_documents(persist_dir="/nonexistent/path/xyz", collection="test_coll")
    assert result["documents"] == []
    assert result["total_chunks"] == 0
    assert result["collection_name"] == "test_coll"
    assert "persist_directory" in result


def test_list_documents_returns_documents_from_store(tmp_path: Path) -> None:
    """list_documents returns unique document IDs and total_chunks from Chroma."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "metadatas": [
            {"file_name": "a.pdf", "page": 1},
            {"file_name": "a.pdf", "page": 2},
            {"document_id": "b.pdf", "page": 1},
        ]
    }

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        result = list_documents(persist_dir=str(tmp_path), collection="test_coll")

    assert result["documents"] == ["a.pdf", "b.pdf"]
    assert result["total_chunks"] == 3
    assert result["collection_name"] == "test_coll"
    assert "document_details" in result
    mock_store.get.assert_called_once_with(include=["metadatas"])


def test_list_documents_handles_empty_metadatas(tmp_path: Path) -> None:
    """list_documents returns empty list when store has no chunks."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {"metadatas": []}

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        result = list_documents(persist_dir=str(tmp_path), collection="test_coll")

    assert result["documents"] == []
    assert result["total_chunks"] == 0
    assert result["document_details"] == {}


def test_list_documents_tag_filter(tmp_path: Path) -> None:
    """list_documents with tag returns only docs with that tag."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "metadatas": [
            {"document_id": "a.pdf", "tag": "amiga"},
            {"document_id": "a.pdf", "tag": "amiga"},
            {"document_id": "b.pdf", "tag": "pico"},
        ]
    }

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        result = list_documents(persist_dir=str(tmp_path), collection="c", tag="amiga")

    assert result["documents"] == ["a.pdf"]
    assert result["total_chunks"] == 2
    assert result["document_details"]["a.pdf"]["tag"] == "amiga"


def test_list_documents_includes_document_details_when_present(tmp_path: Path) -> None:
    """list_documents returns document_details when chunks have them."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "metadatas": [
            {
                "document_id": "doc.pdf",
                "upload_timestamp": "2025-01-15T12:00:00Z",
                "doc_pages": 56,
                "doc_bytes": 12345,
                "doc_total_chunks": 224,
                "tag": "amiga",
            },
            {
                "document_id": "doc.pdf",
                "upload_timestamp": "2025-01-15T12:00:00Z",
                "doc_pages": 56,
                "doc_bytes": 12345,
                "doc_total_chunks": 224,
                "tag": "amiga",
            },
        ]
    }

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        result = list_documents(persist_dir=str(tmp_path), collection="test_coll")

    assert result["documents"] == ["doc.pdf"]
    assert result["document_details"]["doc.pdf"] == {
        "upload_timestamp": "2025-01-15T12:00:00Z",
        "pages": 56,
        "bytes": 12345,
        "chunks": 224,
        "tag": "amiga",
    }


def test_list_documents_aggregates_bytes_across_files(tmp_path: Path) -> None:
    """list_documents sums doc_bytes across unique files (GitHub: many files per doc_id)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "metadatas": [
            {
                "document_id": "owner/repo",
                "document_type": "github",
                "source": "https://github.com/owner/repo/blob/main/a.py",
                "doc_bytes": 100,
            },
            {
                "document_id": "owner/repo",
                "document_type": "github",
                "source": "https://github.com/owner/repo/blob/main/a.py",
                "doc_bytes": 100,
            },
            {
                "document_id": "owner/repo",
                "document_type": "github",
                "source": "https://github.com/owner/repo/blob/main/b.py",
                "doc_bytes": 96,
            },
        ]
    }

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        result = list_documents(persist_dir=str(tmp_path), collection="test_coll")

    assert result["documents"] == ["owner/repo"]
    assert result["document_details"]["owner/repo"]["bytes"] == 196


def test_list_documents_includes_segments_for_youtube(tmp_path: Path) -> None:
    """list_documents returns segments in document_details for YouTube chunks."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "metadatas": [
            {
                "document_id": "dQw4w9WgXcQ",
                "document_type": "youtube",
                "upload_timestamp": "2025-01-20T10:00:00Z",
                "doc_segments": 42,
                "doc_total_chunks": 15,
                "doc_title": "Never Gonna Give You Up",
                "tag": "tutorial",
            },
            {
                "document_id": "dQw4w9WgXcQ",
                "document_type": "youtube",
                "upload_timestamp": "2025-01-20T10:00:00Z",
                "doc_segments": 42,
                "doc_total_chunks": 15,
                "doc_title": "Never Gonna Give You Up",
                "tag": "tutorial",
            },
        ]
    }

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        result = list_documents(persist_dir=str(tmp_path), collection="test_coll")

    assert result["documents"] == ["dQw4w9WgXcQ"]
    assert result["document_details"]["dQw4w9WgXcQ"]["segments"] == 42
    assert result["document_details"]["dQw4w9WgXcQ"]["chunks"] == 15
    assert (
        result["document_details"]["dQw4w9WgXcQ"]["title"] == "Never Gonna Give You Up"
    )
    assert result["document_details"]["dQw4w9WgXcQ"]["document_type"] == "youtube"

