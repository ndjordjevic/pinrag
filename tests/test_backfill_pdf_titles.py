"""Tests for Chroma backfill of PDF doc_title metadata."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pinrag.core.operations import backfill_pdf_doc_titles


def test_backfill_pdf_doc_titles_updates_chunks_with_shared_title(tmp_path: Path) -> None:
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["c1", "c2"],
        "metadatas": [
            {
                "document_type": "pdf",
                "document_id": "x.pdf",
                "document_title": "  Embedded Title  ",
            },
            {"document_type": "pdf", "document_id": "x.pdf", "page": 2},
        ],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        out = backfill_pdf_doc_titles(
            persist_dir=str(tmp_path),
            collection="pinrag",
            batch_size=256,
        )

    assert out["updated_chunks"] == 2
    mock_col.update.assert_called_once()
    kwargs = mock_col.update.call_args.kwargs
    assert kwargs["ids"] == ["c1", "c2"]
    merged = kwargs["metadatas"]
    assert merged[0]["doc_title"] == "Embedded Title"
    assert merged[1]["doc_title"] == "Embedded Title"


def test_backfill_pdf_doc_titles_uses_stem_without_embedded_title(
    tmp_path: Path,
) -> None:
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["only"],
        "metadatas": [
            {"document_type": "pdf", "document_id": "Bare-metal Amiga programming 2021_ocr.pdf"},
        ],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        backfill_pdf_doc_titles(persist_dir=str(tmp_path), collection="pinrag")

    kwargs = mock_col.update.call_args.kwargs
    assert kwargs["metadatas"][0]["doc_title"] == "Bare-metal Amiga programming 2021_ocr"


def test_backfill_pdf_doc_titles_skips_complete(tmp_path: Path) -> None:
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["ok"],
        "metadatas": [
            {"document_type": "pdf", "document_id": "a.pdf", "doc_title": "Already"},
        ],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        out = backfill_pdf_doc_titles(persist_dir=str(tmp_path), collection="pinrag")

    assert out["updated_chunks"] == 0
    mock_col.update.assert_not_called()
