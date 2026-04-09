"""Tests for remove_document title / stem resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pinrag.mcp.tools import remove_document


def test_remove_document_resolves_list_title(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.side_effect = [
        {"ids": []},
        {
            "ids": ["m1"],
            "metadatas": [
                {
                    "document_id": "Amiga Intern 1992.pdf",
                    "document_type": "pdf",
                    "doc_title": "Amiga Intern 1992",
                },
            ],
        },
        {"ids": ["c1"], "metadatas": []},
    ]

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch("pinrag.core.operations.get_use_parent_child", return_value=False):
            result = remove_document(
                document_id="amiga intern 1992",
                persist_dir=str(tmp_path),
                collection="pinrag",
            )

    assert result["document_id"] == "Amiga Intern 1992.pdf"
    assert result["deleted_chunks"] == 1
    mock_store.delete.assert_called_once_with(
        where={"document_id": "Amiga Intern 1992.pdf"}
    )


def test_remove_document_resolves_pdf_stem_without_title(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.side_effect = [
        {"ids": []},
        {
            "ids": ["m1"],
            "metadatas": [
                {
                    "document_id": "Handbook Of Chips.pdf",
                    "document_type": "pdf",
                },
            ],
        },
        {"ids": ["c1"], "metadatas": []},
    ]

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch("pinrag.core.operations.get_use_parent_child", return_value=False):
            result = remove_document(
                document_id="Handbook Of Chips",
                persist_dir=str(tmp_path),
                collection="pinrag",
            )

    assert result["document_id"] == "Handbook Of Chips.pdf"
    mock_store.delete.assert_called_once_with(
        where={"document_id": "Handbook Of Chips.pdf"}
    )


def test_remove_document_ambiguous_title_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.side_effect = [
        {"ids": []},
        {
            "ids": ["a", "b"],
            "metadatas": [
                {"document_id": "a.pdf", "doc_title": "Same Title"},
                {"document_id": "b.pdf", "doc_title": "Same Title"},
            ],
        },
    ]

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch("pinrag.core.operations.get_use_parent_child", return_value=False):
            with pytest.raises(ValueError, match="Ambiguous document"):
                remove_document(
                    document_id="Same Title",
                    persist_dir=str(tmp_path),
                    collection="pinrag",
                )
