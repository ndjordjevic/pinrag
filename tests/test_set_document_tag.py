"""Tests for set_document_tag core operation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from pinrag.core.operations import set_document_tag


def test_set_document_tag_empty_tag_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with pytest.raises(ValueError, match="tag cannot be empty"):
            set_document_tag(
                document_id="x.pdf",
                tag="   ",
                persist_dir=str(tmp_path),
                collection="pinrag",
            )


def test_set_document_tag_returns_zero_when_no_chunks(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": [], "metadatas": []}
    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch(
            "pinrag.core.operations._resolve_remove_document_id",
            return_value="missing.pdf",
        ):
            out = set_document_tag(
                document_id="missing.pdf",
                tag="AMIGA",
                persist_dir=str(tmp_path),
                collection="pinrag",
            )
    assert out["updated_chunks"] == 0
    assert out["tag"] == "AMIGA"
    assert out["document_id"] == "missing.pdf"
    mock_store._collection.update.assert_not_called()


def test_set_document_tag_batches_chroma_update(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    n = 5
    mock_store.get.return_value = {
        "ids": [f"c{i}" for i in range(n)],
        "metadatas": [
            {"document_id": "book.pdf", "page": i + 1, "tag": "old"} for i in range(n)
        ],
    }
    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch(
            "pinrag.core.operations._resolve_remove_document_id",
            return_value="book.pdf",
        ):
            with patch("pinrag.core.operations.get_use_parent_child", return_value=False):
                out = set_document_tag(
                    document_id="book.pdf",
                    tag="RP6502",
                    persist_dir=str(tmp_path),
                    collection="pinrag",
                    batch_size=2,
                )
    assert out["updated_chunks"] == n
    assert out["tag"] == "RP6502"
    assert mock_col.update.call_count == 3
    first_kw = mock_col.update.call_args_list[0].kwargs
    assert first_kw["metadatas"][0]["tag"] == "RP6502"


def test_set_document_tag_updates_parent_docstore(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_store.get.return_value = {
        "ids": ["child1"],
        "metadatas": [{"document_id": "x.pdf", "doc_id": "parent-uuid"}],
    }
    mock_docstore = MagicMock()
    parent_doc = Document(
        page_content="parent text",
        metadata={"tag": "old", "document_id": "x.pdf"},
    )
    mock_docstore.mget.return_value = [parent_doc]

    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch(
            "pinrag.core.operations._resolve_remove_document_id",
            return_value="x.pdf",
        ):
            with patch("pinrag.core.operations.get_use_parent_child", return_value=True):
                with patch(
                    "pinrag.core.operations.get_parent_docstore",
                    return_value=mock_docstore,
                ):
                    out = set_document_tag(
                        document_id="x.pdf",
                        tag="AMIGA",
                        persist_dir=str(tmp_path),
                        collection="pinrag",
                    )

    assert out["parents_updated"] == 1
    mock_docstore.mset.assert_called_once()
    pairs = mock_docstore.mset.call_args[0][0]
    assert len(pairs) == 1
    assert pairs[0][0] == "parent-uuid"
    assert pairs[0][1].metadata["tag"] == "AMIGA"


def test_set_document_tag_resolves_title_like_remove(tmp_path: Path) -> None:
    """Integration-style: real resolver + mocked Chroma collection.update."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_store.get.side_effect = [
        {"ids": []},
        {
            "ids": ["m1"],
            "metadatas": [
                {
                    "document_id": "Amiga Intern 1992.pdf",
                    "doc_title": "Amiga Intern 1992",
                },
            ],
        },
        {
            "ids": ["c1"],
            "metadatas": [
                {
                    "document_id": "Amiga Intern 1992.pdf",
                    "doc_title": "Amiga Intern 1992",
                },
            ],
        },
    ]
    with patch("pinrag.core.operations.get_chroma_store", return_value=mock_store):
        with patch("pinrag.core.operations.get_use_parent_child", return_value=False):
            out = set_document_tag(
                document_id="amiga intern 1992",
                tag="AMIGA",
                persist_dir=str(tmp_path),
                collection="pinrag",
            )
    assert out["document_id"] == "Amiga Intern 1992.pdf"
    assert out["updated_chunks"] == 1
    mock_col.update.assert_called_once()
