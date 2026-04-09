"""Tests for list_collections core helper."""

from __future__ import annotations

from pathlib import Path

import chromadb

from pinrag.core.operations import list_collections


def test_list_collections_empty_dir(tmp_path: Path) -> None:
    """Non-existent persist path returns empty list."""
    missing = tmp_path / "nope"
    out = list_collections(persist_dir=str(missing))
    assert out["collections"] == []
    assert "persist_directory" in out


def test_list_collections_finds_chroma_collections(tmp_path: Path) -> None:
    """Persistent client creates default collection; we list its name."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.get_or_create_collection("alpha")
    client.get_or_create_collection("beta")
    out = list_collections(persist_dir=str(tmp_path))
    assert out["collections"] == ["alpha", "beta"]
