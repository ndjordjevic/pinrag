"""Tests for indexing PDFs into Chroma."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from pinrag.indexing import IndexResult, index_pdf, query_index
from pinrag.vectorstore import get_chroma_store
from tests.helpers.embeddings import MockEmbeddings1536


def test_index_pdf_smoke(tmp_path: Path, sample_pdf_path: Path) -> None:
    """Index a sample PDF into a temp Chroma dir and verify chunks are stored."""
    persist_dir = tmp_path / "chroma_idx"
    result = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_index",
        embedding=MockEmbeddings1536(),
    )

    assert isinstance(result, IndexResult)
    assert result.total_pages > 0
    assert result.total_chunks > 0
    assert result.persist_directory == persist_dir.resolve()

    # Re-open the store and ensure we can retrieve something back.
    store = get_chroma_store(
        persist_directory=str(persist_dir),
        collection_name="test_index",
        embedding=MockEmbeddings1536(),
    )
    docs = store.similarity_search("test", k=2)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    meta = docs[0].metadata
    assert "upload_timestamp" in meta
    assert "doc_pages" in meta
    assert "doc_bytes" in meta
    assert "doc_total_chunks" in meta
    assert meta.get("doc_title")


def test_index_pdf_with_tag(tmp_path: Path, sample_pdf_path: Path) -> None:
    """Index with tag; chunks have tag in metadata."""
    persist_dir = tmp_path / "chroma_idx"
    index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_tag",
        embedding=MockEmbeddings1536(),
        tag="amiga",
    )

    store = get_chroma_store(
        persist_directory=str(persist_dir),
        collection_name="test_tag",
        embedding=MockEmbeddings1536(),
    )
    docs = store.similarity_search("test", k=2)
    assert len(docs) > 0
    assert docs[0].metadata.get("tag") == "amiga"


def test_query_index_filter_by_page_range(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    """query_index with page_min/page_max returns only chunks in that page range."""
    persist_dir = tmp_path / "chroma_idx"
    index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_page_filter",
        embedding=MockEmbeddings1536(),
    )

    # Filter to single page (e.g. page 1 only)
    docs = query_index(
        "test",
        k=10,
        persist_directory=str(persist_dir),
        collection_name="test_page_filter",
        embedding=MockEmbeddings1536(),
        page_min=1,
        page_max=1,
    )
    assert len(docs) > 0
    for d in docs:
        assert d.metadata.get("page") == 1

    # Filter to non-existent page range — no results
    empty = query_index(
        "test",
        k=5,
        persist_directory=str(persist_dir),
        collection_name="test_page_filter",
        embedding=MockEmbeddings1536(),
        page_min=999,
        page_max=999,
    )
    assert len(empty) == 0


def test_query_index_filter_by_tag(tmp_path: Path, sample_pdf_path: Path) -> None:
    """query_index with tag filter returns only chunks from documents with that tag."""
    persist_dir = tmp_path / "chroma_idx"
    index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_tag_filter",
        embedding=MockEmbeddings1536(),
        tag="PI_PICO",
    )

    docs = query_index(
        "test",
        k=10,
        persist_directory=str(persist_dir),
        collection_name="test_tag_filter",
        embedding=MockEmbeddings1536(),
        tag="PI_PICO",
    )
    assert len(docs) > 0
    for d in docs:
        assert d.metadata.get("tag") == "PI_PICO"

    # Filter by non-existent tag — no results
    empty = query_index(
        "test",
        k=5,
        persist_directory=str(persist_dir),
        collection_name="test_tag_filter",
        embedding=MockEmbeddings1536(),
        tag="nonexistent",
    )
    assert len(empty) == 0


def test_query_index_filter_by_document(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    """query_index with document_id filter returns only chunks from that document."""
    persist_dir = tmp_path / "chroma_idx"
    index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_filter",
        embedding=MockEmbeddings1536(),
    )

    # Filter by actual document_id — all results should be from that doc
    docs = query_index(
        "test",
        k=10,
        persist_directory=str(persist_dir),
        collection_name="test_filter",
        embedding=MockEmbeddings1536(),
        document_id="sample-text.pdf",
    )
    assert len(docs) > 0
    for d in docs:
        assert d.metadata.get("document_id") == "sample-text.pdf"

    # Filter by non-existent document_id — no results
    empty = query_index(
        "test",
        k=5,
        persist_directory=str(persist_dir),
        collection_name="test_filter",
        embedding=MockEmbeddings1536(),
        document_id="nonexistent.pdf",
    )
    assert len(empty) == 0


def test_query_index_uses_existing_store(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    """query_index returns results from an already-indexed Chroma store."""
    persist_dir = tmp_path / "chroma_idx"
    result = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_query_index",
        embedding=MockEmbeddings1536(),
    )

    docs = query_index(
        "test",
        k=2,
        persist_directory=result.persist_directory,
        collection_name=result.collection_name,
        embedding=MockEmbeddings1536(),
    )
    assert len(docs) > 0
    assert isinstance(docs[0], Document)


def test_index_pdf_replaces_duplicate(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    """Indexing the same PDF twice replaces old chunks; no duplicates."""
    persist_dir = tmp_path / "chroma_idx"
    coll = "test_replace"
    emb = MockEmbeddings1536()

    result1 = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name=coll,
        embedding=emb,
    )
    store = get_chroma_store(
        persist_directory=str(persist_dir),
        collection_name=coll,
        embedding=emb,
    )
    data1 = store._collection.get(include=[])
    count1 = len(data1.get("ids") or [])

    # Index same PDF again — should replace, not add
    result2 = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name=coll,
        embedding=emb,
    )
    data2 = store._collection.get(include=[])
    count2 = len(data2.get("ids") or [])

    assert count2 == count1, "Re-indexing should replace chunks, not duplicate"
    assert result2.total_chunks == result1.total_chunks


def test_index_pdf_uses_config_chunk_size_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_pdf_path: Path,
) -> None:
    """index_pdf uses PINRAG_CHUNK_SIZE / PINRAG_CHUNK_OVERLAP when not passed."""
    monkeypatch.setenv(
        "PINRAG_USE_PARENT_CHILD", "false"
    )  # use flat mode for chunk_size test
    persist_dir = tmp_path / "chroma_idx"
    emb = MockEmbeddings1536()

    # Default config (no env set) → default chunk size
    monkeypatch.delenv("PINRAG_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("PINRAG_CHUNK_OVERLAP", raising=False)
    result_default = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_config_default",
        embedding=emb,
    )

    # Set env to smaller chunk size → index_pdf (without kwargs) should use it → more chunks
    monkeypatch.setenv("PINRAG_CHUNK_SIZE", "200")
    monkeypatch.setenv("PINRAG_CHUNK_OVERLAP", "0")
    result_from_env = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_config_env",
        embedding=emb,
    )
    assert result_from_env.total_chunks > result_default.total_chunks


def test_index_pdf_respects_chunk_size_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_pdf_path: Path,
) -> None:
    """index_pdf uses explicit chunk_size/chunk_overlap when passed."""
    monkeypatch.setenv(
        "PINRAG_USE_PARENT_CHILD", "false"
    )  # use flat mode for chunk_size test
    persist_dir = tmp_path / "chroma_idx"
    result_default = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_default",
        embedding=MockEmbeddings1536(),
    )
    result_small = index_pdf(
        sample_pdf_path,
        persist_directory=str(persist_dir),
        collection_name="test_small",
        embedding=MockEmbeddings1536(),
        chunk_size=200,
        chunk_overlap=0,
    )
    # Smaller chunks → more chunks for the same content
    assert result_small.total_chunks > result_default.total_chunks
