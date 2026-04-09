"""Tests for PDF display title helper."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from pinrag.indexing.pdf_indexer import pdf_doc_title


def test_pdf_doc_title_prefers_embedded_document_title() -> None:
    docs = [Document("x", metadata={"document_title": "  My Book  "})]
    assert pdf_doc_title(pdf_path=Path("file.pdf"), page_documents=docs) == "My Book"


def test_pdf_doc_title_falls_back_to_filename_stem() -> None:
    docs = [Document("x", metadata={})]
    assert (
        pdf_doc_title(pdf_path=Path("/tmp/foo bar.pdf"), page_documents=docs) == "foo bar"
    )
