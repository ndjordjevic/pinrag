"""Tests for MCP add_file, add_files, and related paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tests.helpers.mcp_patched_server  # noqa: F401 — patch server import-time validation
from pinrag.mcp.tools import add_file, add_files


def test_add_file_empty_path_raises() -> None:
    """add_file raises ValueError when path is empty."""
    with pytest.raises(ValueError, match="path cannot be empty"):
        add_file(path="")
    with pytest.raises(ValueError, match="path cannot be empty"):
        add_file(path="   ")


def test_add_file_empty_collection_uses_default(tmp_path: Path) -> None:
    """add_file uses default collection when collection is empty."""
    pdf_file = tmp_path / "dummy.pdf"
    pdf_file.touch()
    with patch("pinrag.core.operations.index_pdf") as mock_index:
        mock_index.return_value = MagicMock(
            source_path=pdf_file,
            total_pages=1,
            total_chunks=1,
            persist_directory=tmp_path,
            collection_name="test",
        )
        with patch("pinrag.core.operations.get_embedding_model"):
            add_file(path=str(pdf_file), persist_dir=str(tmp_path), collection="")
    mock_index.assert_called_once()


def test_add_file_path_not_found_raises() -> None:
    """add_file raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError, match="Path not found"):
        add_file(path="/nonexistent/file.pdf")


def test_add_file_unsupported_format_raises(tmp_path: Path) -> None:
    """add_file raises ValueError when file format is unsupported."""
    doc_file = tmp_path / "document.docx"
    doc_file.write_bytes(b"PK\x03\x04")  # minimal zip header
    with pytest.raises(ValueError, match="Unsupported format"):
        add_file(path=str(doc_file))


def test_add_file_pdf_success(tmp_path: Path) -> None:
    """add_file returns indexing result when PDF exists and index_pdf succeeds."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.touch()
    fake_result = MagicMock()
    fake_result.source_path = pdf_file
    fake_result.total_pages = 5
    fake_result.total_chunks = 12

    with patch("pinrag.core.operations.index_pdf", return_value=fake_result) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path=str(pdf_file),
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 1
    assert result["indexed"][0]["format"] == "pdf"
    assert result["indexed"][0]["source_path"] == str(pdf_file)
    assert result["indexed"][0]["total_pages"] == 5
    assert result["indexed"][0]["total_chunks"] == 12
    mock_index.assert_called_once()


def test_add_file_plaintext_detection_and_success(tmp_path: Path) -> None:
    """add_file detects plain .txt (no Discord header) and routes to index_plaintext."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Hello world. Plain text content.", encoding="utf-8")
    fake_result = MagicMock()
    fake_result.source_path = txt_file
    fake_result.total_chunks = 2
    fake_result.document_id = "notes.txt"

    with patch(
        "pinrag.core.operations.index_plaintext", return_value=fake_result
    ) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path=str(txt_file),
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 1
    assert result["indexed"][0]["format"] == "plaintext"
    assert result["indexed"][0]["source_path"] == str(txt_file)
    assert result["indexed"][0]["document_id"] == "notes.txt"
    assert result["indexed"][0]["total_chunks"] == 2
    mock_index.assert_called_once()


def test_add_file_discord_txt_still_detected_as_discord(tmp_path: Path) -> None:
    """add_file treats .txt with Guild:/Channel: as Discord, not plaintext."""
    txt_file = tmp_path / "discord_export.txt"
    txt_file.write_text(
        "Guild: Test Server\nChannel: general\n\n[2024-01-01 12:00] User: Hello",
        encoding="utf-8",
    )
    fake_result = MagicMock()
    fake_result.source_path = txt_file
    fake_result.total_chunks = 5

    with patch(
        "pinrag.core.operations.index_discord", return_value=fake_result
    ) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path=str(txt_file),
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 1
    assert result["indexed"][0]["format"] == "discord"
    mock_index.assert_called_once()


def test_add_file_with_tag_passes_tag(tmp_path: Path) -> None:
    """add_file passes tag to index_pdf when provided."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.touch()
    fake_result = MagicMock()
    fake_result.source_path = pdf_file
    fake_result.total_pages = 5
    fake_result.total_chunks = 12

    with patch("pinrag.core.operations.index_pdf", return_value=fake_result) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            add_file(
                path=str(pdf_file),
                persist_dir=str(tmp_path),
                collection="test_coll",
                tag="amiga",
            )

    mock_index.assert_called_once()
    assert mock_index.call_args[1]["tag"] == "amiga"


# --- YouTube ---


def test_add_file_youtube_detection_and_success(tmp_path: Path) -> None:
    """add_file detects YouTube URL and routes to index_youtube."""
    fake_result = MagicMock()
    fake_result.video_id = "dQw4w9WgXcQ"
    fake_result.source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    fake_result.total_segments = 42
    fake_result.total_chunks = 15

    with patch(
        "pinrag.core.operations.index_youtube", return_value=fake_result
    ) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path="https://youtu.be/dQw4w9WgXcQ",
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 1
    assert result["indexed"][0]["format"] == "youtube"
    assert result["indexed"][0]["video_id"] == "dQw4w9WgXcQ"
    assert result["indexed"][0]["total_segments"] == 42
    assert result["indexed"][0]["total_chunks"] == 15
    mock_index.assert_called_once_with(
        "https://youtu.be/dQw4w9WgXcQ",
        persist_directory=str(tmp_path),
        collection_name="test_coll",
        embedding=mock_index.call_args[1]["embedding"],
        tag=None,
        verbose_emitter=None,
    )


def test_add_file_local_path_prioritized_over_youtube_id(tmp_path: Path) -> None:
    """add_file treats local file/dir named like YouTube ID as local, not YouTube."""
    pdf_file = tmp_path / "dQw4w9WgXcQ.pdf"
    pdf_file.touch()
    fake_result = MagicMock()
    fake_result.source_path = pdf_file
    fake_result.total_pages = 1
    fake_result.total_chunks = 1

    with patch("pinrag.core.operations.index_pdf", return_value=fake_result) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path=str(pdf_file),
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 1
    assert result["indexed"][0]["format"] == "pdf"
    mock_index.assert_called_once()


def test_add_file_youtube_transcript_error_returns_failed(tmp_path: Path) -> None:
    """add_file catches transcript errors and returns them in failed."""
    with patch("pinrag.core.operations.index_youtube") as mock_index:
        mock_index.side_effect = RuntimeError(
            "No transcript found for YouTube video xyz"
        )
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path="https://youtu.be/xyz12345678",
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 0
    assert result["total_failed"] == 1
    assert len(result["failed"]) == 1
    assert "No transcript found" in result["failed"][0]["error"]


def test_add_file_youtube_playlist_detection_and_success(tmp_path: Path) -> None:
    """add_file detects playlist URL and routes to index_youtube_playlist."""
    fake_result = MagicMock()
    fake_result.playlist_id = "PLtest"
    fake_result.playlist_title = "Test Playlist"
    fake_result.total_indexed = 2
    fake_result.total_failed = 0
    fake_result.indexed = [
        MagicMock(
            video_id="abc12345678",
            source_url="https://www.youtube.com/watch?v=abc12345678",
            total_segments=2,
            total_chunks=1,
            title="V1",
        ),
        MagicMock(
            video_id="xyz98765432",
            source_url="https://www.youtube.com/watch?v=xyz98765432",
            total_segments=3,
            total_chunks=2,
            title="V2",
        ),
    ]
    fake_result.failed = []

    with patch(
        "pinrag.core.operations.index_youtube_playlist", return_value=fake_result
    ) as mock_index:
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_file(
                path="https://www.youtube.com/playlist?list=PLtest",
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 2
    assert len(result["indexed"]) == 2
    assert result["indexed"][0]["format"] == "youtube_playlist"
    assert result["indexed"][0]["video_id"] == "abc12345678"
    assert result["indexed"][1]["video_id"] == "xyz98765432"
    mock_index.assert_called_once_with(
        "https://www.youtube.com/playlist?list=PLtest",
        persist_directory=str(tmp_path),
        collection_name="test_coll",
        embedding=mock_index.call_args[1]["embedding"],
        tag=None,
        verbose_emitter=None,
    )


def test_add_files_tags_length_mismatch_raises() -> None:
    """add_files raises when tags length does not match paths."""
    with pytest.raises(ValueError, match="tags must have same length"):
        add_files(paths=["a.pdf", "b.pdf"], tags=["amiga"])


def test_add_files_empty_paths_raises() -> None:
    """add_files raises ValueError when path list is empty."""
    with pytest.raises(ValueError, match="paths cannot be empty"):
        add_files(paths=[])


def test_add_files_partial_success(tmp_path: Path) -> None:
    """add_files indexes valid files and reports failures per file."""
    good_pdf = tmp_path / "good.pdf"
    bad_ext = tmp_path / "bad.docx"  # unsupported format
    good_pdf.touch()
    bad_ext.write_bytes(b"PK\x03\x04")  # minimal zip header

    fake_result = MagicMock()
    fake_result.source_path = good_pdf
    fake_result.total_pages = 3
    fake_result.total_chunks = 7

    with patch("pinrag.core.operations.index_pdf", return_value=fake_result):
        with patch("pinrag.core.operations.get_embedding_model"):
            result = add_files(
                paths=[str(good_pdf), str(bad_ext), str(tmp_path / "missing.pdf")],
                persist_dir=str(tmp_path),
                collection="test_coll",
            )

    assert result["total_indexed"] == 1
    assert result["total_failed"] == 2
    assert result["indexed"][0]["source_path"] == str(good_pdf)
    assert len(result["failed"]) == 2
    assert result["fail_summary"] == {
        "blocked": 0,
        "disabled": 0,
        "missing_transcript": 0,
        "other": 2,
    }


def test_add_files_fail_summary_by_reason() -> None:
    """add_files returns fail_summary categorizing failures: blocked, disabled, missing_transcript, other."""
    from pinrag.core import categorize_failures

    # Unit test for categorization
    failed = [
        {
            "path": "v1",
            "error": "YouTube is blocking requests from your IP. Cloud provider.",
        },
        {"path": "v2", "error": "Video xyz has transcripts disabled. Cannot index."},
        {"path": "v3", "error": "Could not retrieve a transcript for the video."},
        {"path": "v4", "error": "No transcript found"},
        {"path": "v5", "error": "File not found"},
    ]
    summary = categorize_failures(failed)
    assert summary["blocked"] == 1
    assert summary["disabled"] == 1
    assert summary["missing_transcript"] == 2
    assert summary["other"] == 1

    # add_files with failures includes fail_summary
    result = add_files(
        paths=["/nonexistent/a.pdf", "/nonexistent/b.pdf"],
        collection="test_coll",
    )
    assert result["total_failed"] == 2
    assert "fail_summary" in result
    assert result["fail_summary"]["other"] == 2
