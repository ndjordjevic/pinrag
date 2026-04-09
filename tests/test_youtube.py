"""Unit tests for YouTube transcript loader, indexer, and citation formatting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from pinrag.indexing.youtube_loader import (
    _fetch_video_title,
    extract_playlist_id,
    extract_video_id,
    fetch_playlist_info,
    load_youtube_transcript_as_documents,
)
from pinrag.rag.formatting import format_docs, format_sources

# --- extract_video_id ---


def test_extract_video_id_bare_id() -> None:
    """extract_video_id returns bare 11-char video ID."""
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("abc123XYZ-_") == "abc123XYZ-_"


def test_extract_video_id_youtube_watch() -> None:
    """extract_video_id extracts from youtube.com/watch?v=ID."""
    assert (
        extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    )
    assert extract_video_id("https://youtube.com/watch?v=xyz12345678") == "xyz12345678"


def test_extract_video_id_youtu_be() -> None:
    """extract_video_id extracts from youtu.be/ID."""
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_embed() -> None:
    """extract_video_id extracts from youtube.com/embed/ID."""
    assert (
        extract_video_id("https://www.youtube.com/embed/abc12345678") == "abc12345678"
    )


def test_extract_video_id_invalid_returns_none() -> None:
    """extract_video_id returns None for invalid input."""
    assert extract_video_id("") is None
    assert extract_video_id("   ") is None
    assert extract_video_id("https://example.com/foo") is None
    assert extract_video_id("short") is None
    assert extract_video_id("toolong12345") is None


# --- extract_playlist_id ---


def test_extract_playlist_id_playlist_url() -> None:
    """extract_playlist_id extracts from playlist URL."""
    assert (
        extract_playlist_id(
            "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdfg"
        )
        == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdfg"
    )


def test_extract_playlist_id_watch_with_list() -> None:
    """extract_playlist_id extracts from watch URL with list param."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdfg"
    assert extract_playlist_id(url) == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdfg"


def test_extract_playlist_id_single_video_returns_none() -> None:
    """extract_playlist_id returns None for single video URL."""
    assert extract_playlist_id("https://youtu.be/dQw4w9WgXcQ") is None
    assert extract_playlist_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is None


def test_extract_playlist_id_invalid_returns_none() -> None:
    """extract_playlist_id returns None for non-YouTube URLs."""
    assert extract_playlist_id("") is None
    assert extract_playlist_id("https://example.com/foo") is None


def test_detect_source_format_youtube_playlist() -> None:
    """_detect_source_format returns 'youtube_playlist' only for dedicated playlist URLs."""
    from pinrag.mcp.tools import _detect_source_format

    assert (
        _detect_source_format(
            "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdfg"
        )
        == "youtube_playlist"
    )


def test_detect_source_format_watch_url_with_list_is_single_video() -> None:
    """_detect_source_format returns 'youtube' for watch URLs with both v= and list= (single video)."""
    from pinrag.mcp.tools import _detect_source_format

    assert (
        _detect_source_format(
            "https://www.youtube.com/watch?v=abc12345678&list=PLtest123"
        )
        == "youtube"
    )
    assert (
        _detect_source_format(
            "https://www.youtube.com/watch?v=wxV6x5BUMH4&list=PLvCRDUYedILfHDoD57Yj8BAXNmNJLVM2r&index=3"
        )
        == "youtube"
    )


# --- fetch_playlist_info ---


@patch("yt_dlp.YoutubeDL")
def test_fetch_playlist_info_success(mock_ydl_cls: MagicMock) -> None:
    """fetch_playlist_info returns playlist title and video IDs."""
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=None)
    mock_ydl.extract_info.return_value = {
        "title": "My Playlist",
        "entries": [
            {"id": "abc12345678", "title": "Video 1"},
            {"id": "xyz98765432", "title": "Video 2"},
        ],
    }
    mock_ydl_cls.return_value = mock_ydl

    result = fetch_playlist_info("PLtest123")

    assert result["playlist_id"] == "PLtest123"
    assert result["playlist_title"] == "My Playlist"
    assert result["video_ids"] == ["abc12345678", "xyz98765432"]


@patch("yt_dlp.YoutubeDL")
def test_fetch_playlist_info_skips_none_entries(mock_ydl_cls: MagicMock) -> None:
    """fetch_playlist_info skips None entries (deleted/private videos)."""
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=None)
    mock_ydl.extract_info.return_value = {
        "title": "Playlist",
        "entries": [{"id": "abc12345678"}, None, {"id": "xyz98765432"}],
    }
    mock_ydl_cls.return_value = mock_ydl

    result = fetch_playlist_info("PLx")

    assert result["video_ids"] == ["abc12345678", "xyz98765432"]


def test_fetch_playlist_info_empty_id_raises() -> None:
    """fetch_playlist_info raises ValueError for empty playlist_id."""
    with pytest.raises(ValueError, match="playlist_id cannot be empty"):
        fetch_playlist_info("")


# --- _fetch_video_title ---


def test_fetch_video_title_success() -> None:
    """_fetch_video_title returns title from oEmbed when available."""

    class FakeResponse:
        def read(self):
            return b'{"title": "Rick Astley - Never Gonna Give You Up"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch(
        "pinrag.indexing.youtube_loader.urllib.request.urlopen",
        return_value=FakeResponse(),
    ):
        title = _fetch_video_title("dQw4w9WgXcQ")
    assert title == "Rick Astley - Never Gonna Give You Up"


def test_fetch_video_title_returns_none_on_error() -> None:
    """_fetch_video_title returns None when fetch fails."""
    with patch(
        "pinrag.indexing.youtube_loader.urllib.request.urlopen",
        side_effect=Exception("network error"),
    ):
        title = _fetch_video_title("dQw4w9WgXcQ")
    assert title is None


# --- load_youtube_transcript_as_documents ---


def test_load_youtube_transcript_invalid_url_raises() -> None:
    """load_youtube_transcript_as_documents raises ValueError for invalid URL."""
    with pytest.raises(ValueError, match="Invalid YouTube URL or video ID"):
        load_youtube_transcript_as_documents("not-a-youtube-url")


def test_load_youtube_transcript_success() -> None:
    """load_youtube_transcript_as_documents returns Documents with metadata."""
    mock_segment = MagicMock()
    mock_segment.text = "Hello world"
    mock_segment.start = 0.0
    mock_segment.duration = 2.5

    with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.return_value = [mock_segment]
        mock_api.return_value = mock_instance
        with patch(
            "pinrag.indexing.youtube_loader._fetch_video_title",
            return_value="Test Video Title",
        ):
            result = load_youtube_transcript_as_documents(
                "https://youtu.be/dQw4w9WgXcQ"
            )

    assert result.video_id == "dQw4w9WgXcQ"
    assert result.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert result.title == "Test Video Title"
    assert result.total_segments == 1
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.page_content == "Hello world"
    assert doc.metadata["document_id"] == "dQw4w9WgXcQ"
    assert doc.metadata["document_type"] == "youtube"
    assert doc.metadata["video_id"] == "dQw4w9WgXcQ"
    assert doc.metadata["title"] == "Test Video Title"
    assert doc.metadata["start"] == 0
    assert doc.metadata["duration"] == 2.5
    assert "youtube.com" in doc.metadata["source"]


def test_load_youtube_transcript_transcripts_disabled_raises() -> None:
    """load_youtube_transcript_as_documents raises RuntimeError when transcripts disabled."""
    from youtube_transcript_api import TranscriptsDisabled

    with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.side_effect = TranscriptsDisabled("xyz")
        mock_instance.list.side_effect = TranscriptsDisabled("xyz")
        mock_api.return_value = mock_instance

        with pytest.raises(RuntimeError, match="transcripts disabled"):
            load_youtube_transcript_as_documents("https://youtu.be/xyz12345678")


def test_load_youtube_transcript_no_transcript_raises() -> None:
    """load_youtube_transcript_as_documents raises RuntimeError when no transcript found."""
    from youtube_transcript_api import NoTranscriptFound

    with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.side_effect = NoTranscriptFound("xyz", None, None)
        mock_instance.list.side_effect = NoTranscriptFound("xyz", None, None)
        mock_api.return_value = mock_instance

        with pytest.raises(RuntimeError, match="No transcript found"):
            load_youtube_transcript_as_documents("https://youtu.be/xyz12345678")


def test_load_youtube_transcript_none_without_error_raises() -> None:
    """load_youtube_transcript_as_documents raises RuntimeError when transcript is None and last_error is None."""
    mock_fetch_none = MagicMock()
    mock_fetch_none.fetch.return_value = None
    mock_transcript_item = MagicMock()
    mock_transcript_item.fetch.return_value = None

    transcript_list = MagicMock()
    transcript_list._manually_created_transcripts = {}
    transcript_list._generated_transcripts = {}
    transcript_list.find_manually_created_transcript.return_value = mock_fetch_none
    transcript_list.find_generated_transcript.return_value = mock_fetch_none
    transcript_list.__iter__ = lambda self: iter([mock_transcript_item])

    with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.list.return_value = transcript_list
        mock_api.return_value = mock_instance

        with pytest.raises(
            RuntimeError, match="No transcript available and no error was captured"
        ):
            load_youtube_transcript_as_documents(
                "https://youtu.be/xyz12345678", languages=()
            )


# --- format_sources (timestamp citations) ---


def test_format_sources_includes_start_for_youtube() -> None:
    """format_sources returns start when present (YouTube chunks)."""
    docs = [
        Document(
            page_content="Hello",
            metadata={"document_id": "vid1", "start": 83, "page": 1},
        ),
        Document(page_content="World", metadata={"document_id": "vid1", "start": 120}),
    ]
    sources = format_sources(docs)
    assert len(sources) == 2
    assert sources[0] == {"document_id": "vid1", "page": 0, "start": 83}
    assert sources[1] == {"document_id": "vid1", "page": 0, "start": 120}


def test_format_sources_includes_title_for_youtube_when_in_metadata() -> None:
    """format_sources adds title for YouTube when doc_title or title is on chunks."""
    docs = [
        Document(
            page_content="A",
            metadata={
                "document_id": "vid1",
                "document_type": "youtube",
                "start": 10,
                "doc_title": "Learn Rust",
            },
        ),
        Document(
            page_content="B",
            metadata={
                "document_id": "vid2",
                "document_type": "youtube",
                "start": 20,
                "title": "Only title key",
            },
        ),
    ]
    sources = format_sources(docs)
    assert sources[0] == {
        "document_id": "vid1",
        "page": 0,
        "start": 10,
        "title": "Learn Rust",
    }
    assert sources[1] == {
        "document_id": "vid2",
        "page": 0,
        "start": 20,
        "title": "Only title key",
    }


def test_format_sources_page_for_pdf() -> None:
    """format_sources returns page for PDF chunks (no start)."""
    docs = [
        Document(page_content="Page 1", metadata={"document_id": "doc.pdf", "page": 1}),
        Document(page_content="Page 2", metadata={"document_id": "doc.pdf", "page": 2}),
    ]
    sources = format_sources(docs)
    assert sources[0] == {"document_id": "doc.pdf", "page": 1}
    assert sources[1] == {"document_id": "doc.pdf", "page": 2}


def test_format_docs_citation_label_timestamp() -> None:
    """format_docs uses t. M:SS for chunks with start (YouTube)."""
    docs = [
        Document(page_content="Hello", metadata={"document_id": "vid1", "start": 83}),
    ]
    out = format_docs(docs, number_chunks=True)
    assert "t. 1:23" in out
    assert "doc: vid1" in out


def test_format_docs_citation_label_page() -> None:
    """format_docs uses p. N for chunks with page (PDF)."""
    docs = [
        Document(
            page_content="Page 16", metadata={"document_id": "doc.pdf", "page": 16}
        ),
    ]
    out = format_docs(docs, number_chunks=True)
    assert "p. 16" in out
    assert "doc: doc.pdf" in out


def test_format_docs_citation_label_plaintext() -> None:
    """format_docs uses document_id for plaintext chunks (no page)."""
    docs = [
        Document(
            page_content="Hello world",
            metadata={"document_id": "notes.txt", "document_type": "plaintext"},
        ),
    ]
    out = format_docs(docs, number_chunks=True)
    assert "notes.txt" in out
    assert "p. " not in out  # plaintext: no page number
