"""Tests for verbose MCP logging collection and phase emissions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from pinrag.indexing.youtube_indexer import YouTubeIndexResult
from pinrag.mcp import tools as mcp_tools
from tests.helpers.mcp_patched_server import mcp_logging, mcp_server


def test_emit_verbose_noop_when_disabled() -> None:
    """emit_verbose should no-op when PINRAG_VERBOSE_LOGGING is off."""
    ctx = MagicMock()
    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=False):
        asyncio.run(mcp_logging.emit_verbose(ctx, "phase=disabled-test"))
    ctx.log.assert_not_called()


def test_emit_verbose_noop_when_enabled() -> None:
    """emit_verbose remains a no-op even when verbose logging is enabled."""
    ctx = MagicMock()
    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True):
        asyncio.run(mcp_logging.emit_verbose(ctx, "phase=emit-test"))
    ctx.log.assert_not_called()


def test_collector_buffers_and_drains() -> None:
    """VerboseCollector should buffer events and return them on drain."""
    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True):
        collector = mcp_logging.VerboseCollector(scope="tool", name="test_tool")
        collector.emit("phase=a")
        collector.emit("phase=b")
        collector.emit("phase=c")
        drained = collector.drain()
    assert drained == [
        "tool=test_tool phase=a",
        "tool=test_tool phase=b",
        "tool=test_tool phase=c",
    ]


def test_collector_drain_returns_empty_when_empty() -> None:
    """Draining an empty collector should return an empty list."""
    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True):
        collector = mcp_logging.VerboseCollector(scope="tool", name="test")
        drained = collector.drain()
    assert drained == []


def test_make_verbose_emitter_returns_collector_backed_emitter() -> None:
    """make_verbose_emitter should return an emitter backed by a collector."""
    with patch(
        "pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True
    ):
        emitter = mcp_logging.make_verbose_emitter(
            MagicMock(), scope="tool", name="add_document_tool"
        )
        assert hasattr(emitter, "_collector")
        collector = emitter._collector  # type: ignore[attr-defined]
        emitter("phase=step1")
        emitter("phase=step2")
        drained = collector.drain()
        assert any("phase=step1" in line for line in drained)
        assert any("phase=step2" in line for line in drained)


def test_make_verbose_emitter_from_worker_thread() -> None:
    """Worker-thread emitter should collect events and drain correctly."""

    async def _run() -> None:
        ctx = MagicMock()
        with patch(
            "pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True
        ):
            emitter = mcp_logging.make_verbose_emitter(
                ctx, scope="tool", name="add_document_tool"
            )
            await asyncio.to_thread(emitter, "phase=a", "info")
            await asyncio.to_thread(emitter, "phase=b", "info")
            collector = emitter._collector  # type: ignore[attr-defined]
            drained = collector.drain()
            assert any("phase=a" in line for line in drained)
            assert any("phase=b" in line for line in drained)

    asyncio.run(_run())


def test_add_document_tool_includes_verbose_log_when_enabled_for_url() -> None:
    """add_document_tool responses include _verbose_log when verbose logging is enabled."""

    def _fake_add_files(*args, **kwargs) -> dict:
        verbose_emitter = kwargs.get("verbose_emitter")
        if verbose_emitter is not None:
            verbose_emitter("phase=one", "info")
            verbose_emitter("phase=two", "info")
        return {"indexed": [], "failed": [], "total_indexed": 0, "total_failed": 0}

    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True):
        with patch("pinrag.mcp.server.add_files", side_effect=_fake_add_files):
            with patch("pinrag.mcp.server.config.get_persist_dir", return_value="/tmp"):
                with patch(
                    "pinrag.mcp.server.config.get_collection_name", return_value="pinrag"
                ):
                    out = asyncio.run(
                        mcp_server.add_document_tool(
                            paths=["https://youtu.be/demo"]
                        )
                    )

    assert "_verbose_log" in out
    assert "_server_version" in out
    assert out["_verbose_log"] == [
        "tool=add_document_tool phase=one",
        "tool=add_document_tool phase=two",
    ]


def test_add_document_tool_omits_verbose_log_when_disabled_for_url() -> None:
    """add_document_tool omits _verbose_log when verbose logging is disabled."""

    def _fake_add_files(*args, **kwargs) -> dict:
        verbose_emitter = kwargs.get("verbose_emitter")
        if verbose_emitter is not None:
            verbose_emitter("phase=one", "info")
        return {"indexed": [], "failed": [], "total_indexed": 0, "total_failed": 0}

    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=False):
        with patch("pinrag.mcp.server.add_files", side_effect=_fake_add_files):
            with patch("pinrag.mcp.server.config.get_persist_dir", return_value="/tmp"):
                with patch(
                    "pinrag.mcp.server.config.get_collection_name", return_value="pinrag"
                ):
                    out = asyncio.run(
                        mcp_server.add_document_tool(
                            paths=["https://youtu.be/demo"]
                        )
                    )

    assert "_server_version" in out
    assert "_verbose_log" not in out


def test_add_document_tool_mirrors_verbose_to_output_on_error_for_url() -> None:
    """Verbose lines are mirrored to stderr when add_document_tool raises."""

    def _fake_add_files(*args, **kwargs) -> dict:
        verbose_emitter = kwargs.get("verbose_emitter")
        if verbose_emitter is not None:
            verbose_emitter("phase=before_error", "info")
        raise RuntimeError("boom")

    with patch("pinrag.mcp.logging_utils.config.get_verbose_logging", return_value=True):
        with patch("pinrag.mcp.server.add_files", side_effect=_fake_add_files):
            with patch(
                "pinrag.mcp.server.mirror_verbose_to_output_panel"
            ) as mirror_mock:
                with patch(
                    "pinrag.mcp.server.config.get_persist_dir", return_value="/tmp"
                ):
                    with patch(
                        "pinrag.mcp.server.config.get_collection_name",
                        return_value="pinrag",
                    ):
                        try:
                            asyncio.run(
                                mcp_server.add_document_tool(
                                    paths=["https://youtu.be/demo"]
                                )
                            )
                            assert False, "expected RuntimeError"
                        except RuntimeError as exc:
                            assert str(exc) == "boom"

    mirror_mock.assert_called_once_with(
        ["tool=add_document_tool phase=before_error"]
    )


def test_add_files_emits_verbose_phase_events() -> None:
    """add_files should emit verbose path lifecycle events."""
    events: list[tuple[str, str]] = []

    def _emit(message: str, level: str = "debug") -> None:
        events.append((message, level))

    fake_result = {"indexed": [{"path": "x"}], "failed": []}
    with patch("pinrag.core.operations.add_file", return_value=fake_result):
        mcp_tools.add_files(
            paths=["https://youtu.be/demo"],
            persist_dir="/tmp",
            collection="pinrag",
            verbose_emitter=_emit,
        )

    messages = [m for m, _ in events]
    assert any("phase=add_files_start" in m for m in messages)
    assert any("phase=add_files_path_start" in m for m in messages)
    assert any("phase=add_files_path_done" in m for m in messages)
    assert any("phase=add_files_done" in m for m in messages)


def test_add_file_youtube_emits_verbose_vision_related_phases() -> None:
    """add_file should emit youtube phase events and pass verbose emitter down."""
    events: list[tuple[str, str]] = []

    def _emit(message: str, level: str = "debug") -> None:
        events.append((message, level))

    fake_result = YouTubeIndexResult(
        video_id="abc123xyz00",
        source_url="https://www.youtube.com/watch?v=abc123xyz00",
        total_segments=12,
        total_chunks=16,
        persist_directory=Path("/tmp"),
        collection_name="pinrag",
        title="Demo",
    )
    with patch("pinrag.core.operations.detect_source_format", return_value="youtube"):
        with patch(
            "pinrag.core.operations.get_embedding_model", return_value=MagicMock()
        ):
            with patch("pinrag.core.operations.index_youtube", return_value=fake_result):
                mcp_tools.add_file(
                    path="https://youtu.be/abc123xyz00",
                    persist_dir="/tmp",
                    collection="pinrag",
                    verbose_emitter=_emit,
                )

    messages = [m for m, _ in events]
    assert any("phase=detect_format" in m and "youtube" in m for m in messages)
    assert any("phase=youtube_index_start" in m for m in messages)
    assert any("phase=youtube_index_done" in m for m in messages)
