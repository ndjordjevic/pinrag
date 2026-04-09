"""Logging configuration and MCP tool notification helpers."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.context_injection import find_context_parameter

from pinrag import config

logger = logging.getLogger(__name__)

_PINRAG_TQDM_PATCHED = False


def _suppress_chroma_telemetry() -> None:
    """Disable Chroma telemetry to avoid PostHog INFO messages in Cursor Output."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _suppress_progress_bars() -> None:
    """Silence tqdm on stderr for embeddings without forcing ``TQDM_DISABLE`` into the environment.

    If ``TQDM_DISABLE`` is unset, patch ``tqdm`` so default ``disable=None`` becomes disabled.
    Explicit ``TQDM_DISABLE=1`` / ``0`` / etc. is left to tqdm's own env handling (no patch).
    """
    global _PINRAG_TQDM_PATCHED
    raw = (os.environ.get("TQDM_DISABLE") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return
    if raw in ("1", "true", "yes", "on"):
        return
    if _PINRAG_TQDM_PATCHED:
        return
    try:
        from tqdm import tqdm
    except ImportError:
        return

    _orig = tqdm.__init__

    def _pinrag_tqdm_init(self, *args, disable=None, **kwargs):  # type: ignore[no-untyped-def]
        if disable is None:
            disable = True
        return _orig(self, *args, disable=disable, **kwargs)

    tqdm.__init__ = _pinrag_tqdm_init  # type: ignore[method-assign]
    _PINRAG_TQDM_PATCHED = True


def _configure_pinrag_logger() -> None:
    """Suppress PinRAG Python logging; MCP notifications are the primary channel."""
    root = logging.getLogger("pinrag")
    root.handlers.clear()
    root.propagate = False
    root.setLevel(logging.CRITICAL + 1)


def _suppress_dependency_loggers() -> None:
    """Suppress verbose logs from dependencies that show as [error] in Cursor."""
    for name in ("mcp", "chromadb", "posthog", "openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Rotated text.*")


def configure_logging() -> None:
    """Configure PinRAG logging and dependency suppression after .env is loaded."""
    _suppress_chroma_telemetry()
    _suppress_progress_bars()
    _configure_pinrag_logger()
    _suppress_dependency_loggers()


def _user_friendly_api_error(exc: Exception) -> str | None:
    """Return a short, user-friendly message for API key / auth errors, or None."""
    msg = str(exc).lower()
    if (
        "api key" in msg
        or "api_key" in msg
        or "authentication" in msg
        or "invalid key" in msg
        or "no api key" in msg
    ):
        if "anthropic" in msg or "claude" in msg:
            return "Anthropic API key missing or invalid. Set ANTHROPIC_API_KEY in ~/.pinrag/.env (or your config)."
        if (os.environ.get("PINRAG_LLM_PROVIDER") or "").strip().lower() == "cerebras":
            return "Cerebras API key missing or invalid. Set CEREBRAS_API_KEY in ~/.pinrag/.env (or your config)."
        return "OpenAI API key not found or invalid. Set OPENAI_API_KEY in ~/.pinrag/.env (or your config)."
    if "rate" in msg and "limit" in msg:
        return "API rate limit exceeded. Please try again in a moment."
    return None


def _build_tool_summary(name: str, kwargs: dict[str, Any]) -> str:
    """Build a short summary string for tool logging. Always redacts sensitive content."""
    if name == "add_document_tool":
        paths = kwargs.get("paths", [])
        n = len(paths)
        if not paths:
            return "paths=[]"
        if n == 1:
            p = paths[0]
            preview = p[:60] + "..." if len(p) > 60 else p
            return f"paths={preview!r}"
        return f"paths={n} paths"
    if name == "query_tool":
        q = (kwargs.get("query") or "").strip()
        return f"query=<{len(q)} chars>" if q else "query=<empty>"
    if name == "list_documents_tool":
        return f"tag={kwargs.get('tag', '')!r}"
    if name == "remove_document_tool":
        return f"document_id={kwargs.get('document_id', '')!r}"
    if name == "set_document_tag_tool":
        return (
            f"document_id={kwargs.get('document_id', '')!r} "
            f"tag={kwargs.get('tag', '')!r}"
        )
    return ""


_ClientLogLevel = Literal["debug", "info", "warning", "error"]
_VALID_CLIENT_LOG_LEVELS: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error"}
)
VerboseSyncEmitter = Callable[[str, str], None]


async def _emit_client_log(
    ctx: Context | None, message: str, level: str = "info"
) -> None:
    """Best-effort tool log emission to MCP clients via notifications/message."""
    if ctx is None:
        return
    try:
        effective_level = cast(
            _ClientLogLevel,
            level if level in _VALID_CLIENT_LOG_LEVELS else "info",
        )
        # FastMCP Context.log sends an MCP 'notifications/message'
        maybe_awaitable = ctx.log(effective_level, message, logger_name="pinrag")
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
    except Exception:
        logger.debug(
            "client_log_emit_failed level=%s message=%s", level, message, exc_info=True
        )


def _verbose_logging_enabled() -> bool:
    """Return whether verbose MCP notifications are enabled."""
    return config.get_verbose_logging()


class VerboseCollector:
    """Thread-safe collector that buffers verbose events and flushes once."""

    def __init__(self, scope: str, name: str) -> None:
        self._prefix = f"{scope}={name}"
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def emit(self, message: str, level: str = "info") -> None:
        if not _verbose_logging_enabled():
            return
        _ = level
        entry = f"{self._prefix} {message}".strip()
        with self._lock:
            self._lines.append(entry)

    def drain(self) -> list[str]:
        """Return buffered lines and clear the collector."""
        with self._lock:
            snapshot = list(self._lines)
            self._lines.clear()
        return snapshot


async def emit_verbose(ctx: Context | None, message: str, level: str = "info") -> None:
    """Best-effort verbose emission hook.

    Verbose events are surfaced through tool responses (``_verbose_log``).
    Resources return plain text, so this hook is intentionally a no-op.
    """
    if not _verbose_logging_enabled():
        return
    _ = (ctx, message, level)


def make_verbose_emitter(
    ctx: Context | None, *, scope: str, name: str
) -> VerboseSyncEmitter:
    """Create a worker-thread-safe emitter backed by a VerboseCollector.

    Call ``emitter._collector.drain()`` after the tool finishes to retrieve
    all buffered lines for inclusion in the tool response.
    """
    collector = VerboseCollector(scope, name)

    def _emit_sync(message: str, level: str = "info") -> None:
        collector.emit(message, level)

    _emit_sync._collector = collector  # type: ignore[attr-defined]
    return _emit_sync


def mirror_verbose_to_output_panel(lines: list[str]) -> None:
    """Mirror verbose lines to MCP Output panel via stderr.

    Cursor's stdio MCP Output panel only surfaces server-side stderr, so we emit
    one batched payload per tool call to keep logs visible where users look.
    """
    if not lines or not _verbose_logging_enabled():
        return
    payload = "[pinrag verbose]\n" + "\n".join(lines) + "\n"
    try:
        sys.stderr.write(payload)
        sys.stderr.flush()
    except Exception:
        logger.debug("stderr_verbose_mirror_failed", exc_info=True)


def _resolve_context_from_kwargs(fn: Any, kwargs: dict[str, Any]) -> Context | None:
    """Resolve Context from kwargs using annotation-based parameter lookup (FastMCP style)."""
    param_name = find_context_parameter(fn)
    if param_name is None:
        return None
    val = kwargs.get(param_name)
    return val if isinstance(val, Context) else None


def _log_tool_errors(fn):
    """Log tool entry, success, and errors to MCP clients via Context notifications.

    Preserves the wrapped function's signature so MCP/FastMCP schema introspection sees all parameters.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        summary = _build_tool_summary(fn.__name__, kwargs)
        ctx = _resolve_context_from_kwargs(fn, kwargs)
        start_msg = f"tool={fn.__name__} status=start {summary}".strip()
        await _emit_client_log(ctx, start_msg)
        start = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
            elapsed_s = time.perf_counter() - start
            await _emit_client_log(
                ctx, f"tool={fn.__name__} status=ok duration_s={elapsed_s:.2f}"
            )
            return result
        except Exception as e:
            elapsed_s = time.perf_counter() - start
            await _emit_client_log(
                ctx,
                f"tool={fn.__name__} status=error duration_s={elapsed_s:.2f}",
                level="error",
            )
            friendly = _user_friendly_api_error(e)
            if friendly:
                raise RuntimeError(friendly) from e
            raise

    wrapper.__signature__ = inspect.signature(fn)
    wrapper.__annotations__ = fn.__annotations__
    return wrapper
