"""Format retrieved documents as context and as citation sources."""

from __future__ import annotations

from langchain_core.documents import Document


def _format_timestamp(seconds: int | float) -> str:
    """Format seconds as M:SS (e.g. 83 -> '1:23')."""
    total_secs = int(round(float(seconds)))
    m, s = divmod(total_secs, 60)
    if m > 0:
        return f"{m}:{s:02d}"
    return f"0:{s:02d}"


def _youtube_display_title(meta: dict) -> str | None:
    """Human-readable video title for YouTube chunks, if present."""
    if meta.get("document_type") != "youtube":
        return None
    raw = meta.get("doc_title") or meta.get("title")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _citation_label(meta: dict) -> str:
    """Return citation label: 'p. N' for PDF, 't. M:SS' for YouTube, chunk index for GitHub, else '?'."""
    start = meta.get("start")
    if start is not None:
        try:
            return f"t. {_format_timestamp(start)}"
        except (TypeError, ValueError):
            pass
    if meta.get("document_type") == "github":
        ci = meta.get("chunk_index")
        if ci is not None:
            return f"p. {ci}"
        page = meta.get("page", "?")
        return f"p. {page}"
    if meta.get("document_type") == "discord":
        channel = meta.get("channel")
        return f"channel: {channel}" if channel else "discord"
    if meta.get("document_type") == "plaintext":
        return meta.get("document_id") or "plaintext"
    page = meta.get("page", "?")
    return f"p. {page}"


def format_docs(docs: list[Document], *, number_chunks: bool = False) -> str:
    """Turn a list of chunk documents into a single context string for the prompt.

    Each chunk is separated by a blank line. If number_chunks is True, each block
    is prefixed with [N] (doc: <document_id>, <label>). By default only chunk text
    is included; use sources / citation UX outside the LLM for provenance.

    Args:
        docs: Retrieved chunk documents (with metadata such as page, start, document_id).
        number_chunks: Whether to add [1], [2], ... and doc/label (off by default).

    Returns:
        A single string suitable for the {context} placeholder in the RAG prompt.

    """
    if not docs:
        return "No relevant context found."

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        doc_id = (
            meta.get("document_id")
            or meta.get("file_name")
            or meta.get("source")
            or "?"
        )
        label = _citation_label(meta)
        if number_chunks:
            parts.append(f"[{i}] (doc: {doc_id}, {label})\n{doc.page_content}")
        else:
            parts.append(doc.page_content)

    return "\n\n".join(parts)


def format_sources(docs: list[Document]) -> list[dict[str, str | int]]:
    """Build a list of unique source references from retrieved documents for citations.

    Deduplicates by (document_id, page_or_start_or_source). Each item has "document_id",
    "document_type" when known, "page" (for PDFs), "start" when present (YouTube timestamp),
    "source" for web pages / GitHub blob URLs / plaintext paths, optional "channel" and
    message range fields for Discord, and optional "title" for YouTube.

    Args:
        docs: Retrieved chunk documents.

    Returns:
        List of dicts suitable for query citations and CLI source tables.

    """
    seen: set[tuple[str, int | str]] = set()
    out: list[dict[str, str | int]] = []
    for doc in docs:
        meta = doc.metadata
        doc_id = str(
            meta.get("document_id")
            or meta.get("file_name")
            or meta.get("source")
            or "?"
        )
        start = meta.get("start")
        if start is not None:
            try:
                start_int = int(round(float(start)))
                key = (doc_id, start_int)
                if key not in seen:
                    seen.add(key)
                    item: dict[str, str | int] = {
                        "document_id": doc_id,
                        "page": 0,
                        "start": start_int,
                        "document_type": str(meta.get("document_type") or "youtube"),
                    }
                    yt_title = _youtube_display_title(meta)
                    if yt_title:
                        item["title"] = yt_title
                    out.append(item)
            except (TypeError, ValueError):
                key = (doc_id, 0)
                if key not in seen:
                    seen.add(key)
                    item = {
                        "document_id": doc_id,
                        "page": 0,
                        "document_type": str(meta.get("document_type") or "youtube"),
                    }
                    yt_title = _youtube_display_title(meta)
                    if yt_title:
                        item["title"] = yt_title
                    out.append(item)
        elif meta.get("document_type") == "github":
            src = str(meta.get("source") or "")
            key = (doc_id, src) if src else (doc_id, 0)
            if key not in seen:
                seen.add(key)
                item = {
                    "document_id": doc_id,
                    "page": 0,
                    "document_type": "github",
                }
                if src:
                    item["source"] = src
                yt_title = _youtube_display_title(meta)
                if yt_title:
                    item["title"] = yt_title
                out.append(item)
        elif meta.get("document_type") == "web":
            src = str(meta.get("source") or meta.get("source_url") or "")
            ci = meta.get("chunk_index")
            try:
                ci_k = int(ci) if ci is not None else 0
            except (TypeError, ValueError):
                ci_k = 0
            key = (doc_id, src) if src else (doc_id, ci_k)
            if key not in seen:
                seen.add(key)
                item = {"document_id": doc_id, "page": 0, "document_type": "web"}
                if src:
                    item["source"] = src
                yt_title = _youtube_display_title(meta)
                if yt_title:
                    item["title"] = yt_title
                out.append(item)
        elif meta.get("document_type") == "discord":
            ms_raw = meta.get("message_start")
            me_raw = meta.get("message_end")
            try:
                ms_i = int(ms_raw) if ms_raw is not None else None
            except (TypeError, ValueError):
                ms_i = None
            try:
                me_i = int(me_raw) if me_raw is not None else None
            except (TypeError, ValueError):
                me_i = None
            ci_fallback = meta.get("chunk_index")
            key = (
                (doc_id, ms_i)
                if ms_i is not None
                else (doc_id, int(ci_fallback) if ci_fallback is not None else 0)
            )
            if key not in seen:
                seen.add(key)
                item = {"document_id": doc_id, "page": 0, "document_type": "discord"}
                if ms_i is not None:
                    item["message_start"] = ms_i
                if me_i is not None:
                    item["message_end"] = me_i
                ch = meta.get("channel")
                if ch:
                    item["channel"] = str(ch)
                yt_title = _youtube_display_title(meta)
                if yt_title:
                    item["title"] = yt_title
                out.append(item)
        elif meta.get("document_type") == "plaintext":
            src = str(meta.get("source") or "")
            try:
                page = int(meta.get("page", 0))
            except (TypeError, ValueError):
                page = 0
            key = (doc_id, src) if src else (doc_id, page)
            if key not in seen:
                seen.add(key)
                item = {"document_id": doc_id, "page": page, "document_type": "plaintext"}
                if src:
                    item["source"] = src
                yt_title = _youtube_display_title(meta)
                if yt_title:
                    item["title"] = yt_title
                out.append(item)
        else:
            try:
                page = int(meta.get("page", 0))
            except (TypeError, ValueError):
                page = 0
            key = (doc_id, page)
            if key not in seen:
                seen.add(key)
                dtype = meta.get("document_type")
                if not dtype:
                    dtype = "pdf" if page > 0 else "unknown"
                item = {"document_id": doc_id, "page": page, "document_type": str(dtype)}
                yt_title = _youtube_display_title(meta)
                if yt_title:
                    item["title"] = yt_title
                out.append(item)
    return out
