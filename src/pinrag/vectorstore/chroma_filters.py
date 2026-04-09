"""Chroma metadata ``where`` filters for retrieval and indexing (dedup, scoped search)."""

from __future__ import annotations


def build_retrieval_filter(
    document_id: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    tag: str | None = None,
    document_type: str | None = None,
) -> dict | None:
    """Build Chroma where filter from document_id, page range, tag, and/or document_type.

    Single page: page_min=64, page_max=64 filters to page 64 only.
    Tag filter returns only chunks with that tag (documents indexed with tag).
    document_type: "pdf", "youtube", "discord", "github", or "plaintext" to filter by source type.
    """
    conditions: list[dict] = []
    if document_id and str(document_id).strip():
        conditions.append({"document_id": document_id.strip()})
    if page_min is not None:
        conditions.append({"page": {"$gte": page_min}})
    if page_max is not None:
        conditions.append({"page": {"$lte": page_max}})
    if tag and str(tag).strip():
        conditions.append({"tag": tag.strip()})
    if document_type and str(document_type).strip():
        conditions.append({"document_type": document_type.strip()})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
