"""Transport-agnostic PinRAG operations (query, index, list, remove)."""

from pinrag.core.format_detection import (
    GITHUB_OWNER_RE,
    GITHUB_REPO_RE,
    categorize_failures,
    detect_file_format,
    detect_source_format,
    is_github_url,
    resolve_persist_dir_path,
    resolve_user_content_path,
)
from pinrag.core.operations import (
    VerboseSyncEmitter,
    add_file,
    add_files,
    backfill_pdf_doc_titles,
    list_collections,
    list_documents,
    query,
    remove_document,
    set_document_tag,
)

__all__ = [
    "GITHUB_OWNER_RE",
    "GITHUB_REPO_RE",
    "VerboseSyncEmitter",
    "add_file",
    "add_files",
    "backfill_pdf_doc_titles",
    "categorize_failures",
    "detect_file_format",
    "detect_source_format",
    "is_github_url",
    "list_collections",
    "list_documents",
    "query",
    "remove_document",
    "set_document_tag",
    "resolve_persist_dir_path",
    "resolve_user_content_path",
]
