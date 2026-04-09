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
    list_documents,
    query,
    remove_document,
)

__all__ = [
    "GITHUB_OWNER_RE",
    "GITHUB_REPO_RE",
    "VerboseSyncEmitter",
    "add_file",
    "add_files",
    "categorize_failures",
    "detect_file_format",
    "detect_source_format",
    "is_github_url",
    "list_documents",
    "query",
    "remove_document",
    "resolve_persist_dir_path",
    "resolve_user_content_path",
]
