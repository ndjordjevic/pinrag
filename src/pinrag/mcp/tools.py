"""Backward-compatible re-exports — core logic lives in pinrag.core."""

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

# Legacy private names (tests / callers that had imports from this module)
_GITHUB_OWNER_RE = GITHUB_OWNER_RE
_GITHUB_REPO_RE = GITHUB_REPO_RE
_categorize_failures = categorize_failures
_detect_file_format = detect_file_format
_detect_source_format = detect_source_format
_is_github_url = is_github_url
_resolve_persist_dir_path = resolve_persist_dir_path
_resolve_user_content_path = resolve_user_content_path

__all__ = [
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
    # Legacy aliases
    "_GITHUB_OWNER_RE",
    "_GITHUB_REPO_RE",
    "_categorize_failures",
    "_detect_file_format",
    "_detect_source_format",
    "_is_github_url",
    "_resolve_persist_dir_path",
    "_resolve_user_content_path",
]
