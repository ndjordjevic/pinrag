"""Load GitHub repository contents into LangChain Documents for RAG indexing."""

from __future__ import annotations

import base64
import fnmatch
import logging
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

import requests
from langchain_core.documents import Document

from pinrag.config import get_github_default_branch

logger = logging.getLogger(__name__)


# Default exclude: lock files, caches, binaries, large artifacts
_DEFAULT_EXCLUDE = (
    ".git/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "node_modules/",
    ".venv/",
    "venv/",
    "uv.lock",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.pdf",
    "*.mp4",
    "*.webm",
    "*.mp3",
    "*.wav",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.rar",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
)

# Extensions we consider text (include by default if no include pattern)
_TEXT_EXTENSIONS = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".md",
        ".mdx",
        ".rst",
        ".txt",
        ".adoc",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".go",
        ".rs",
        ".rb",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".cxx",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".vue",
        ".svelte",
        ".astro",
        ".gitignore",
        ".pio",
        ".cmake",
        ".ld",
        ".s",
        ".asm",
    }
)
_TEXT_BASENAMES = frozenset({"dockerfile", ".env.example"})


def _parse_github_url(url: str) -> tuple[str, str, str]:
    """Parse GitHub URL into (owner, repo, branch). Branch defaults to configured default branch."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError(
            f"Invalid GitHub URL: {url!r}. "
            "Expected https://github.com/owner/repo or github.com/owner/repo/tree/branch"
        )
    if raw.startswith("//"):
        raw = "https:" + raw
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid GitHub URL: {url!r}. "
            "Expected https://github.com/owner/repo or github.com/owner/repo/tree/branch"
        ) from e

    host = (parsed.netloc or "").lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host != "github.com":
        raise ValueError(
            f"Invalid GitHub URL: {url!r}. "
            "Expected https://github.com/owner/repo or github.com/owner/repo/tree/branch"
        )

    parts = [unquote(p) for p in (parsed.path or "").split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError(f"Invalid GitHub URL: {url!r}. Path cannot contain '..'")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid GitHub URL: {url!r}. "
            "Expected https://github.com/owner/repo or github.com/owner/repo/tree/branch"
        )
    owner, repo = parts[0], parts[1]
    if not owner or not repo:
        raise ValueError(f"Could not parse owner/repo from GitHub URL: {url!r}")
    # Remove .git suffix if present
    if repo.endswith(".git"):
        repo = repo[:-4]
    branch: str | None = None
    if len(parts) >= 3 and parts[2] == "tree":
        branch = "/".join(parts[3:]).strip("/") or None
    return owner, repo, branch or get_github_default_branch()


def _matches_patterns(path: str, include: list[str] | None, exclude: list[str]) -> bool:
    """Return True if path matches include (if set) and does not match exclude."""
    if include is not None and include:
        if not any(fnmatch.fnmatch(path, p) for p in include):
            return False
    parts = path.split("/")
    for p in exclude:
        p_stripped = p.rstrip("/")
        if p.endswith("/"):
            if (
                p_stripped in parts
                or path.startswith(p_stripped + "/")
                or path.startswith(p)
            ):
                return False
        else:
            if fnmatch.fnmatch(parts[-1] if parts else path, p):
                return False
    return True


def _is_default_text_file(path: str) -> bool:
    """Return True when path is considered text under default include behavior."""
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if name in _TEXT_BASENAMES:
        return True
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return ext in _TEXT_EXTENSIONS


def _infer_language(file_path: str) -> str:
    """Infer language from file extension."""
    ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    lang_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".md": "markdown",
        ".mdx": "markdown",
        ".rst": "rst",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".html": "html",
        ".css": "css",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".vue": "vue",
        ".svelte": "svelte",
        ".pio": "pio",
        ".cmake": "cmake",
        ".ld": "ld",
        ".s": "asm",
        ".asm": "asm",
    }
    return lang_map.get(ext, "plaintext")


@dataclass(frozen=True)
class GitHubLoadResult:
    """Result of loading a GitHub repo into Documents."""

    owner: str
    repo: str
    branch: str
    documents: list[Document]
    files_loaded: int
    total_chars: int
    failed_files: list[dict[str, str]] = field(default_factory=list)


def load_github_repo_as_documents(
    repo_url: str,
    *,
    token: str | None = None,
    branch: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_bytes: int = 524288,
) -> GitHubLoadResult:
    """Load a GitHub repository's text files into LangChain Documents.

    Uses GitHub API to list files (recursive tree) and fetch content via Contents API.
    Each file becomes one Document; chunking is done by the indexer.

    Args:
        repo_url: GitHub URL (e.g. https://github.com/owner/repo or github.com/owner/repo/tree/main).
        token: GitHub personal access token (optional for public repos; needed for private).
        branch: Override branch. If None, parsed from URL or defaults to main.
        include_patterns: Glob patterns for files to include (e.g. ["*.md", "src/**/*.py"]).
        exclude_patterns: Glob patterns to exclude. Default excludes lock files, caches, binaries.
        max_file_bytes: Skip files larger than this (default 512 KB).

    Returns:
        GitHubLoadResult with documents, owner, repo, branch, files_loaded, total_chars.

    Raises:
        ValueError: If repo_url is not a valid GitHub URL.
        RuntimeError: If API requests fail (404, 403, etc.).

    """
    owner, repo, default_branch = _parse_github_url(repo_url)
    use_branch = branch if branch else default_branch
    exclude = list(exclude_patterns) if exclude_patterns else list(_DEFAULT_EXCLUDE)

    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 1. Get tree SHA from branch
    commit_resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{use_branch}",
        headers=headers,
        timeout=30,
    )
    if commit_resp.status_code == 404:
        raise RuntimeError(
            f"GitHub repo or branch not found: {owner}/{repo} @ {use_branch}. "
            "Check the URL and branch. For private repos, set GITHUB_TOKEN."
        )
    if commit_resp.status_code == 403:
        raise RuntimeError(
            f"GitHub API rate limit or access denied for {owner}/{repo}. "
            "Set GITHUB_TOKEN for higher limits and private repo access."
        )
    commit_resp.raise_for_status()
    commit_data = commit_resp.json()
    tree_sha = commit_data.get("commit", {}).get("tree", {}).get("sha")
    if not tree_sha:
        raise RuntimeError(f"Could not get tree SHA for {owner}/{repo} @ {use_branch}")

    # 2. Get recursive tree
    tree_resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{tree_sha}",
        headers=headers,
        params={"recursive": "1"},
        timeout=60,
    )
    tree_resp.raise_for_status()
    tree_data = tree_resp.json()
    if tree_data.get("truncated"):
        logger.warning(
            "GitHub tree response truncated for %s/%s @ %s; index may be incomplete.",
            owner,
            repo,
            use_branch,
        )
    items = tree_data.get("tree", [])

    # 3. Filter to blobs (files) that match patterns and are text
    file_paths: list[str] = []
    for item in items:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path:
            continue
        if not _matches_patterns(path, include_patterns, exclude):
            continue
        if not include_patterns and not _is_default_text_file(path):
            continue
        file_paths.append(path)

    # 4. Fetch each file's content
    base_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    documents: list[Document] = []
    failed_files: list[dict[str, str]] = []
    total_chars = 0

    for path in file_paths:
        try:
            file_resp = requests.get(
                f"{base_url}/{path}",
                headers=headers,
                params={"ref": use_branch},
                timeout=30,
            )
            if file_resp.status_code != 200:
                failed_files.append(
                    {
                        "path": path,
                        "error": f"GitHub contents API returned HTTP {file_resp.status_code}",
                    }
                )
                continue
            file_data = file_resp.json()
            if file_data.get("type") != "file":
                continue
            size = file_data.get("size", 0)
            if size > max_file_bytes:
                continue
            content_b64 = file_data.get("content")
            if not content_b64:
                failed_files.append(
                    {
                        "path": path,
                        "error": "GitHub contents API returned empty file content",
                    }
                )
                continue
            try:
                raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug(
                    "Failed decoding GitHub content for %s/%s:%s", owner, repo, path
                )
                failed_files.append(
                    {
                        "path": path,
                        "error": f"Could not decode GitHub file content: {e}",
                    }
                )
                continue
            total_chars += len(raw)
            repo_id = f"{owner}/{repo}"
            source_url = f"https://github.com/{owner}/{repo}/blob/{use_branch}/{path}"
            meta = {
                "document_id": repo_id,
                "document_type": "github",
                "source": source_url,
                "repo": repo_id,
                "branch": use_branch,
                "language": _infer_language(path),
                "doc_bytes": size,
            }
            documents.append(Document(page_content=raw, metadata=meta))
        except Exception as e:
            logger.debug("Failed fetching GitHub file %s/%s:%s", owner, repo, path)
            failed_files.append({"path": path, "error": str(e)})
            continue

    if failed_files:
        logger.warning(
            "GitHub load partial failures for %s/%s @ %s: %d file(s) failed",
            owner,
            repo,
            use_branch,
            len(failed_files),
        )

    return GitHubLoadResult(
        owner=owner,
        repo=repo,
        branch=use_branch,
        documents=documents,
        files_loaded=len(documents),
        total_chars=total_chars,
        failed_files=failed_files,
    )
