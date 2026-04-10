"""Path resolution, URL validation, and file/source format detection for indexing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pinrag.indexing import extract_playlist_id, extract_video_id

# GitHub.com /owner/repo path segments (defense-in-depth; indexer also validates).
GITHUB_OWNER_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")
GITHUB_REPO_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?$")


def resolve_user_content_path(path_str: str) -> Path:
    """Resolve a local path for add/query tools; rejects null bytes and ``..`` segments."""
    if "\x00" in path_str:
        raise ValueError("Path cannot contain null bytes")
    expanded = Path(path_str).expanduser()
    if ".." in expanded.parts:
        raise ValueError("Path must not use parent directory segments (..)")
    return expanded.resolve()


def resolve_persist_dir_path(path_str: str) -> Path:
    """Resolve Chroma persist directory; rejects null bytes (``..`` allowed for config)."""
    if "\x00" in path_str:
        raise ValueError("Path cannot contain null bytes")
    return Path(path_str).expanduser().resolve()


def is_github_url(s: str) -> bool:
    """Return True if string is an https (or http) URL to github.com /owner/repo."""
    raw = (s or "").strip()
    if not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
        return False
    if raw.startswith("//"):
        raw = "https:" + raw
    if "://" not in raw:
        raw = "https://" + raw
    try:
        p = urlparse(raw)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.netloc or "").lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host != "github.com":
        return False
    parts = [x for x in (p.path or "").split("/") if x]
    if any(x == ".." for x in parts):
        return False
    if len(parts) < 2:
        return False
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo or len(repo) > 100:
        return False
    return bool(GITHUB_OWNER_RE.match(owner) and GITHUB_REPO_RE.match(repo))


def categorize_failures(failed: list[dict[str, str]]) -> dict[str, int]:
    """Categorize failed items by error reason for a summary.

    Returns a dict with keys: blocked, disabled, missing_transcript, other.
    - blocked: YouTube IP blocking (too many requests, cloud provider IP).
    - disabled: Transcripts disabled by video creator.
    - missing_transcript: No transcript available (e.g. TranscriptsDisabled, NoTranscriptFound).
    - other: All other failures.
    """
    summary: dict[str, int] = {
        "blocked": 0,
        "disabled": 0,
        "missing_transcript": 0,
        "other": 0,
    }
    for item in failed:
        err = (item.get("error") or "").lower()
        if "transcripts disabled" in err:
            summary["disabled"] += 1
        elif any(
            x in err for x in ("blocking", "blocked", "cloud provider", "ip block")
        ):
            summary["blocked"] += 1
        elif any(
            x in err
            for x in (
                "could not retrieve",
                "no transcript",
                "transcriptsdisabled",
                "notranscriptfound",
            )
        ):
            summary["missing_transcript"] += 1
        else:
            summary["other"] += 1
    return summary


def is_web_docs_url(s: str) -> bool:
    """Return True for any http(s) URL that parses cleanly with a host.

    Callers should use this only as a fallback after ``is_github_url`` and YouTube
    detection — ``detect_source_format`` enforces that precedence. Always rejects
    obvious non-URLs (null bytes, control chars, missing scheme/host).
    """
    raw = (s or "").strip()
    if not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
        return False
    if "://" not in raw:
        return False
    try:
        p = urlparse(raw)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False
    return True


def detect_source_format(
    path_or_url: str,
) -> (
    Literal[
        "youtube",
        "youtube_playlist",
        "pdf",
        "discord",
        "plaintext",
        "github",
        "web",
        "directory",
    ]
    | None
):
    """Detect supported source format from a path or URL.

    Returns one of ``"github"``, ``"youtube_playlist"``, ``"youtube"``, ``"pdf"``,
    ``"discord"``, ``"plaintext"``, ``"web"``, ``"directory"``, or ``None`` if
    unsupported. A watch URL with both ``v=`` and ``list=`` is treated as a single
    video; only dedicated playlist URLs index the full playlist. Any http(s) URL
    that isn't a GitHub repo or YouTube link falls through to ``"web"``.
    """
    s = (path_or_url or "").strip()
    if not s:
        return None
    if is_github_url(s):
        return "github"
    has_playlist = extract_playlist_id(s) is not None
    has_video = extract_video_id(s) is not None
    if has_video and has_playlist:
        return "youtube"
    if has_playlist:
        return "youtube_playlist"
    if has_video:
        return "youtube"
    if is_web_docs_url(s):
        return "web"
    base = resolve_user_content_path(s)
    if base.exists():
        if base.is_dir():
            return "directory"
        file_fmt = detect_file_format(base)
        if file_fmt:
            return file_fmt
        return None
    return None


def detect_file_format(path: Path) -> Literal["pdf", "discord", "plaintext"] | None:
    """Detect supported file format: PDF, DiscordChatExporter TXT, or plain TXT.

    Returns "pdf", "discord", "plaintext", or None if unsupported.
    For .txt: Discord (Guild:/Channel: header) takes precedence; else plaintext.
    """
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        # DiscordChatExporter has header with Guild: and Channel:
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:2048]
            lines = head.split("\n")[:30]
            has_guild = any(line.strip().startswith("Guild:") for line in lines)
            has_channel = any(line.strip().startswith("Channel:") for line in lines)
            if has_guild and has_channel:
                return "discord"
        except OSError:
            pass
        return "plaintext"
    return None
