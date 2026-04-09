#!/usr/bin/env python3
"""Index all Discord exports in data/discord-channels/ into Chroma.

Scans the base dir and each channel subfolder (e.g. alicia1200/) for .txt files
(DiscordChatExporter format). Run manually or via cron.

Usage:
  uv run python scripts/index_discord_channels.py
  uv run python scripts/index_discord_channels.py --base-dir /path/to/discord-exports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinrag.config import DEFAULT_PERSIST_DIR, get_collection_name  # noqa: E402
from pinrag.core import add_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index Discord channel exports from data/discord-channels/*/ into Chroma."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/discord-channels"),
        help="Base directory containing channel subfolders (default: data/discord-channels)",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="Chroma persistence directory",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: from config)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag for all indexed documents in this run",
    )
    args = parser.parse_args()

    base = args.base_dir.expanduser().resolve()
    if not base.exists():
        print(f"Base directory does not exist: {base}", file=sys.stderr)
        return 1
    if not base.is_dir():
        print(f"Base path is not a directory: {base}", file=sys.stderr)
        return 1

    collection = (args.collection or "").strip() or get_collection_name()

    # Index base dir first (files at root), then each channel subdirectory
    dirs_to_scan = [base] + sorted([d for d in base.iterdir() if d.is_dir()])
    # Dedupe: if base has no subdirs, we only have [base]
    dirs_to_scan = list(dict.fromkeys(dirs_to_scan))

    total_indexed = 0
    total_failed = 0

    for channel_dir in dirs_to_scan:
        result = add_file(
            path=str(channel_dir),
            persist_dir=args.persist_dir,
            collection=collection,
            tag=args.tag,
        )
        n_ok = result["total_indexed"]
        n_fail = result["total_failed"]
        total_indexed += n_ok
        total_failed += n_fail

        if n_ok:
            label = channel_dir.name if channel_dir != base else "(root)"
            for item in result["indexed"]:
                doc_id = item.get("document_id", item.get("source_path", "?"))
                chunks = item.get("total_chunks", "?")
                print(f"  {label}: {doc_id} ({chunks} chunks)")
        if n_fail:
            for item in result["failed"]:
                print(f"  {channel_dir.name} FAILED: {item.get('path', '?')} - {item.get('error', '?')}", file=sys.stderr)

    print(f"\nIndexed {total_indexed} file(s), {total_failed} failed.")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
