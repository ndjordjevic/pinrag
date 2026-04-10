"""Smoke-test the trafilatura web extractor against real doc sites.

Run:
    uv run python scripts/smoke_web_extract.py <url>
    uv run python scripts/smoke_web_extract.py --all

Prints a short summary (status, bytes, title, extracted markdown length, first lines)
plus the first ~80 lines of the extracted markdown so a human can eyeball the quality
of trafilatura output before we commit to it as the default extractor.

Targets for `--all` come from notes/web-docs-indexing-research.md §10.
"""

from __future__ import annotations

import argparse
import sys

import httpx
import trafilatura

TARGETS = [
    "https://picocomputer.github.io/",
    "https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html",
    "https://docs.langchain.com/",
    "https://docs.crewai.com/",
]


def smoke_one(url: str, *, preview_lines: int = 80) -> int:
    print(f"\n=== {url} ===")
    try:
        resp = httpx.get(
            url,
            follow_redirects=True,
            timeout=20,
            headers={"User-Agent": "pinrag-smoke/0.1"},
        )
    except Exception as e:
        print(f"  FETCH ERROR: {e}")
        return 1
    print(f"  status={resp.status_code} bytes={len(resp.content)} final_url={resp.url}")
    if resp.status_code != 200:
        return 1

    html = resp.text
    md = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        include_formatting=True,
        favor_precision=True,
    )
    if not md:
        print("  EXTRACTED: <empty> (likely JS-rendered)")
        return 2

    bare = trafilatura.bare_extraction(html, output_format="markdown")
    title = getattr(bare, "title", None) if bare else None

    lines = md.splitlines()
    print(f"  extracted_chars={len(md)} extracted_lines={len(lines)} title={title!r}")
    print("  --- preview ---")
    for line in lines[:preview_lines]:
        print(f"  | {line}")
    if len(lines) > preview_lines:
        print(f"  ... ({len(lines) - preview_lines} more lines)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="URL to extract. Omit with --all.")
    parser.add_argument("--all", action="store_true", help="Run against all default targets.")
    parser.add_argument("--preview", type=int, default=80, help="Lines of markdown to print per target.")
    args = parser.parse_args()

    if args.all:
        urls = TARGETS
    elif args.url:
        urls = [args.url]
    else:
        parser.error("Pass a URL or --all")
        return 2

    worst = 0
    for url in urls:
        rc = smoke_one(url, preview_lines=args.preview)
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    sys.exit(main())
