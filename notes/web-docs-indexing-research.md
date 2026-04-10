# Web Docs Indexing — Research & Design Proposal

**Status:** Decisions locked, implementation plan in §11.
**Date:** 2026-04-10 (decisions locked same day)
**Goal:** Let users point PinRAG at a documentation site (e.g. `https://docs.langchain.com/`, `https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html`, `https://picocomputer.github.io/`, `https://docs.crewai.com/`) and have PinRAG discover, fetch, clean, and index the relevant pages as a new `document_type="web"`.

## Locked decisions (2026-04-10)

1. **Scraper:** pure-Python builtin only. No external MCP scraper server, no Firecrawl/Crawl4ai backends, no Playwright. If a site is JS-only and produces empty extraction, we log and skip the page.
2. **Discovery:** both `llms.txt` / `llms-full.txt` fast-path **and** `sitemap.xml` (+ `robots.txt` sitemap hints) **and** BFS fallback. Try in that order.
3. **Scope boundary:** same host, same path prefix, **no subdomains**. No opt-out in v1.
4. **Backend pluggability:** **not** in v1. `web_loader.py` is a direct implementation, not a protocol with swappable backends. We add that abstraction only if/when a second backend is ever introduced.
5. **Extractor smoke targets:** the four URLs at the top of this doc. These are what we'll manually run against during development to sanity-check `trafilatura` output before committing to it as the default.

---

## 1. Why this is a distinct source type

We already have `pdf`, `youtube`, `discord`, `plaintext`, `github`. A URL to a doc site is none of these:

- **Not GitHub**: even when docs are generated from a GitHub repo, the rendered HTML is almost always richer (processed MDX, cross-page nav, search index, auto-generated API refs). Indexing the repo loses the navigation context and breaks in-page anchors. Many great doc sites (Raspberry Pi, Pico Computer, Anthropic, CrewAI hosted docs) don't even expose a 1:1 source repo.
- **Not plaintext**: HTML has to be cleaned (nav, footer, cookie banners, sidebars stripped; code blocks preserved; tables flattened sanely).
- **Unbounded surface**: unlike a PDF (one file) or a YouTube video (one transcript), a doc site is an N-page graph where *which* pages belong to "the docs" is itself a discovery problem.

So this is a new `document_type="web"` with its own loader + indexer, parallel to `github_loader.py`/`github_indexer.py`.

---

## 2. The two hard problems

### 2.1 Discovery — *which* pages to fetch

Given a single seed URL, we need to decide the page set. Options, in rough order of preference:

1. **`sitemap.xml` (and `sitemap_index.xml`)** — the single best signal. Almost every modern docs site (Docusaurus, MkDocs Material, Mintlify, VitePress, Nextra, Sphinx with `sphinx-sitemap`, Hugo, Docsaurus-like Mintlify hosts) publishes one. `https://docs.langchain.com/sitemap.xml` and `https://docs.crewai.com/sitemap.xml` both exist. Parse, filter to URLs under the seed prefix, done. This should be the **primary** discovery strategy.
2. **`llms.txt` / `llms-full.txt`** ([llmstxt.org](https://llmstxt.org/)) — new convention adopted by Anthropic, Mintlify, LangChain, and others specifically to feed LLMs. `llms-full.txt` is often a single pre-assembled markdown blob of the whole site. If present, we can **skip crawling entirely** and just ingest that file. **This should be checked first** before sitemap. Huge win for sites that support it (LangChain does — see `https://docs.langchain.com/llms.txt`).
3. **BFS crawl from seed, scoped to path prefix** — fallback when no sitemap and no `llms.txt`. Start at seed, parse HTML, collect `<a href>` links that stay under the seed's path prefix (same host, path starts with seed path minus the filename). Respect a depth limit and a max-pages cap. This is how we'd handle `https://picocomputer.github.io/` if it has no sitemap.
4. **`robots.txt`** — always parse to (a) discover additional sitemap URLs (`Sitemap:` directive) and (b) respect `Disallow`. Non-negotiable for being a good citizen.

**Discovery heuristics we'll likely need:**
- Scope rule: stay within the host *and* under the seed URL's directory prefix. Example: seed `https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html` should NOT pull `https://www.raspberrypi.com/products/...`. Default prefix = directory portion of the seed (`/documentation/microcontrollers/`).
- Drop obvious noise: URLs ending in `.zip`, `.pdf`, `.png`, etc.; fragments (`#section`) collapsed to the base URL; query strings normalized.
- Hard caps to prevent runaway crawls: `max_pages` (default ~200), `max_depth` (default 5), total byte budget.

### 2.2 Content extraction — *what* part of a page to keep

Raw `<html>` is unusable: nav bars, sidebars, "Edit this page" links, cookie banners, table of contents widgets, footer, analytics snippets. We need the *article body* and its structure (headings, code blocks, tables, lists) as clean markdown.

**Library options (Python):**

| Library | What it gives us | Notes |
| --- | --- | --- |
| **`trafilatura`** | Main-content extraction + markdown output | Best-in-class for news/blog; decent for docs. Fast, pure Python. Handles boilerplate removal well. |
| **`readability-lxml`** | Mozilla Readability port | Mature but tuned for articles, weaker on doc sites with heavy sidebars. Considered and rejected. |
| **`beautifulsoup4`** | Link extraction + sitemap XML parsing | Used as support lib for URL discovery, not primary extraction. |
| **`markdownify`** | HTML→Markdown fallback | Used only if `trafilatura` returns empty on a page where we can still locate a `<main>` / `<article>` element. |

**Chosen default stack:** `httpx` (async) + `beautifulsoup4` (link + sitemap parsing) + `trafilatura` (main extractor) + `markdownify` (fallback on trafilatura misses). All pure-Python, lightweight, no native deps beyond what chroma already pulls in.

**JavaScript-rendered sites:** Out of scope. If `trafilatura` returns empty text from a page, we log a warning (`"empty extraction, likely JS-rendered"`) and skip it, counting it as a failed page. No Playwright, no `--render-js` flag in v1.

---

## 3. Why not MCP-as-client for scraping

The user asked "smart way (hmmm maybe with mcp?)". We considered and rejected making PinRAG open an MCP client session to an external scraper server (Firecrawl MCP, Exa MCP, etc.) during indexing:

- **Nested MCP stacks.** PinRAG is itself an MCP server for Cursor/VS Code. Having PinRAG-the-server act as an MCP client to *another* server while serving a request is operationally ugly and introduces a hard runtime dependency on a second server being installed and running.
- **Packaging regression.** Users would need to install and configure a second MCP server just to index docs. Breaks the "install pinrag, done" story.
- **Cost / keys.** The good hosted scrapers (Firecrawl, Exa) are paid. Making them default would contradict PinRAG's free-first positioning (local embeddings, free-tier LLMs).
- **No intelligence needed.** The "smart" part of scraping doc sites is mechanical: parse sitemap, extract main content, convert to markdown. `trafilatura` + sitemap parsing covers the target sites. No LLM required in the extraction loop.

**Conclusion:** built-in pure-Python scraper, no swappable backend.

---

## 4. How this plugs into existing PinRAG architecture

The GitHub indexer is the closest analogue — it's also a URL-driven, multi-file source — so we should mirror its structure.

### 4.1 New files

```
src/pinrag/indexing/
    web_loader.py          # discover + fetch + extract → list[Document]
    web_indexer.py         # chunk + embed + upsert (mirrors github_indexer.py)
```

### 4.2 Format detection (`src/pinrag/core/format_detection.py`)

Add `"web"` to the `detect_source_format` return literal. New `is_web_docs_url(s)` — any `http(s)://` URL that isn't already classified as `github` / `youtube` / `youtube_playlist`. The ordering in `detect_source_format` matters: GitHub and YouTube must still win for URLs they own; `web` is the catch-all for anything else.

### 4.3 `operations.add_file` / `add_files`

New branch `if fmt == "web":` that calls `index_web(...)`, parallel to the existing `if fmt == "github":` branch (operations.py:328). Returns a result dict with:

```python
{
    "path": url,
    "format": "web",
    "site": "docs.langchain.com",      # netloc
    "root_url": "https://docs.langchain.com/",
    "pages_indexed": 47,
    "pages_failed": 2,
    "total_chunks": 812,
    "discovery": "sitemap" | "llms_full" | "crawl",
}
```

### 4.4 Metadata per chunk

Reuse existing keys where possible so `query` / `list_documents` / `set_document_tag` / `remove_document` all work without special-casing:

- `document_type = "web"`
- `document_id = "<host>/<path_prefix>"` — e.g. `docs.langchain.com/` — this is the logical *doc set*, so removing it nukes all pages from that site/section. Mirrors how GitHub uses `owner/repo`.
- `doc_id` — per-page UUID (if we go parent/child) or per-page stable hash.
- `source_url` — full URL of the page the chunk came from. **This is what shows up in citations.**
- `doc_title` — `<title>` or first `<h1>` of the page, used by `list_documents`' title fallback logic in `operations.py:68`.
- `section_heading` — the nearest H1/H2 the chunk lives under (if we do structure-aware chunking, which we already support).
- `upload_timestamp`, `tag`, `doc_bytes`, `doc_total_chunks` — same as other types.

### 4.5 Query / list / remove integration

- `query(document_type="web")` — add `"web"` to the allowed filter values in `operations.query` and in the MCP tool docstring at `mcp/server.py:158`.
- `list_documents` — already groups by `document_id`, so a site would show up as one row like `docs.langchain.com/ — 47 pages, 812 chunks`. Just needs a title-fallback entry in `_ensure_list_document_title` (operations.py:68) for `document_type == "web"`.
- `remove_document` — works as-is because it deletes by `document_id`. User removes "docs.langchain.com/" and every page chunk goes.
- `set_document_tag` — same, works as-is.

### 4.6 Chunking

Re-use existing `chunk_documents` pipeline. HTML→markdown output is structurally very similar to the markdown files we already chunk from GitHub repos, so `get_structure_aware_chunking()` + parent/child should just work. Per-page Documents flow in, chunks flow out, same as `github_indexer._index_github_flat` / `_index_github_parent_child`.

### 4.7 Config (new env vars, all with sensible defaults)

```
PINRAG_WEB_MAX_PAGES=200
PINRAG_WEB_MAX_DEPTH=5
PINRAG_WEB_MAX_PAGE_BYTES=1048576        # 1 MiB per page
PINRAG_WEB_REQUEST_TIMEOUT=20
PINRAG_WEB_CONCURRENCY=4                 # parallel fetches
PINRAG_WEB_USER_AGENT="pinrag/<version> (+https://github.com/ndjordjevic/pinrag)"
PINRAG_WEB_RESPECT_ROBOTS=1              # default on; opt-out for whitelisted use
PINRAG_WEB_RATE_LIMIT_PER_HOST=2         # requests/sec per host
PINRAG_WEB_PREFER_LLMS_TXT=1             # auto-detect llms-full.txt first
```

Pattern mirrors `get_github_max_file_bytes()` / `get_github_token()` at `config.py:481`.

### 4.8 Dependencies (new)

- `httpx` (async HTTP, stdlib-friendly)
- `beautifulsoup4` (link extraction, sitemap)
- `trafilatura` (main content extraction)
- `markdownify` (HTML→MD fallback)

All pure-Python, small, no native deps. No optional extras for web in v1.

---

## 5. No pluggable backend in v1

Deliberately rejected. `web_loader.py` is a single concrete implementation around `httpx` + `trafilatura`. We do not introduce a `WebScrapeBackend` protocol, a `PINRAG_WEB_BACKEND` env var, or stub classes for Firecrawl/Crawl4ai. Adding that abstraction before we actually have two backends is premature — we'd be designing a seam we might never use, and it complicates the loader for no benefit. If we ever add a second backend, we refactor then.

---

## 6. UX / contract for the user

The user just passes a URL to whatever they use today:

```
pinrag add https://docs.langchain.com/
```

…or via MCP:

```
add_files({"paths": ["https://docs.langchain.com/"], "tag": "langchain-docs"})
```

PinRAG reports back:

```
Discovered 47 pages via sitemap.xml
Fetched 47, indexed 47 (812 chunks)
Registered as document_id=docs.langchain.com/
```

Subsequent queries:

```
pinrag query "how does LCEL handle streaming?" --type web --doc docs.langchain.com/
```

Citations show `source_url` so the user can click through to the live page — this is the key UX win over indexing the docs GitHub repo.

---

## 7. Risks / open questions

1. **Politeness vs. completeness.** Strict robots.txt + rate limiting can slow or block indexing of some doc sites. Do we allow users to override per-run with `--ignore-robots`? Default should be respect.
2. **Update / re-index semantics.** Docs change. Do we fingerprint each page (ETag / Last-Modified / content hash) and skip unchanged pages on re-index? Probably yes — this is a meaningful efficiency win but adds state. Could store the hash in chunk metadata.
3. **Partial failures.** If 3 of 200 pages 404, we index the 197 and report the 3 in `failed_files` (same pattern as `GitHubLoadResult.failed_files`).
4. **Deduplication within a site.** Many doc sites render the same content at `/foo/` and `/foo/index.html`. Normalize URLs (strip trailing slash differences, strip fragments) before dedup.
5. **Scope drift.** If seed is `https://docs.langchain.com/` and the sitemap includes `https://api.langchain.com/...`, we **drop** those URLs. Same host only, no subdomains, no opt-out in v1.
6. **Versioned docs.** `docs.langchain.com/v0.3/...` vs `/v0.2/...`. The user probably wants one version. We could default to whatever the seed URL points at and restrict to that version's path prefix.
7. **JS-rendered sites.** Declared out of scope for v1 per §2.2. Need a clear error message when `trafilatura` returns empty text from a JS-shell page: *"This page appears to require JavaScript. Install the `web-js` extra and pass `--render-js` to try again."*
8. **llms-full.txt chunking.** A 2 MB `llms-full.txt` blob, chunked as one giant plaintext, would lose per-page `source_url` citations — which is the whole UX advantage. Either (a) parse its section headers back into per-page Documents, or (b) fall back to sitemap crawl when citations matter. Worth deciding up front.
9. **Storage bloat.** A big docs site can be 500+ pages / thousands of chunks. Should `list_documents` hide per-page detail and show a single rollup entry? Current behavior (group by `document_id`) handles this if we set `document_id` to the site, not per page.

---

## 8. Suggested minimum viable v1 scope

1. Builtin backend only: `httpx` + `trafilatura` + sitemap parsing + same-prefix BFS fallback.
2. `llms-full.txt` fast-path when present (but split it back into per-page Documents so citations stay useful).
3. `document_type="web"` wired through `query`, `list_documents`, `remove_document`, `set_document_tag`.
4. Respect `robots.txt` and rate-limit per host, non-negotiable.
5. Env-var config for caps; no CLI flags yet beyond the URL itself and `--tag`.
6. Integration test against a tiny controlled HTML fixture (not the live internet) plus one marked `@pytest.mark.integration` test against a known-stable small site.

Explicit **non-goals** for v1:
- JS rendering (Playwright).
- Firecrawl / Crawl4ai backends and any backend pluggability.
- Incremental re-index (ETag/Last-Modified).
- Cross-subdomain crawling.
- Using an external MCP scraper server.

---

## 9. Locked answers (previously open questions)

- **Document granularity:** one `document_id` per site (e.g. `docs.langchain.com/`). Page identity lives in `source_url` per chunk. This keeps `list_documents` tidy and makes `remove_document docs.langchain.com/` nuke the whole set.
- **`llms-full.txt` handling:** parse section headers back into per-URL Documents so citations stay meaningful. If we can't recover per-URL structure from the file, fall back to sitemap crawling for that site instead of indexing it as one giant blob.
- **`max_pages`:** default 200. Clear log line when hit: `"stopped at max_pages=200, set PINRAG_WEB_MAX_PAGES to raise"`.
- **Backend pluggability:** none in v1 (see §5).

---

## 10. Extractor smoke targets

Before wiring the loader into `add_file`, run `trafilatura` against each of these by hand (a small throwaway script under `scripts/`, not a test) and visually inspect the markdown output:

1. `https://picocomputer.github.io/` — small static site, probably no sitemap; exercises BFS fallback.
2. `https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html#pico2` — big marketing-styled site, path-prefix scoping will matter a lot; `#pico2` fragment should be stripped.
3. `https://docs.langchain.com/` — Mintlify docs; has `llms.txt` *and* sitemap. Exercises the `llms-full.txt` fast-path and the per-URL split logic.
4. `https://docs.crewai.com/` — another Mintlify site; used as a second llms.txt / sitemap target to catch site-specific quirks.

Acceptance for the smoke check: on each URL the extractor produces markdown where (a) the main article body is present, (b) code blocks are preserved as fenced code, (c) navigation/sidebar/footer boilerplate is absent or negligible, (d) headings survive as `#`/`##`. If any target fails and can't be fixed with a small post-processing tweak, escalate back to this doc before committing.

---

## 11. Implementation plan

Sequenced so each step is independently testable and reviewable. No branch required yet — this is a plan, not a commit log.

### Step 1 — Dependencies

- Add to `pyproject.toml` core deps: `httpx>=0.28`, `beautifulsoup4>=4.12`, `trafilatura>=2.0`, `markdownify>=0.14`.
- `uv sync` and confirm no version conflicts with existing langchain pins.

### Step 2 — Extractor smoke script (throwaway)

- `scripts/smoke_web_extract.py`: takes a URL, fetches with `httpx`, runs `trafilatura.extract(..., output_format='markdown', include_links=True, include_tables=True)`, prints to stdout.
- Run against all four targets in §10. Capture observations in a scratch file under `scripts/` (don't commit).
- **Decision gate:** if `trafilatura` is clearly bad on 2+ targets, pause and re-evaluate (maybe add `markdownify` on `<main>` as the primary, not the fallback). Otherwise proceed.

### Step 3 — URL utilities (`web_url.py` or inside `web_loader.py`)

- `normalize_url(u)` — lowercase host, strip fragment, collapse trailing slash, drop tracking query params (`utm_*`, `ref`, `fbclid`).
- `same_scope(candidate, seed)` — returns True iff `candidate` host == seed host AND `candidate.path` starts with `seed.dir_prefix`. `dir_prefix` is the seed URL's path up to and including the last `/` (so seed `.../pico-series.html` → prefix `/documentation/microcontrollers/`).
- `is_noise_url(u)` — True for file extensions we don't want (`.zip`, `.pdf`, `.png`, `.tar.gz`, etc.).
- Unit tests: `tests/test_web_url.py` covering the four seed URLs and edge cases (fragments, trailing slash, subdomain drop, noise).

### Step 4 — Discovery module (inside `web_loader.py`)

Three functions, tried in order until one yields a usable URL list:

1. `discover_llms_full(seed, client) -> list[FetchedPage] | None`
   - Fetches `{scheme}://{host}/llms-full.txt` (then `/llms.txt` fallback).
   - If 200 and parseable, splits by section headers into per-URL `FetchedPage` objects with the URL lifted from the section heading where possible.
   - Returns pre-extracted content, so Step 5 for these pages is a no-op.
2. `discover_sitemap(seed, client) -> list[str] | None`
   - Fetches `{host}/sitemap.xml`, `{host}/sitemap_index.xml`, and any `Sitemap:` directives in `robots.txt`.
   - Parses with `beautifulsoup4` (XML mode) to pull `<loc>` URLs.
   - Recursively follows sitemap index files.
   - Filters via `same_scope` + `is_noise_url`.
   - Deduplicates via `normalize_url`.
3. `discover_bfs(seed, client, limits) -> list[str]`
   - Seeded at `seed`. Queue-based BFS, depth-limited.
   - At each page: fetch HTML, extract `<a href>` with `beautifulsoup4`, resolve relative → absolute, filter via `same_scope` + `is_noise_url`, enqueue new ones.
   - Stops at `max_pages` / `max_depth`.

Each returns a discovery tag (`"llms_full" | "sitemap" | "crawl"`) that ends up in the result dict.

### Step 5 — Fetch + extract (`fetch_and_extract`)

- Async, `httpx.AsyncClient`, `max_concurrency` semaphore from `PINRAG_WEB_CONCURRENCY`.
- Per-host rate limiter (`asyncio.Semaphore` + sleep). `PINRAG_WEB_RATE_LIMIT_PER_HOST`.
- `robots.txt` check: load once per host with `urllib.robotparser`, skip disallowed URLs.
- Per-page: `httpx.get` → respect `Content-Length` ≤ `PINRAG_WEB_MAX_PAGE_BYTES` → `trafilatura.extract(html, output_format='markdown', favor_precision=True)`.
- On empty trafilatura result: fallback path tries `BeautifulSoup(html).select_one('main, article, [role=main]')` → `markdownify.markdownify(str(el))`. If still empty, mark page as failed with reason `"empty extraction"`.
- Extract `<title>` and first `<h1>` separately (trafilatura gives both via `bare_extraction`) for `doc_title` metadata.

### Step 6 — Loader: wire discovery + extraction into Documents

- `load_web_docs_as_documents(seed_url, *, limits) -> WebLoadResult`
- Dataclass mirroring `GitHubLoadResult`:
  ```python
  @dataclass(frozen=True)
  class WebLoadResult:
      seed_url: str
      host: str
      path_prefix: str
      documents: list[Document]       # one per page
      discovery: str                  # "llms_full" | "sitemap" | "crawl"
      failed_pages: list[dict[str, str]]
  ```
- Each `Document` has `page_content` = markdown and metadata `{source_url, doc_title, section_heading?}` (rest filled by indexer).

### Step 7 — Indexer (`web_indexer.py`)

- Mirror `github_indexer.py` structure:
  - `index_web(seed_url, *, persist_directory, collection_name, embedding, tag) -> WebIndexResult`
  - `_index_web_flat` and `_index_web_parent_child` helpers.
- Set `document_type="web"`, `document_id=f"{host}{path_prefix}"` (e.g. `docs.langchain.com/`), upload timestamp, optional tag.
- Upsert semantics: before indexing, `store._collection.delete(where={"document_id": repo_id})` and `remove_parent_docs_for_document(...)` when parent/child enabled, matching github_indexer.py:113.
- Reuse existing `chunk_documents` with `respect_structure=get_structure_aware_chunking()` — markdown with headings will flow through cleanly.

### Step 8 — Format detection

- `src/pinrag/core/format_detection.py`:
  - Add `"web"` to `detect_source_format` return literal.
  - New `is_web_docs_url(s)` — any `http(s)://` URL that URL-parses cleanly and isn't already GitHub/YouTube.
  - Update `detect_source_format` to check `is_github_url` → YouTube → `is_web_docs_url` → file path, preserving current precedence.

### Step 9 — `operations.add_file` branch

- Add `if fmt == "web":` branch right after the GitHub branch (operations.py:328), pattern-matching the github block exactly:
  - `_emit_verbose(..., "phase=web_index_start ...")`
  - Call `index_web(...)`.
  - Emit `phase=web_index_done host=... pages=... chunks=...`.
  - Return result dict with keys: `path`, `format: "web"`, `site`, `root_url`, `pages_indexed`, `pages_failed`, `total_chunks`, `discovery`.
  - Exception branch mirrors github's, with `phase=web_index_error`.
- Update the `Unsupported format` error message in `add_file` to mention web URLs too.

### Step 10 — Query / list / remove wiring

- `operations.query`: accept `"web"` in the `document_type` filter validation path. No retriever changes — metadata filter is generic.
- `operations._ensure_list_document_title`: add the `web` case — fallback title = host + prefix.
- `operations._list_title_from_chunk_meta`: same.
- `remove_document` / `set_document_tag`: already `document_id`-based, should work without edits. Add a smoke test to confirm.
- `mcp/server.py`: update docstrings where the allowed `document_type` values are listed (lines 158, 582, 594) to include `"web"`.

### Step 11 — Config

- `src/pinrag/config.py`: add getters mirroring `get_github_max_file_bytes`:
  - `get_web_max_pages() -> int` (default 200)
  - `get_web_max_depth() -> int` (default 5)
  - `get_web_max_page_bytes() -> int` (default 1 MiB)
  - `get_web_request_timeout() -> float` (default 20)
  - `get_web_concurrency() -> int` (default 4)
  - `get_web_rate_limit_per_host() -> float` (default 2.0)
  - `get_web_user_agent() -> str` (default `f"pinrag/{version} (+https://github.com/ndjordjevic/pinrag)"`)
  - `get_web_respect_robots() -> bool` (default True)
  - `get_web_prefer_llms_txt() -> bool` (default True)
- Update `env_validation.py` to list the new vars.
- Update `notes/env-vars.example.md` with the new vars.

### Step 12 — Tests

- **Unit, no network:**
  - `tests/test_web_url.py` — normalization, scope, noise detection (see Step 3).
  - `tests/test_web_discovery.py` — feed fake sitemap XML and HTML blobs to the discovery functions with an httpx `MockTransport`. Cover: sitemap, sitemap index, llms-full split, BFS with a small fixture graph, scope-dropping of subdomain URLs.
  - `tests/test_web_loader.py` — `httpx.MockTransport` serving 3 tiny HTML pages; assert correct number of Documents, correct `source_url` metadata, correct host/prefix in result.
  - `tests/test_web_indexer.py` — end-to-end with mocked loader + real Chroma (same pattern as `test_github_indexer.py`), verifying upsert semantics.
  - `tests/test_format_detection.py` — extend to cover the new `"web"` detection.
- **Integration, network:** one `@pytest.mark.integration` test against `https://picocomputer.github.io/` (small, stable, no JS). Skipped in CI unless explicitly opted in, matching existing integration markers.

### Step 13 — CLI / MCP surface

- `pinrag add <url>` already works via `add_file` once format detection returns `"web"` — no CLI changes required.
- MCP `add_files` tool likewise picks it up automatically.
- Update the help text in `mcp/server.py` for `add_files`, `query`, `list_documents_tool` to mention web docs.
- Update `README.md` and `LAUNCHGUIDE.md` with a "Web docs" section showing the four example URLs.

### Step 14 — Documentation / release

- Update `notes/implementation-checklist.md` with a new phase entry for web docs.
- Bump `pyproject.toml` version (minor bump, likely 0.10.0 since this adds a source type).
- Update `CHANGELOG` / release notes if the project keeps one.
- Cross-link this research doc from the checklist.

### Rough sizing

- Steps 1–6 (loader + discovery + extraction): biggest unknowns, ~60% of effort. Smoke script (Step 2) is the go/no-go for trafilatura.
- Step 7 (indexer): mechanical, ~15% — it's a copy/adapt of `github_indexer.py`.
- Steps 8–10 (wiring): ~10%, mostly one-line additions across known call sites.
- Steps 11–14 (config, tests, docs): ~15%.

### Non-obvious risks we're accepting

- **trafilatura on Mintlify sites.** `docs.langchain.com` and `docs.crewai.com` are both Mintlify. If Mintlify serves JS-shell HTML, extraction will fail site-wide and we'll fall back to llms-full.txt for those. That's actually fine — both publish `llms.txt`. But it means the BFS path is mostly exercised by `picocomputer.github.io` in practice.
- **llms-full.txt per-URL split.** The llmstxt.org spec doesn't mandate per-URL anchors inside `llms-full.txt`. If a site concatenates without source attribution, we lose per-page citations. Behavior in that case: ingest as a single Document with `source_url=<llms-full.txt url>`. Document this caveat in the README so users aren't surprised.
- **Raspberry Pi site.** `raspberrypi.com` is a marketing site, not a docs framework. The path-prefix scoping (`/documentation/microcontrollers/`) is the main defense against pulling the entire marketing site. Smoke test this carefully in Step 2.
