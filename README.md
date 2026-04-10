<!-- mcp-name: io.github.ndjordjevic/pinrag -->

<!-- Raster PNG: many directories (e.g. MCP Market) strip HTML or block SVG in READMEs; plain Markdown image survives more often. -->
![PinRAG logo](https://raw.githubusercontent.com/ndjordjevic/pinrag/main/docs/pinrag-icon.png)

# PinRAG

[![PyPI](https://img.shields.io/github/v/release/ndjordjevic/pinrag?logo=pypi&logoColor=white&label=PyPI&sort=semver)](https://pypi.org/project/pinrag/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![io.github.ndjordjevic/pinrag on MCP Marketplace](https://mcp-marketplace.io/api/badge?slug=io-github-ndjordjevic-pinrag)](https://mcp-marketplace.io/server/io-github-ndjordjevic-pinrag) [![pinrag MCP server](https://glama.ai/mcp/servers/ndjordjevic/pinrag/badges/score.svg)](https://glama.ai/mcp/servers/ndjordjevic/pinrag)

## Overview

PinRAG is for **when you want to learn about something and your materials are scattered**—PDFs and ebooks, GitHub repos, YouTube videos, Discord discussions, and plain notes. You index those materials into **one shared RAG index**, then **ask questions from Cursor, VS Code (GitHub Copilot), or any MCP-capable assistant** and get answers **with citations** pointing back to pages, timestamps, files, or threads.

Under the hood it is **Retrieval-Augmented Generation** built with **LangChain** and exposed as an **MCP (Model Context Protocol) server**: add documents from the editor, query with natural language, list or remove what you indexed. Supported inputs include PDFs, local text files and directories, Discord exports, YouTube (transcript from URL, playlist, or ID), and GitHub repo URLs. For YouTube you can optionally add **vision** so on-screen code, diagrams, and UI text are merged with the transcript in the same chunks—see [YouTube vision enrichment](#youtube-vision-enrichment-optional).

## Features

- **Multi-format indexing** — PDF (.pdf), local files or directories, plain text (.txt), Discord export (.txt), YouTube (video or playlist URL, or video ID), GitHub repo (URL), web documentation sites (URL)
- **Optional YouTube vision** — Off by default. When enabled, runs a vision model (OpenAI, Anthropic, or OpenRouter native video) and merges structured on-screen context with the transcript so RAG chunks carry searchable code names, labels, and diagrams—not speech alone. OpenRouter mode avoids local ffmpeg/video download; openai/anthropic use scene keyframes and require `pinrag[vision]` + **ffmpeg** (see [YouTube vision enrichment](#youtube-vision-enrichment-optional))
- **RAG with citations** — Answers cite source context: PDF page, YouTube timestamp, document name for plain text and Discord, chunk index for GitHub repos, source URL for web documentation
- **Document tags** — Tag documents at index time (e.g. `AMIGA`, `PI_PICO`) for filtered search
- **Metadata filtering** — `query_tool` supports `document_id`, `tag`, `document_type`, PDF `page_min`/`page_max`, and `response_style` (thorough or concise)
- **MCP tools** — `add_document_tool`, `query_tool`, `list_documents_tool`, `remove_document_tool`, `set_document_tag_tool`, `list_collections_tool`; optional `collection` on tools overrides `PINRAG_COLLECTION_NAME` for that call
- **MCP resources** — `pinrag://documents` (indexed documents) and `pinrag://server-config` (env vars and config); click in Cursor’s MCP panel to view
- **MCP prompt** — `use_pinrag` (parameter: request) for querying, indexing, listing, or removing documents
- **Configurable LLM** — OpenRouter (default, free `openrouter/free` router), OpenAI, Anthropic, or [Cerebras Inference](https://inference-docs.cerebras.ai/introduction) (OpenAI-compatible API); set via `PINRAG_LLM_PROVIDER` and `PINRAG_LLM_MODEL` in MCP `env` or your shell
- **Local embeddings** — Nomic (`PINRAG_EMBEDDING_MODEL`, default `nomic-embed-text-v1.5`); no API key; first run downloads model weights (~270 MB, cached)
- **Retrieval & chunking options** — Structure-aware chunking (on by default); optional FlashRank re-ranking, multi-query expansion, and parent-child chunks for PDFs (see Configuration)
- **Observability** — MCP tool notifications (`ctx.log`) plus optional [LangSmith](https://smith.langchain.com) tracing
- **Built with** — LangChain, Chroma; optional OpenRouter, OpenAI, Anthropic, FlashRank

## Installation

Add PinRAG as an MCP server in your editor. Install [**uv**](https://docs.astral.sh/uv/) and ensure `uvx` is on your `PATH`—that runs PinRAG from PyPI without a prior `pip install`.

**Cursor:** add this under `mcpServers` in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pinrag": {
      "command": "uvx",
      "args": ["--refresh", "pinrag"],
      "env": {
        "OPENROUTER_API_KEY": "your-openrouter-api-key-here",
        "PINRAG_PERSIST_DIR": "/absolute/path/to/your/pinrag-data"
      }
    }
  }
}
```

**VS Code (GitHub Copilot):** run **MCP: Open User Configuration** from the Command Palette (or add `.vscode/mcp.json` in a workspace), then merge this shape—top-level key is `servers`:

```json
{
  "servers": {
    "pinrag": {
      "command": "uvx",
      "args": ["--refresh", "pinrag"],
      "env": {
        "OPENROUTER_API_KEY": "your-openrouter-api-key-here",
        "PINRAG_PERSIST_DIR": "/absolute/path/to/your/pinrag-data"
      }
    }
  }
}
```

## Quick Start

### HTTP server mode

For clients that speak MCP over HTTP (e.g. **[pinrag-cli](https://github.com/ndjordjevic/pinrag-cli)** with `--server`), run:

```bash
pinrag server [--host 127.0.0.1] [--port 8765]
```

This starts a **streamable-HTTP** MCP endpoint at `http://<host>:<port>/mcp`. The default `pinrag` stdio command for editors is unchanged; `pinrag server` is additive. Connect pinrag-cli with `--server http://127.0.0.1:8765/mcp`.

### Configure MCP server

Put API keys and any PinRAG settings in the MCP entry’s **`env` block**. The server does not load `.env` files when the editor launches it.

### Use in chat

| Action | Tool |
|--------|------|
| Index files, directories, or URLs | `add_document_tool` — required **`paths`**: list of local paths (PDFs, plain or DiscordChatExporter `.txt`, directories) or URLs (YouTube videos, playlist URLs, GitHub repos, web documentation sites; bare YouTube video IDs allowed). Optional **`tags`** (one per path). For GitHub URLs only: **`branch`**, **`include_patterns`**, **`exclude_patterns`**. |
| List indexed documents | `list_documents_tool` — returns **`documents`** (IDs), **`total_chunks`**, and optional **`tag`** filter. **`document_details`** may include `document_type`, tags, page / message / segment counts, titles, aggregated **`bytes`**, and **`upload_timestamp`** when present in metadata. |
| Query with filters | `query_tool` — required **`query`**. Optional **`document_id`**, **`tag`**, **`document_type`**, **`page_min`** / **`page_max`** (PDF ranges), **`response_style`** (`thorough` or `concise`; leave empty to use **`PINRAG_RESPONSE_STYLE`**). |
| Remove a document | `remove_document_tool` — required **`document_id`** (exact value from **`list_documents_tool`**). |
| View resources (read-only) | In the MCP panel, open **Resources** and choose **`pinrag://documents`** (indexed docs) or **`pinrag://server-config`** (effective config, including **`PINRAG_VERSION`**). |

Ask in chat: *"Add /path/to/amiga-book.pdf with tag AMIGA"*, *"Index https://youtu.be/xyz and ask what it says"*, *"Index https://github.com/owner/repo and ask about the codebase"*, or *"Index https://docs.langchain.com/ and summarize its memory APIs"*. The AI will invoke the tools for you. Citations show page numbers for PDFs, timestamps (e.g. `t. 1:23`) for YouTube, document names for plain text and Discord exports, chunk index labels for GitHub, and source URLs for web documentation.

### GitHub indexing

Index a repo with **`add_document_tool`** and a URL in **`paths`**, e.g. `https://github.com/owner/repo`, `https://github.com/owner/repo/tree/branch`, or `github.com/owner/repo` (scheme optional).

**GitHub-only options:** **`branch`**, **`include_patterns`** / **`exclude_patterns`** — defaults already favor common text and source files and skip bulky artifacts; use patterns when you need files outside that set. Files over **`PINRAG_GITHUB_MAX_FILE_BYTES`** (default 512 KiB) are skipped.

**Auth:** Set **`GITHUB_TOKEN`** in MCP **`env`** (or the shell) for **private** repos or fewer rate-limit hits on big indexes; small public runs often work without it. Use a classic or fine-grained PAT with repo read access; there is no OAuth in PinRAG.

### Web documentation indexing

Point **`add_document_tool`** at any documentation site URL, e.g. `https://docs.langchain.com/`, `https://docs.crewai.com/`, or `https://picocomputer.github.io/`. PinRAG discovers pages via (in order) **`llms.txt`** / **`llms-full.txt`** (Mintlify-style), **`sitemap.xml`** (including `robots.txt` `Sitemap:` hints and nested sitemap indexes), then a scoped BFS crawl from the seed URL.

**Scope:** exact host match (no subdomains) plus path prefix derived from the seed — e.g. `https://docs.example.com/guide/` only indexes pages under `/guide/`. Use the site root URL to capture the full docs tree.

**Extraction:** `text/markdown` responses (from llms.txt fast paths) pass through; HTML runs through `trafilatura` with a BeautifulSoup + `markdownify` fallback that scopes to `<main>` / `<article>` / `[role=main]`.

**Limits & politeness:** controlled by **`PINRAG_WEB_MAX_PAGES`** (default 200), **`PINRAG_WEB_MAX_DEPTH`** (5), **`PINRAG_WEB_MAX_PAGE_BYTES`** (1 MiB), **`PINRAG_WEB_CONCURRENCY`** (4), **`PINRAG_WEB_RATE_LIMIT_PER_HOST`** (2.0/sec), and **`PINRAG_WEB_RESPECT_ROBOTS`** (`true`). Some sites (e.g. Cloudflare-protected pages) may return 403 to pure-Python clients; that's a known limitation.

**Citations:** web chunks carry a `source_url` metadata field; answers cite per-page URLs, and the `document_id` is `<host><path_prefix>` so `remove_document_tool` / `set_document_tag_tool` operate on the whole site at once.

### YouTube indexing and IP blocking

Transcript-heavy indexing—especially from cloud or high-volume IPs—may return errors like *"YouTube is blocking requests from your IP"*. Point **`youtube-transcript-api`** at a proxy via MCP **`env`** (or your shell):

```env
PINRAG_YT_PROXY_HTTP_URL=http://user:pass@proxy.example.com:80
PINRAG_YT_PROXY_HTTPS_URL=http://user:pass@proxy.example.com:80
```

**`PINRAG_YT_PROXY_*`** affects transcript fetches only; yt-dlp steps (titles, playlists) do not use it. Residential or rotating proxies usually fare better than raw datacenter IPs.

When some paths fail (e.g. a few videos in a playlist), **`add_document_tool`** includes **`fail_summary`** with counts keyed by `blocked`, `disabled`, `missing_transcript`, and `other`.

### YouTube vision enrichment (optional)

Default indexing is **transcript-only**. Set **`PINRAG_YT_VISION_ENABLED=true`** to add vision captions for on-screen content, time-aligned with the transcript and chunked with metadata such as `has_visual`, `frame_count`, and `visual_source`.

**`PINRAG_YT_VISION_PROVIDER`:**

- **`openai`** (default) or **`anthropic`**: **`yt-dlp`** download → scene-based frames → **one multimodal call per frame**. Needs **`pinrag[vision]`**, **ffmpeg**/**ffprobe** on `PATH`, and **`OPENAI_API_KEY`** or **`ANTHROPIC_API_KEY`** (install the extra in the same env as `pinrag`, e.g. `uv sync --extra vision` or `pip install 'pinrag[vision]'`).
- **`openrouter`**: **one** OpenRouter request per video via **`video_url`** (default **`google/gemini-2.5-flash`**). **`OPENROUTER_API_KEY`** only—no download, ffmpeg, or **`pinrag[vision]`**; choose a video-capable model if you override **`PINRAG_YT_VISION_MODEL`**.

**Ops:** Re-index after changing vision settings. For **openai**/**anthropic**, tune cost and timeouts with **`PINRAG_YT_VISION_MAX_FRAMES`** and optional **`PINRAG_YT_VISION_IMAGE_DETAIL=high`** (clearer small text, more tokens). **MCP stdio:** yt-dlp progress goes to **stderr** so **stdout** stays JSON-clean. Downloading video can breach YouTube ToS or local rules—your call. **Docker:** build with **`BUILD_WITH_VISION=1`** for ffmpeg + **`pinrag[vision]`** (see `Dockerfile`).

### Tips

- **`pinrag` not found:** MCP inherits your login **PATH**. After `pipx` or `uv tool install`, restart the editor and confirm `which pinrag`.
- **`PINRAG_PERSIST_DIR`:** Use a stable **absolute** path in MCP **`env`** (e.g. `~/.pinrag/chroma_db`) so the vector store does not depend on the server process cwd.
- **FlashRank:** Install **`pinrag[rerank]`** in the same tool env (`pipx install 'pinrag[rerank]'` / `uv tool install 'pinrag[rerank]'`); tunables are in **Configuration**.
- **YouTube vision:** Follow [YouTube vision enrichment](#youtube-vision-enrichment-optional) for env and deps; **re-index** after changing vision settings.
- **`pinrag://server-config`:** MCP **Resources** → this URI for **`PINRAG_VERSION`**, effective LLM/embeddings/chunking, and API key **set / not set** status.

## Configuration

The **`pinrag://server-config`** MCP resource prints **`PINRAG_VERSION`** (package version, not an env var you set) and effective values for the variables below, plus which API keys are set. Use the table as the full env reference.

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| **LLM** | | |
| **Provider & model** | | |
| `PINRAG_LLM_PROVIDER` | `openrouter` | `openrouter`, `openai`, `anthropic`, or `cerebras` |
| `PINRAG_LLM_MODEL` | *(provider default)* | When unset: OpenRouter `openrouter/free`, OpenAI `gpt-4o-mini`, Anthropic `claude-haiku-4-5`, Cerebras `llama3.1-8b`. Override with any model id (e.g. OpenRouter `anthropic/claude-sonnet-4-6`, Cerebras `gpt-oss-120b`). |
| **OpenRouter** | | |
| `PINRAG_OPENROUTER_MODEL_FALLBACKS` | *(unset)* | Comma-separated fallback model slugs sent as OpenRouter’s `models` list. The gateway tries the next slug when the primary (`PINRAG_LLM_MODEL`) fails (rate limits, downtime, etc.). Use extra free models here to stay zero-cost. Legacy alias: `PINRAG_LLM_MODEL_FALLBACKS`. |
| `PINRAG_OPENROUTER_SORT` | *(unset)* | Optional `provider.sort` — `price`, `throughput`, or `latency`. When unset, OpenRouter uses its default provider selection. Prefer leaving this unset if you set `PINRAG_OPENROUTER_PROVIDER_ORDER` to pin a specific backend (avoids conflicting routing signals). |
| `PINRAG_OPENROUTER_PROVIDER_ORDER` | *(unset)* | Comma-separated provider names for `provider.order` (tried in sequence). Example: `Cerebras` with `PINRAG_LLM_MODEL=openai/gpt-oss-120b` to prefer [Cerebras-backed routing](https://openrouter.ai/docs/guides/routing/provider-selection). Use exact labels from the model’s provider list on OpenRouter. |
| `OPENROUTER_APP_URL` | `https://github.com/ndjordjevic/pinrag` | App attribution (`HTTP-Referer`). Override with your site URL (see [OpenRouter app attribution](https://openrouter.ai/docs/app-attribution)). PinRAG copies this into `OPENROUTER_HTTP_REFERER` for the OpenRouter Python SDK. |
| `OPENROUTER_APP_TITLE` | `PinRAG` | App title (`X-Title`). Override to label usage in the OpenRouter dashboard. PinRAG copies this into `OPENROUTER_X_OPEN_ROUTER_TITLE` for the SDK. |
| **API keys** | | |
| `OPENROUTER_API_KEY` | *(required when using OpenRouter for LLM, evaluators, or YouTube vision)* | Required when `PINRAG_LLM_PROVIDER=openrouter`, `PINRAG_EVALUATOR_PROVIDER=openrouter`, or YouTube vision with `PINRAG_YT_VISION_PROVIDER=openrouter`. |
| `OPENAI_API_KEY` | *(required for OpenAI LLM or OpenAI YouTube vision)* | Required when `PINRAG_LLM_PROVIDER=openai`, or when `PINRAG_YT_VISION_ENABLED=true` and `PINRAG_YT_VISION_PROVIDER=openai`. |
| `OPENAI_BASE_URL` | *(optional)* | Override OpenAI API base URL (e.g. `https://openrouter.ai/api/v1` with `OPENAI_API_KEY` set to your OpenRouter key for vision or other OpenAI-compatible calls). |
| `CEREBRAS_API_KEY` | *(required for Cerebras LLM)* | Required when `PINRAG_LLM_PROVIDER=cerebras`. Get a key from the [Cerebras cloud console](https://cloud.cerebras.ai). |
| `PINRAG_CEREBRAS_BASE_URL` | `https://api.cerebras.ai/v1` | Override the OpenAI-compatible base URL for Cerebras (e.g. dedicated inference endpoints). |
| `ANTHROPIC_API_KEY` | *(required for Anthropic LLM or Anthropic YouTube vision)* | Required when `PINRAG_LLM_PROVIDER=anthropic`, `PINRAG_EVALUATOR_PROVIDER=anthropic`, or YouTube vision with `PINRAG_YT_VISION_PROVIDER=anthropic`. |
| **Embeddings** | | |
| `PINRAG_EMBEDDING_MODEL` | `nomic-embed-text-v1.5` | Local Nomic model id (via `langchain-nomic`). First run downloads weights (~270 MB, cached). No API key. |
| **Storage & chunking** | | |
| `PINRAG_PERSIST_DIR` | `chroma_db` | Chroma vector store directory (default is relative to the server process cwd unless you set an absolute path; e.g. `~/.pinrag/chroma_db` for a fixed location) |
| `PINRAG_CHUNK_SIZE` | `1000` | Text chunk size (chars) |
| `PINRAG_CHUNK_OVERLAP` | `200` | Chunk overlap (chars) |
| `PINRAG_STRUCTURE_AWARE_CHUNKING` | `true` | Apply structure-aware chunking heuristics for code/table boundaries |
| `PINRAG_COLLECTION_NAME` | `pinrag` | Chroma collection name. Single shared collection by default. |
| `ANONYMIZED_TELEMETRY` | `False` via `setdefault` when unset | Chroma telemetry flag. PinRAG’s MCP logging setup calls `os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")` so empty/unset behaves like opt-out; set `true` in `env` if you want Chroma’s telemetry on. |
| **Retrieval** | | |
| `PINRAG_RETRIEVE_K` | `20` | Retrieval pool size when re-ranking is **off**. When re-ranking is **on**, `PINRAG_RERANK_RETRIEVE_K` falls back to this value if unset, then results are cut to `PINRAG_RERANK_TOP_N`. |
| **Parent-child retrieval** | | |
| `PINRAG_USE_PARENT_CHILD` | `false` | Set to `true` to embed small chunks and return larger parent chunks (supported for PDF, GitHub, YouTube, and Discord indexing—not plain `.txt`). Requires re-indexing. |
| `PINRAG_PARENT_CHUNK_SIZE` | `2000` | Parent chunk size (chars) when `PINRAG_USE_PARENT_CHILD=true`. |
| `PINRAG_CHILD_CHUNK_SIZE` | `800` | Child chunk size (chars) when `PINRAG_USE_PARENT_CHILD=true`. |
| **Re-ranking** | | |
| `PINRAG_USE_RERANK` | `false` | Set to `true` to enable FlashRank re-ranking: fetch more chunks, re-score locally, pass top N to the LLM. Requires `pip install pinrag[rerank]` and no API key. |
| `PINRAG_RERANK_RETRIEVE_K` | *(inherits `PINRAG_RETRIEVE_K`)* | Chunks to fetch before FlashRank when `PINRAG_USE_RERANK=true`. If unset, equals `PINRAG_RETRIEVE_K` (not a separate hard-coded 20). |
| `PINRAG_RERANK_TOP_N` | `10` | Chunks passed to the LLM after re-ranking when `PINRAG_USE_RERANK=true` (capped by the pre-rerank fetch size). |
| **Multi-query** | | |
| `PINRAG_USE_MULTI_QUERY` | `false` | Set to `true` to generate alternative phrasings of the user query via LLM, retrieve per variant, and merge (unique union). Improves recall for terse or ambiguous queries. |
| `PINRAG_MULTI_QUERY_COUNT` | `4` | Number of **alternative** queries to generate (default 4, max 10). The original query is still included in retrieval when merging. |
| **Response style** | | |
| `PINRAG_RESPONSE_STYLE` | `thorough` | RAG answer style: `thorough` (detailed) or `concise`. Used by evaluation target and as default when MCP `query` omits `response_style`. |
| **MCP notifications** | | |
| `PINRAG_VERBOSE_LOGGING` | `false` | Set `true` to emit detailed per-phase MCP notifications for tool/resource execution (format detection, transcript load, vision path/steps, chunk upserts). Default keeps concise start/ok/error lifecycle logs. |
| **GitHub indexing** | | |
| `GITHUB_TOKEN` | *(optional)* | Personal access token for GitHub API. Required for private repos; increases rate limits for public repos. |
| `PINRAG_GITHUB_MAX_FILE_BYTES` | `524288` (512 KB) | Skip files larger than this when indexing GitHub repos. |
| `PINRAG_GITHUB_DEFAULT_BRANCH` | `main` | Default branch when not specified in the GitHub URL. |
| **Plain text indexing** | | |
| `PINRAG_PLAINTEXT_MAX_FILE_BYTES` | `524288` (512 KB) | Skip plain .txt files larger than this when indexing. |
| **Web docs indexing** | | |
| `PINRAG_WEB_MAX_PAGES` | `200` | Maximum pages fetched per web indexing run. |
| `PINRAG_WEB_MAX_DEPTH` | `5` | Maximum BFS crawl depth from the seed URL (ignored for llms.txt / sitemap fast paths). |
| `PINRAG_WEB_MAX_PAGE_BYTES` | `1048576` (1 MiB) | Skip pages whose response body exceeds this size. |
| `PINRAG_WEB_REQUEST_TIMEOUT` | `20` | Per-request HTTP timeout in seconds (connect + read). |
| `PINRAG_WEB_CONCURRENCY` | `4` | Maximum concurrent fetches per host. |
| `PINRAG_WEB_RATE_LIMIT_PER_HOST` | `2.0` | Token-bucket refill rate (requests / second) per host. |
| `PINRAG_WEB_USER_AGENT` | `PinRAGBot/<version> (+https://github.com/ndjordjevic/pinrag)` | HTTP `User-Agent` header for web indexing. Some sites block generic bots; override if needed. |
| `PINRAG_WEB_RESPECT_ROBOTS` | `true` | `true` / `false` — honor `robots.txt` disallow rules when crawling. |
| `PINRAG_WEB_PREFER_LLMS_TXT` | `true` | Try `llms.txt` / `llms-full.txt` before sitemap / BFS. Disable to force sitemap or crawl discovery. |
| **YouTube transcript proxy** | | |
| `PINRAG_YT_PROXY_HTTP_URL` | *(none)* | HTTP proxy URL for transcript fetches (e.g. `http://user:pass@proxy:80`). Use when YouTube blocks your IP. |
| `PINRAG_YT_PROXY_HTTPS_URL` | *(none)* | HTTPS proxy URL for transcript fetches. Same as HTTP when using a generic proxy. |
| **YouTube vision (optional)** | | |
| `PINRAG_YT_VISION_ENABLED` | `false` | `true` / `1` / `yes` / `on` enables on-screen enrichment for YouTube. **`openai`** / **`anthropic`**: needs **`pinrag[vision]`**, **ffmpeg** on `PATH`, and the matching API key. **`openrouter`**: needs **`OPENROUTER_API_KEY`** only (native `video_url` path; no local download). |
| `PINRAG_YT_VISION_PROVIDER` | `openai` | `openai`, `anthropic`, or **`openrouter`**. Independent of **`PINRAG_LLM_PROVIDER`** (RAG LLM and vision can use different providers). Legacy alias: **`PINRAG_VISION_PROVIDER`**. |
| `PINRAG_YT_VISION_MODEL` | *(per provider)* | If unset: OpenAI **`gpt-4o-mini`**, Anthropic **`claude-sonnet-4-6`**, OpenRouter **`google/gemini-2.5-flash`**. Use a **vision-capable** id. Legacy alias: **`PINRAG_VISION_MODEL`**. |
| `PINRAG_YT_VISION_MAX_FRAMES` | `8` | **Download + frame path only** (`openai` / `anthropic`): cap analyzed keyframes after scene detect. **Ignored** for **`openrouter`** (single video request). |
| `PINRAG_YT_VISION_MIN_SCENE_SCORE` | `27.0` | **Download + frame path only:** PySceneDetect `AdaptiveDetector` threshold (higher → fewer cuts). **Ignored** for **`openrouter`**. |
| `PINRAG_YT_VISION_IMAGE_DETAIL` | `low` | **OpenAI frame path only:** `low`, `high`, or `auto` for `image_url.detail`. **Ignored** for **`anthropic`** (full frames) and **`openrouter`** (`video_url`). |
| **LangSmith (optional)** | | |
| `LANGSMITH_TRACING` | *(off)* | Set `true` to send traces to [LangSmith](https://smith.langchain.com). Requires `LANGSMITH_API_KEY`. |
| `LANGSMITH_API_KEY` | *(none)* | API key from LangSmith **Settings → API keys**. |
| `LANGSMITH_PROJECT` | *(LangChain default)* | Project name for traces (e.g. `pinrag`). |
| `LANGSMITH_ENDPOINT` | US API (implicit) | **EU workspaces:** set `https://eu.api.smith.langchain.com` so traces land in your EU project. If your account uses `eu.smith.langchain.com` in the browser, you need this. US-region workspaces can omit it (default API host). |
| **Evaluators (LLM-as-judge)** | | |
| `PINRAG_EVALUATOR_PROVIDER` | `openai` | `openai`, `anthropic`, or `openrouter` — which LLM runs LLM-as-judge graders. Used only during evaluation runs (LangSmith experiments). |
| `PINRAG_EVALUATOR_MODEL` | *(provider default)* | Model for **correctness** grading (e.g. `gpt-4o`, `claude-sonnet-4-6`, `openrouter/free` when evaluator provider is OpenRouter). With OpenRouter, the default free router may rotate models; graders use strict JSON schema—set this to a **specific** free slug from [openrouter.ai/models](https://openrouter.ai/models) if you need stable structured output. OpenRouter routing env vars below also apply to graders when `PINRAG_EVALUATOR_PROVIDER=openrouter`. |
| `PINRAG_EVALUATOR_MODEL_CONTEXT` | *(provider default)* | Model for **groundedness** grading (large retrieved context; e.g. `gpt-4o-mini`, `claude-haiku-4-5`, `openrouter/free` when evaluator provider is OpenRouter). Same OpenRouter note as `PINRAG_EVALUATOR_MODEL`. When the evaluator provider is OpenRouter, `PINRAG_OPENROUTER_MODEL_FALLBACKS`, `PINRAG_OPENROUTER_SORT`, and `PINRAG_OPENROUTER_PROVIDER_ORDER` apply to the grader client. |

> **Re-indexing when changing embedding model:** Changing `PINRAG_EMBEDDING_MODEL` requires re-indexing; vector dimensions must match the model used at index time (including indexes created under an older default embedding).
>
> **Re-indexing when enabling parent-child:** Setting `PINRAG_USE_PARENT_CHILD=true` requires re-indexing; the new structure (child chunks in Chroma, parent chunks in docstore) is created only during indexing for supported document types (not plain `.txt`).
>
> **Re-indexing when toggling YouTube vision:** Turning `PINRAG_YT_VISION_ENABLED` on or off, changing `PINRAG_YT_VISION_PROVIDER`, or changing vision model / `PINRAG_YT_VISION_IMAGE_DETAIL` / frame limits, requires re-indexing affected YouTube documents for chunks to reflect the new behavior.

### Monitoring & Observability

For query performance metrics (latency, timing, token usage) and debugging, use [LangSmith](https://smith.langchain.com). Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in MCP `env` or your shell; optionally set `LANGSMITH_PROJECT` (see table above). **If your LangSmith workspace is in the EU region** (you use `eu.smith.langchain.com` in the browser), you **must** also set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`; without it, traces may not show up in the EU deployment. US-region accounts use the default API host and do not need `LANGSMITH_ENDPOINT`. See [`notes/langsmith-setup.md`](notes/langsmith-setup.md) for more detail.

For MCP-side introspection, set `PINRAG_VERBOSE_LOGGING=true` to surface detailed phase events in `notifications/message` (e.g., YouTube transcript load, whether vision runs, and chunk upsert milestones).

### Multiple providers and collections

Vector dimension is fixed per Chroma collection and must match the **`PINRAG_EMBEDDING_MODEL`** used when chunks were written. The default id **`nomic-embed-text-v1.5`** is a 768-d Nomic model; another **`PINRAG_EMBEDDING_MODEL`** value may imply a different size—check that model’s documentation.

- **Default:** `PINRAG_COLLECTION_NAME` defaults to **`pinrag`**. Do not change **`PINRAG_EMBEDDING_MODEL`** for an existing collection without re-indexing into a new collection (or wiping the old one); otherwise adds/queries can fail with embedding dimension errors.
- **Per-model collections:** Use a **stable pair** of **`PINRAG_EMBEDDING_MODEL`** + **`PINRAG_COLLECTION_NAME`** (+ **`PINRAG_PERSIST_DIR`** if you isolate stores) for each index. To query a collection, set the **same** env values you used when indexing it. You can index the same sources again under another pair (change `env`, restart MCP if needed, run `add_document_tool`).
- **MCP tools:** Each tool uses **`config.get_persist_dir()`** and **`config.get_collection_name()`** by default; optional **`collection`** on a tool call overrides the collection name for that request. **`list_collections_tool`** lists collection names in the configured persist directory (optional **`persist_dir`** override).

## MCP reference

Tools, prompt, and read-only resources from the **`pinrag`** MCP server (`FastMCP("PinRAG")`). Tool results are JSON objects that always include **`_server_version`**; with **`PINRAG_VERBOSE_LOGGING=true`** they may include **`_verbose_log`**.

`add_document_tool` returns `indexed`, `failed`, counts, `persist_directory`, `collection_name`, and **`fail_summary`** when any path failed. `query_tool` returns **`answer`** and **`sources`** (each entry: **`document_id`**, **`page`**—PDF page, often `0` for non-PDF—plus optional **`start`** in seconds for YouTube).

### `query_tool`

Natural-language question; optional filters narrow retrieval (`""` / omit when unused):

| Parameter | Description |
|-----------|-------------|
| `query` | Question (required) |
| `document_id` | Limit to this document — exact ref from `list_documents_tool`, list title, or unique PDF filename stem |
| `page_min`, `page_max` | Inclusive PDF page range (must pass both; one page: same value twice) |
| `tag` | Only chunks with this tag |
| `document_type` | `pdf`, `youtube`, `discord`, `github`, `plaintext`, or `web` |
| `response_style` | `thorough` or `concise`. Empty (the schema default) or any other string → resolved via **`PINRAG_RESPONSE_STYLE`** (see `server.py`: only those two literals override env). |

Filters can be combined. The **`sources`** list uses **`page`** for PDFs and **`start`** (seconds) for YouTube; answers may show **t. M:SS** labels derived from **`start`**. GitHub citations use chunk-index-style **p. N** labels in the answer text. Web docs sources carry a per-page **`source_url`**.

Example: *"What is OpenOCD? In the Pico doc, pages 16–17 only"* → `query_tool(query="What is OpenOCD?", document_id="RP-008276-DS-1-getting-started-with-pico.pdf", page_min=16, page_max=17)`.

### `add_document_tool`

Index locals (PDF, plain or Discord `.txt`, directories), YouTube (video URL, playlist URL, or bare id), GitHub URLs (scheme optional), or web documentation sites (any http(s) URL). **`paths`** batches work items; one failed path does not roll back others. Persists to **`PINRAG_PERSIST_DIR`** / **`PINRAG_COLLECTION_NAME`** only (no MCP parameters for those).

| Parameter | Description |
|-----------|-------------|
| `paths` | Required list: files, dirs, URLs, or video ids |
| `tags` | Optional; one per `paths` entry, same order |
| `branch` | GitHub only: branch override |
| `include_patterns` | GitHub only: glob include list |
| `exclude_patterns` | GitHub only: glob exclude list |

### `list_documents_tool`

Returns **`documents`**, **`total_chunks`**, **`persist_directory`**, **`collection_name`**, and **`document_details`** (tags, titles, counts, aggregated **`bytes`** when present, `upload_timestamp`, etc.). If **`tag`** is set, **`total_chunks`** counts only chunks with that tag (not the whole collection).

| Parameter | Description |
|-----------|-------------|
| `tag` | Optional: only docs that have this tag |


### `remove_document_tool`

Deletes every chunk for **`document_id`**. Accepts the exact ref from `list_documents_tool`, the list title, or a unique PDF filename stem.

| Parameter | Description |
|-----------|-------------|
| `document_id` | Required — exact ref, list title, or unique PDF stem |

### `set_document_tag_tool`

Sets or replaces the **`tag`** on every indexed chunk for one document. Useful to add or correct a tag after indexing, without re-indexing. Same document-targeting rules as `remove_document_tool`.

| Parameter | Description |
|-----------|-------------|
| `document_id` | Required — exact ref, list title, or unique PDF stem |
| `tag` | Required — non-empty tag string |
| `collection` | Optional override (default: `PINRAG_COLLECTION_NAME`) |

### MCP prompt: `use_pinrag`

Built-in routing blurb: **`request`** is interpolated as the first line; the rest lists when to use each tool and their parameters (matches `use_pinrag` in `server.py`). Listed wherever the client exposes MCP prompts (e.g. Cursor).

| Parameter | Description |
|-----------|-------------|
| `request` | Optional user goal (may be empty) |

### MCP resources

| Resource | Description |
|----------|-------------|
| `pinrag://documents` | Plain-text listing for the server’s configured collection (from `format_documents_list`) |
| `pinrag://server-config` | Printable dump of effective env/config (includes **`PINRAG_VERSION`**, key operational vars, API key presence) |

## Running tests

From the repo root, install dev extras (e.g. `uv sync --extra dev`).

- **Fast (no `integration`):**  
  `uv run pytest tests/ -q -m "not integration"`  
  Skips anything marked `integration` in `pyproject.toml` (network, API keys, optional assets, MCP stdio). Any test that uses the `sample_pdf_path` fixture gets that marker automatically in `tests/conftest.py`, so the sample PDF under `data/pdfs/` is only needed for the full run.

- **Full suite:**  
  `uv run pytest tests/ -q`  

  **Secrets:** For MCP stdio tests, the subprocess env starts from your shell, then any missing `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are filled from `tests/.mcp_stdio_integration.env` (copy from `tests/mcp_stdio_integration.env.example`; only those keys are read from the file—already-set env wins). Override the file with `PINRAG_MCP_ITEST_ENV_FILE`. After merge, `test_mcp_stdio_repo.py` requires `OPENAI_API_KEY`, and `PINRAG_LLM_PROVIDER` must pass its credential check (e.g. export `OPENROUTER_API_KEY` when using OpenRouter). `test_mcp_stdio_pypi.py` also requires a working OpenAI key.

  **PDF / stdio:** Default PDF is `data/pdfs/sample-text.pdf` (not in git). Override with `PINRAG_MCP_ITEST_PDF` / `PINRAG_MCP_ITEST_QUERY`. Stdio tests need `uv` on `PATH`, or set `PINRAG_TEST_UV` to the binary path.

  **PyPI MCP test:** Marked `pypi_mcp`; skip with `-m "not pypi_mcp"` or `PINRAG_MCP_ITEST_SKIP_PYPI=1`. Pin the install with `PINRAG_MCP_ITEST_PYPI_SPEC` (default `pinrag` = latest on PyPI).

  Verbose: `--log-cli-level=INFO`.

The `data/` directory is gitignored—create `data/pdfs/` (and similar) locally; nothing under `data/` is committed.

## License

MIT License. Full text in [LICENSE](LICENSE).
