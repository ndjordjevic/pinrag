# PinRAG Implementation Checklist

## Phase 0: Planning & Setup

- [x] **Framework Decision**
  - [x] Evaluate LangChain for simple RAG pipeline
  - [x] Evaluate LangGraph for complex workflows with state management
  - [x] Make decision and document rationale (✅ Decision: Start with LangChain)

- [x] **Project Setup**
  - [x] Initialize Python project structure
  - [x] Set up dependency management (requirements.txt or pyproject.toml)
  - [x] Create project directory structure
  - [x] Set up development environment

---

## Phase 1: MVP - Core Functionality **(v1.0.0)**

**Goal:** Working PDF RAG that can load PDFs, answer questions, and provide basic source citations.

### PDF Processing
- [x] **PDF Library Selection**
  - [x] Research PDF parsing libraries (PyPDF2, pdfplumber, pypdf, etc.)
  - [x] Choose PDF library (**pypdf**)
  - [x] Install and test basic PDF text extraction (**pypdf** smoke-tested)

- [x] **Basic PDF Processing**
  - [x] Implement single PDF loading
  - [x] Extract text from PDF files
  - [x] Test parsing with sample PDFs and verify output (e.g. `scripts/print_pdf_page.py`)
  - [x] Extract basic metadata (page numbers, document name, document title/author from PDF)
  - [x] Handle basic text-based PDFs (raise ValueError when no text extracted; Phase 1 text-only)

### Chunking & Embeddings
- [x] **Chunking Strategy**
  - [x] Choose chunk size (default: 1000 chars)
  - [x] Choose chunk overlap (default: 200 chars)
  - [x] Implement text splitter (LangChain RecursiveCharacterTextSplitter)
  - [x] Preserve page numbers in chunks
  - [x] Add document identifier to chunks

- [x] **Embedding Setup**
  - [x] Choose embedding model (**local Nomic** `nomic-embed-text-v1.5` via `langchain-nomic`; historical context in [embedding-provider-decision.md](embedding-provider-decision.md))
  - [x] Set up embedding model client (`get_embedding_model()` → `NomicEmbeddings`, `inference_mode="local"`)
  - [x] Test embedding generation (incl. pipeline test: PDF load → chunk → embed a few chunks in `tests/test_embeddings.py::test_embed_pdf_chunks`)

### Vector Store
- [x] **Vector Database Selection**
  - [x] Research vector databases (Chroma, FAISS, Pinecone, etc.) — see [vector-database-decision.md](vector-database-decision.md)
  - [x] Choose vector database (**Chroma** for local, simple setup)
  - [x] Install and configure vector database (`chromadb`, `langchain-chroma`; `pinrag.vectorstore.get_chroma_store()` with persist dir `chroma_db`)
  - [x] Unit tests for Chroma store (`tests/test_vectorstore.py`: get_chroma_store, persist dir, add_documents + similarity_search)

- [x] **Vector Store Implementation**
  - [x] Implement embedding storage with metadata (via `index_pdf` → `get_chroma_store().add_documents()`)
  - [x] Implement similarity search/retrieval (`query_index` + Chroma `similarity_search`; tests in `tests/test_indexing.py` and `tests/test_vectorstore.py`)
  - [x] Test basic retrieval functionality (CLI: `scripts/index_cli.py`, `scripts/query_rag_cli.py`; tests: `test_indexing.py`, `test_vectorstore.py`)

### RAG Pipeline
- [x] **LLM Setup**
  - [x] Choose LLM provider (OpenAI for chat/completion)
  - [x] Set up LLM client (`pinrag.llm.get_chat_model()` — ChatOpenAI, gpt-4o-mini; OPENAI_API_KEY from env)
  - [x] Test basic LLM calls (`tests/test_llm.py`, `scripts/test_llm_cli.py`)

- [x] **RAG Chain Implementation**
  - [x] Design prompt template with context and question (`pinrag.rag.prompts.RAG_PROMPT`)
  - [x] Implement retrieval step (wire `query_index` in chain)
  - [x] Implement context formatting (`format_docs`, numbered with doc/page for citations)
  - [x] Implement generation step (prompt | llm | StrOutputParser)
  - [x] Implement response formatting with citations (`format_sources`, `RAGResult`-like output)
  - [x] Build RAG pipeline (plain Python `run_rag()` with `@traceable`; CLI: `scripts/rag_cli.py`)

- [x] **Observability & Development Tools**
  - [x] Setup LangSmith (tracing, observability; see `notes/langsmith-setup.md`)
  - [x] Learn how to read and interpret LangSmith traces (understand trace structure, timing, inputs/outputs, debugging workflow)
  - [x] Understand all code written so far (review PDF processing, chunking, embeddings, vector store, RAG chain, scripts/CLIs, tests) before moving to MCP implementation

### MCP Server - Basic
- [x] **MCP Server Setup**
  - [x] Research MCP server implementation patterns (see `notes/mcp-server-research.md`)
  - [x] Install MCP Python SDK (`mcp>=1.26.0`)
  - [x] Create `src/pinrag/mcp/` module structure
  - [x] Set up FastMCP server framework (`mcp/server.py`)
  - [x] Create entry point script (`scripts/mcp_server.py`)

- [x] **Basic MCP Tools**
  - [x] Implement `query_pdf` tool (wraps `run_rag()`)
  - [x] Implement `add_pdf` tool (wraps `index_pdf()`)
  - [x] Implement `remove_pdf` tool (remove PDF and all its chunks/embeddings from Chroma)
  - [x] Implement `list_pdfs` tool (list all indexed books in PinRAG)
  - [x] Add error handling and input validation
  - [x] Test with MCP Inspector (`npx @modelcontextprotocol/inspector`)
  - [x] Add unit tests (`tests/test_mcp_server.py`)
  - [x] Test MCP server from Cursor (add PinRAG to Cursor MCP config, invoke query_pdf / add_pdf / list_pdfs; see `notes/cursor-mcp-setup.md` and `.cursor/mcp.json`)

---

## Phase 1.5: Configuration, Persistence & Error Handling

### Configuration & Persistence
- [x] **Configuration Management** (env vars)
  - [x] Set up configuration — `PINRAG_*` and provider API keys via MCP `env` or shell; see `notes/env-vars.example.md` and README
  - [x] Make chunk size configurable (`PINRAG_CHUNK_SIZE`, default 1000)
  - [x] Make chunk overlap configurable (`PINRAG_CHUNK_OVERLAP`, default 200)

- [x] **Persistence** (done in Phase 1)
  - [x] Implement vector store persistence — Chroma persists to `chroma_db` (or custom path) on disk
  - [x] Test persistence across restarts — `test_index_pdf_smoke` re-opens store and queries; index survives process restarts
  - [x] Handle data directory setup — `get_chroma_store()` creates persist dir if missing (`mkdir(parents=True, exist_ok=True)`)

- [x] **Duplicate PDF detection**
  - [x] On add_pdf/index_pdf: if document_id (file name) already in index, delete existing chunks then add new ones (replace) to avoid duplicates

### Error Handling
- [x] **Basic Error Handling**
  - [x] Handle PDF loading errors — propagate; tools raise FileNotFoundError, ValueError with clear messages
  - [x] Handle embedding generation errors — propagate; logged and returned to client
  - [x] Handle retrieval errors — propagate; logged and returned to client
  - [x] Handle LLM generation errors — propagate; logged and returned to client
  - [x] Add basic error messages — tools validate and raise with clear messages; `_log_tool_errors` decorator emits MCP notifications and then re-raises

### Testing - MVP
- [x] **Basic Testing**
  - [x] Test PDF loading with sample PDF (`tests/test_pypdf_loader.py`)
  - [x] Test chunking and metadata preservation (`tests/test_chunking.py`)
  - [x] Test embedding generation (`tests/test_embeddings.py`)
  - [x] Test retrieval functionality (`tests/test_vectorstore.py`, `tests/test_indexing.py`)
  - [x] Test end-to-end RAG pipeline (`tests/test_rag.py`)
  - [x] Test MCP tools

- [x] **Configuration, Persistence & Error Handling**
  - [x] Test configuration (load config, override chunk size/overlap) — `tests/test_config.py` (env getters); `test_index_pdf_uses_config_chunk_size_from_env` (index_pdf uses env); `test_index_pdf_respects_chunk_size_override` (override via kwargs)
  - [x] Test persistence (index PDF, restart or new process, verify index survives and queries work) — covered by `test_index_pdf_smoke`
  - [x] Test error handling (PDF load failures, embedding/retrieval/LLM errors return clear messages; CLI and MCP) — `test_mcp_server.py` covers validation errors; errors propagate and are logged via `_log_tool_errors`
  - [x] Test duplicate PDF detection (add same PDF twice: replace; verify no duplicate chunks) — `test_index_pdf_replaces_duplicate`

### Deployment & Distribution - MVP
- [x] **Document how others can start and use this MCP**
  - [x] PyPI install: `pip install pinrag` → `pinrag` CLI; config from cwd or `~/.config/pinrag/` (see `README.md`, `notes/distribution-options.md`)
  - [x] Cursor MCP config: command `pinrag` (global install) or `uvx` + `args: ["--refresh", "pinrag"]` (no cwd needed; Chroma under `PINRAG_PERSIST_DIR`, default project-local `chroma_db`)
  - **Publish new version to PyPI:** Bump `version` in `pyproject.toml` → `git add -A && git commit -m "vX.Y.Z: ..." && git push` → `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z` → `uv build && uv publish` (use `__token__` + PyPI API token when prompted; or `uv cache clean` before `uv tool install pinrag --force` to get latest)
  - [x] Github action to publish to PyPI on a new tag/release

---

## Phase 2: Enhanced Features

**Goal:** Document management, better metadata, improved retrieval, modernize the RAG chain.

### Releases - GitHub
- [x] **Manual GitHub Release for stable versions**
  - [x] When ready (e.g. after Phase 2 milestone), pick the latest `vX.Y.Z` tag and create a GitHub Release with notes (no assets).
  - [x] Prefer a **manual** release (not per-tag) so only meaningful milestones get a Release page.
  - [x] CLI option: `gh release create vX.Y.Z --notes \"Summary of changes\"` (requires GitHub CLI + auth).

### RAG Chain Rewrite
- [x] **Replace LCEL with plain Python (2-step RAG)**
  - *Done*: `run_rag()` plain function + `@traceable` now powers the RAG pipeline (retrieve → format context → generate answer + sources). LCEL (`RunnablePassthrough`/`RunnableLambda`/`StrOutputParser`) has been removed in favor of this pattern from the LangChain docs.

### PDF Processing - Enhanced
- [x] **Multiple PDF Support**
  - [x] Implement batch PDF loading (new `add_pdfs` MCP tool)
  - [x] Test with multiple PDFs (unit coverage for partial success/failure)

- [x] **Enhanced Metadata**
  - [x] Add explicit section/heading labels to chunk metadata (`section` key; heading-like lines detected and carried forward)
  - [x] Add upload timestamps for indexed documents (stored on chunks, aggregated per document in list_pdfs)
  - [x] Store document size stats (pages, bytes, total chunks) per document (stored on chunks, aggregated per document in list_pdfs)

### Chunking - Enhanced
- [x] **Section/heading metadata** — done (heading-like lines detected; `section` key added to chunk metadata)
- [x] **Character offsets** — enable `add_start_index=True` on `RecursiveCharacterTextSplitter`; each chunk gets `start_index` metadata (char offset in page)

### Vector Store - Enhanced
- [x] **Document Tag** (single tag per document)
  - [x] Add optional per-document tag at indexing time (e.g. `tag: "amiga"`)
  - [x] Persist tag on all chunks for that document (queryable via metadata filters)
  - [x] Expose tag in `list_pdfs` output (per-document details)

- [x] **Metadata Filtering**
  - [x] Implement filter by document (optional `document_id` on query_index, run_rag, query_pdf)
  - [x] Implement filter by page range (optional `page_min`, `page_max`; single page: page_min=64, page_max=64)
  - [x] Implement filter by tag (optional `tag` on query_index, run_rag, query_pdf)

---

## Phase 3: Advanced Configuration & Deployment

### Configuration - Enhanced
- [x] **Embedding and LLM configuration**
  - [x] Make embedding model configurable (env or config file)
  - [x] Make LLM provider configurable (env or config file)

### MCP Server - Enhanced
- [x] **MCP Resources** (read-only data by URI)
  - [x] Expose list of indexed documents as resource (`pinrag://documents`)

- [x] **MCP Prompts** (pre-built templates with parameters)
  - [x] “Use PinRAG (query, index, list, remove)” (`use_pinrag`)

### Retriever Abstraction
- [x] **Retriever abstraction (pinrag 2.0)**
  - [x] Use LangChain Retriever (`store.as_retriever()`) in RAG pipeline instead of `query_index`
  - [x] Refactor `run_rag()` to accept a retriever

### Error Handling - Enhanced
- [x] **Advanced Error Handling**
  - [x] **Handle corrupted or unreadable PDFs** — In `load_pdf_as_documents`, catch `PyPdfError` and `OSError` when opening/reading the PDF and re-raise as `ValueError` with a user-facing message.
  - [x] **Zero-retrieval and LLM failure handling (graceful degradation)** — In `run_rag`: when retrieval returns 0 chunks, return early with a clear message (no LLM call); when the LLM invoke fails, return a short error message (rate limit, timeout, or generic) instead of propagating the traceback.

### Testing - Enhanced
- [x] **Comprehensive Testing**
  - [x] Integration tests for multiple PDFs — `test_run_rag_multiple_pdfs_document_id_and_tag_filters` in `test_rag.py`: indexes two PDFs (same content, different names/tags), then queries with document_id and tag filters and asserts sources are from the correct document; unfiltered query returns sources from either doc.

---

## Phase 4: Advanced Retrieval & Evaluation

**Goal:** Close the correctness gap (currently 60% OpenAI vs 73% Anthropic baseline) by improving retrieval quality, then iterate on prompts using evaluation results.

### Evaluation Iteration
- [x] **Evaluation framework** (do before complex retrieval changes). See `notes/evaluation-strategy.md`.
  - [x] Create evaluation dataset (question / expected-answer pairs). Golden dataset **pinrag-golden** in LangSmith: 30 examples from *Bare-metal Amiga programming 2021_ocr.pdf* (easy → hard); script: `scripts/create_eval_dataset.py`.
  - [x] Implement evaluators and target: correctness, relevance, groundedness, retrieval relevance (LLM-as-judge) plus code evaluators (has_sources, answer_not_empty, source_in_expected_docs); target function wrapping `run_rag` with `document_id`/`tag` from dataset inputs. See evaluation-strategy §4–5.
  - [x] Run the first experiment with Anthropic for LLM and Cohere for embeddings.
  - [x] Re-run the experiment with OpenAI both for LLM and embeddings.
  - [x] Make sure you understand the metrics and why the results are like they are and some answers are wrong. **Why grades are bad:** (1) Retrieval often misses expected pages (93% of failures) → multi-query, hybrid search, increase k; (2) deep-in-book content underretrieved (55% of wrong-answer pages > page 100) → hybrid search, reranker; (3) topically adjacent wrong chunks rank high → reranker, multi-query; (4) nearly-correct answers marked 0 (~2–3 cases) → lenient reference or partial-credit grader.
  - [x] **Increase retrieval k (one constant):** In `src/pinrag/evaluation/target.py`, change `k=5` to e.g. `k=8` or `k=10` in `create_retriever(...)`; re-run evaluation and compare correctness. **Done (k=10):** correctness 53.3% → 60.0% (+2 questions); baseline 73.3% — modest gain, gap remains; next: multi-query / hybrid / reranker.

### Discord Channel Ingestion
*Context: DiscordChatExporter TXT format — header block (`Guild: / Channel:`) then messages as `[date] author\ncontent\n{Reactions}/{Attachments}/{Embed}` blocks. Files live in `data/discord-channels/<channel-slug>/`. One full export per channel (historical), then periodic incremental exports (date-filtered) appended as new files.*

- [x] **Discord TXT loader** — Write `src/pinrag/indexing/discord_loader.py` (or extend `pdf_indexer.py`) that:
  - Reads a DiscordChatExporter `.txt` file.
  - Strips the header block, `{Reactions}`, `{Attachments}` CDN URLs, `{Embed}` noise, and raw emoji.
  - Converts each message (or conversational window of N messages) into a `Document` with metadata: `source=discord`, `channel`, `guild`, `author`, `timestamp`, `document_id`, `tag`.
  - Chunks conversation in context-preserving windows (e.g. 10–20 consecutive messages per chunk) rather than per-message — preserves thread context for retrieval.
- [x] **MCP tool `add_document_tool`** — Single tool for adding files: accepts paths (file, directory, YouTube, GitHub), auto-detects format (PDF vs DiscordChatExporter .txt via Guild:/Channel: header), routes to index_pdf or index_discord (or YouTube/GitHub), indexes into vector store. Replaces separate add_pdf/add_discord_export.
- [x] **Incremental ingestion** — Per-file `document_id`: each Discord TXT gets `discord-{channel}--{sanitized-filename-stem}` (e.g. `discord-alicia-1200-pcb--retro-tinkering-...-1404852770154217664` for full, `...--...-after-2026-03-06` for incremental). Additive: new files add chunks without replacing others. Query by `tag` for whole channel.
- [x] **CLI / script helper** — `scripts/index_discord_channels.py` scans `data/discord-channels/` (root + subfolders) and indexes all Discord `.txt` files via the add-document indexing logic. Run manually or cron. Options: `--base-dir`, `--persist-dir`, `--collection`, `--tag`.
- [x] **Test: index Discord export and query via MCP** — Index the full export from `data/discord-channels/` (e.g. alicia-1200-pcb channel) using `add_document_tool`, then run questions about Alicia 1200 (ROM programmer, PSU, etc.) via `query_tool` in Cursor. Confirm answers and sources use the correct `document_id`, `channel`, and `tag` metadata.

### Retrieval Improvement
- [x] **Advanced Retrieval Strategies** (ordered by expected impact; items deferred to later phases noted below)
  - [x] **Re-ranking** ⭐ — Implemented Cohere Re-Rank via `ContextualCompressionRetriever` + `CohereRerank`. Config: `PINRAG_USE_RERANK` (default false), `PINRAG_RETRIEVE_K` (default 20), `PINRAG_RERANK_RETRIEVE_K` (default 20), `PINRAG_RERANK_TOP_N` (default 10). **Recommended when enabled:** k=20→top_n=10 with `claude-haiku-4-5`; achieves 10/10 on hard-10 and 30/30 on golden. No-rerank: k=10 suffices for min_k on golden (two questions need k=10). See `evaluation-strategy.md §8` for full sweep.
  - [x] **Query translation / multi-query** ⭐⭐⭐ — Implemented via `langchain_classic.retrievers.multi_query.MultiQueryRetriever`. Config: `PINRAG_USE_MULTI_QUERY` (default false), `PINRAG_MULTI_QUERY_COUNT` (default 4). Generate 3–5 query variants via LLM, retrieve per variant, merge (unique union), then optionally rerank. When rerank is off, merged docs are truncated to `PINRAG_RETRIEVE_K` to avoid context overflow. See `pinrag.rag.multiquery`, `evaluation-strategy.md` (multiquery-stress dataset).
  - [x] **Query preprocessing** — Implemented in `pinrag.rag.query_preprocess`: normalize whitespace, strip common MCP/agent boilerplate (e.g. "User question:", "Query:", "Search the documents for:"). Applied before retrieval only; original query kept for prompt.
  - [x] **Prompt engineering iteration** — Implemented: (1) internal step-by-step instruction in system prompt (“Think step-by-step internally before finalizing your answer, but do not reveal hidden reasoning”); (2) response style **thorough** (default) vs **concise** via `get_rag_prompt(response_style)` and `run_rag(..., response_style=...)`. Config: `PINRAG_RESPONSE_STYLE` (thorough | concise); evaluation target and MCP `query` tool support it. See `pinrag.rag.prompts`, `evaluation-strategy.md` (response style comparison).

### Chunking Improvement
- [x] **Chunk size tuning** (needs evaluation framework first) — Tune chunk size to ~512 tokens (~2000 chars) with 10–20% overlap; benchmark retrieval quality before/after with evaluation dataset.
- [x] **Parent-child retrieval** — Implemented via LangChain `ParentDocumentRetriever`. Embed small child chunks (400 chars default) for precise matching, return larger parent chunks (2000 chars default) for context. Config: `PINRAG_USE_PARENT_CHILD` (default false), `PINRAG_PARENT_CHUNK_SIZE`, `PINRAG_CHILD_CHUNK_SIZE`. Uses `LocalFileStore` docstore for persistence. Requires re-indexing when enabling.
- [x] **Structure-aware chunking** — Implemented in `pinrag.chunking.splitter`: `_ensure_code_block_breaks()` (detect C and 68k assembly, insert `\n\n` at prose↔code transitions) and `_ensure_table_breaks()` (detect aligned/register tables, insert `\n\n` at prose↔table transitions). RecursiveCharacterTextSplitter then prefers breaking at those boundaries. Config: `PINRAG_STRUCTURE_AWARE_CHUNKING` (default true). Tests: `tests/test_structure_aware_chunking.py`, `tests/test_structure_config.py`. Helps digitally-born C/technical PDFs (e.g. Pi Pico book); Amiga OCR PDFs benefit less due to two-column extraction.

---

## Phase 5: Polish & Production Ready

**Goal:** Add YouTube and GitHub indexing, then production-grade reliability, deployment packaging, and documentation.

### MCP & Configuration Polish
- [x] **Revise and polish MCP tools params** — Review tool parameter names, descriptions, defaults, and validation across all MCP tools.
- [x] **Revise and polish MCP resources and prompts** — Review resource URIs, prompt templates, and argument descriptions for clarity and consistency.
- [x] **Revise and polish MCP server params** — Configure via MCP `env` / process environment (12-factor, no YAML). Documented in `notes/env-vars.example.md`, README config table, and server config resource (set vs defaults).

### New Document Types (Indexing)
- [x] **YouTube indexing** — Index video transcripts via `youtube-transcript-api`. Load transcript → chunk → Chroma; metadata: `document_id` (video ID), `document_type`, `video_id`, `start` (seconds), `duration`, `title` (via oEmbed), `source` (canonical URL). MCP `add_document_tool` detects YouTube URLs/IDs; citations show timestamp (e.g. `t. 1:23`). Handle `TranscriptsDisabled`, `NoTranscriptFound` with clear errors. `list_documents_tool` and `documents_resource` show title and segments.
- [x] **YouTube vision enrichment (optional)** — When `PINRAG_YT_VISION_ENABLED=true`, download video (yt-dlp), scene-detect keyframes (`pinrag[vision]` + ffmpeg), run OpenAI or Anthropic vision on frames, merge on-screen descriptions with transcript segments before chunking; metadata includes `has_visual`, `frame_count`, `visual_source`. Default off; see README.
- [x] **GitHub repo indexing** — Index repo contents (README, code, docs). Load via GitHub API; chunk markdown and code; metadata: `document_id` (owner/repo), `source` (blob URL per file). MCP `add_document_tool` detects GitHub URLs; optional: `branch`, include/exclude patterns (e.g. `*.md`, `docs/`). File-size cap via PINRAG_GITHUB_MAX_FILE_BYTES (default 512 KB).
- [x] **YouTube playlist indexing** — Index all videos in a playlist. Uses yt-dlp (`flat_playlist`, no API key) to fetch video IDs; for each video, reuses `index_youtube` (youtube-transcript-api → chunk → Chroma). MCP `add_document_tool` detects playlist URLs (`youtube.com/playlist?list=...`); each video is a separate document (document_id = video_id); metadata: `playlist_id`, `playlist_title` for filtered search. Parent-child indexing applied per video when enabled.
- [x] **Plain TXT file indexing** — Index plain text files (.txt) that are not Discord exports. Load file → chunk → Chroma; metadata: `document_id` (filename), `document_type` ("plaintext"), `source` (file path). MCP `add_document_tool` detects .txt; distinguish from Discord format (check for `Guild:`/`Channel:` header—Discord takes precedence). Citations show document_id (no page; optional line range if tracked). Reuse existing chunking; optional file-size cap via config.

### Improve Logging
- [x] **MCP tool/resource logging** — Tool lifecycle events are emitted through MCP `notifications/message` (`ctx.log`) so clients can render structured logs without stderr duplication/noise. Logs include tool name + args summary, completion timing, and errors.

### Multi-threading
- [x] **Multi-threading** — MCP server tools/resources converted to async with `anyio.to_thread.run_sync`; heavy indexing (YouTube playlist, GitHub repo) no longer blocks the event loop. You can index from one agent and query or read the documents resource from another at the same time.

### Deployment & Operations
- [x] **Deployment packaging** — Distribute via PyPI (`pip install pinrag` / `pipx install pinrag` / `uv tool install pinrag`). See notes/distribution-options.md. Docker and hosted MCP deferred for now.
- [x] **Backup and restore** — Documentation only: README notes that users should back up `~/.pinrag/chroma_db` (or `PINRAG_PERSIST_DIR`) for migration and recovery. Restore = copy the directory back. No programmatic export/import.

### Monitoring & Observability
- [x] **Metrics & Logging** — LangSmith for query performance (latency, timing, token usage); README documents setup (notes/langsmith-setup.md). MCP notifications provide structured tool lifecycle logs in clients. Error tracking via existing exception logging. Retrieval quality metrics remain eval-only (LLM-as-judge per query is costly).

### Misc
- [x] Update PyPi package description: set in `pyproject.toml` — `description` = one-line summary (PyPI summary); `readme = "README.md"` = long description (PyPI project page body). Re-publish to PyPI for changes to appear.
- [x] Test pinrag mcp in visual studio code locally.
- [x] Polish README.md
- [x] **Investigate** to add pinrag mcp to visual studio code extensions marketplace — see `notes/vscode-marketplace-investigation.md`. Summary: (A) VS Code extension wrapping `uvx --refresh pinrag` for Extensions view; (B) protocol install URL (`vscode:mcp/install?…`) for README; (C) curated list (process TBD). Recommend B first, then A.
- [x] **Investigate** how to add pinrag mcp to cursor mcp server list — see `notes/cursor-mcp-list-investigation.md`. Summary: Submit via (1) [Cursor Directory](https://cursor.directory/plugins/new) (replaces deprecated cursor/mcp-servers), (2) cursor.store/mcp/new, (3) mcp-marketplace.io/submit. Cursor MCP install link: `cursor.com/en/install-mcp?name=pinrag&config=...`
- [x] Code review of the src and tests folders.
- [x] Fix code review findings.
- [x] Test two modes of running the stdio pinrag mcp server: one from the repo and one from PyPI (same MCP steps).
  - [x] **`uv run --project <repo> pinrag`:** integration test [`tests/test_mcp_stdio_repo.py`](tests/test_mcp_stdio_repo.py) (`test_pdf_roundtrip_local_repo`) — spawns `uv run --project <repo> pinrag`, calls `add_document_tool` → `list_documents_tool` → `query_tool` → `remove_document_tool` on a local PDF (default `data/pdfs/sample-text.pdf`, or `PINRAG_MCP_ITEST_PDF` / `PINRAG_MCP_ITEST_QUERY`), uses a temp `PINRAG_PERSIST_DIR` (never the repo `chroma_db`). **Keys:** `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment (CI secrets), or from `tests/.mcp_stdio_integration.env` (copy from [`tests/mcp_stdio_integration.env.example`](tests/mcp_stdio_integration.env.example), gitignored), or `PINRAG_MCP_ITEST_ENV_FILE` pointing at a `KEY=value` file; optional deprecated: `PINRAG_ITEST_USE_CURSOR_MCP_JSON=1` merges `~/.cursor/mcp.json` `pinrag-dev` env. **Run:** `pytest -m integration tests/test_mcp_stdio_repo.py -v`. **Verbose progress (optional):** add `--log-cli-level=INFO`.
  - [x] **PyPI (same as `uvx pinrag` / `uv tool run --from pinrag pinrag`):** integration test [`tests/test_mcp_stdio_pypi.py`](tests/test_mcp_stdio_pypi.py) (`test_pdf_roundtrip_pypi_package`) — shared flow in [`tests/helpers/mcp_stdio_flow.py`](tests/helpers/mcp_stdio_flow.py). **Opt out:** `PINRAG_MCP_ITEST_SKIP_PYPI=1`. **Pin version:** `PINRAG_MCP_ITEST_PYPI_SPEC=pinrag==…`. **Run:** `pytest -m integration tests/test_mcp_stdio_pypi.py -v` (marker `pypi_mcp`). Exclude PyPI test: `-m "integration and not pypi_mcp"`.
- [x] Check README.md for the pinrag mcp server and update it if needed. Check it how it looks like on GitHub.
- [x] Check pinrag repo and what's been pushed there and should we expose it to the users?
- [x] Investigate advertising & distribution for the PinRAG MCP server (IDE lists, directories, promotion) — see `notes/mcp-advertising-distribution-strategy.md`.
- [x] Implement distribution strategies
- [x] Investigate how to smartly extract YouTube video content beside just the transcript.
- [x] Implement local Nomic embeddings.
- [x] Add OpenRouter as a default provider for LLM.
- [x] Investigate how to use Cerebras skill to use LiteLLM via OpenRouter to the openrouter/openai/gpt-oss-120b model with Cerebras as the inference provider to improve the RAG pipeline.
- [x] Add Cerebras Inference provider to the pinrag mcp server.
- [x] Polish README.md and make it more user-friendly.
- [x] Implement advertising strategies
- [x] **pinrag-cli Phases 0–3a** — See [notes/pinrag-client-study.md](notes/pinrag-client-study.md). **Phase 0:** transport-agnostic ops in `pinrag.core`. **Phase 1:** REPL, in-process `BackendClient`, slash commands, Rich output, prompt_toolkit + line history. **Phase 2:** `pinrag server` (streamable-http), `pinrag-cli --server …` MCP client, Rich streaming progress, JSON session history, `/switch`. **Phase 2.5:** in-session conversational memory—rolling Q/A window folded into the next query (bounded), `/clear`, `PINRAG_CLI_MEMORY` / `PINRAG_CLI_MEMORY_TURNS`; JSON `/history` unchanged. **Phase 3a:** CLI config—`~/.config/pinrag-cli/config.toml` + per-project `.pinrag-cli.toml` (precedence: CLI flags → env → project → user → defaults); CLI-owned keys only (`collection`, `server_url`, `response_style`, memory); `--response-style`; `/config` + `/config set` (writes user TOML); reload preserves `/switch` and rolling memory when unrelated settings change. **Later:** Phase 3b–3d (sessions polish, multi-collection UX, Textual TUI) in that note.

---

## Phase 6: Document Intelligence & Advanced Indexing

**Goal:** Smarter document understanding — richer PDF parsing, better chunking strategies, and query-time document intelligence, and misc.

### Misc
- [ ] Marketing strategies for the open-source version.
- [ ] Add pinrag mcp to visual studio code extensions marketplace.
- [ ] Add pinrag mcp to cursor mcp server list.
- [ ] Check OpenRouter AI for a free AI models.
- [ ] Check RAG accuracy (independent benchmark) for our RAG pipeline comparing to existing RAG pipelines.

### Advanced PDF Processing
- [ ] **OCR integration** — Detect image-only pages and run OCR automatically (or offer as option in `add_pdf`). Research done: ocrmypdf + Tesseract used externally.
- [ ] **Complex structure handling** — Handle multi-column layouts, tables (preserve as atomic chunks), code blocks, and embedded images/diagrams.
- [ ] **pdfplumber extraction backend** — Add pdfplumber as optional PDF extraction backend for better table extraction (atomic chunks), multi-column layout handling, and richer structure signals. Addresses two-column garbling in OCR'd PDFs (e.g. Amiga book). See notes/pdf-parsing-library-analysis.md. MIT licensed.
- [ ] **Richer structure signals** — Use PDF outline/bookmarks as section/heading labels; detect headings via font size/style (pdfplumber or PyMuPDF).
- [ ] **Filter by section** — Implement section filter in `query_index`, `run_rag`, `query_pdf` (depends on reliable `section` metadata above).

### Advanced Chunking
- [ ] **Multi-representation indexing** (optional) — Embed per-chunk summaries/propositions in the vector store; store full raw docs in a separate doc store keyed by doc_id. Ref: rag-from-scratch 12.
- [ ] **RAPTOR / hierarchical indexing** (optional) — Cluster leaf chunks, summarize clusters, recurse; index all levels together. Ref: RAPTOR paper; rag-from-scratch 13.

### Advanced Retrieval
*(Deferred from Phase 4 — overlaps with multi-query/hybrid or requires re-indexing infrastructure.)*
- [ ] **Hybrid search** ⭐⭐ — Add a keyword (BM25) retrieval path alongside vector search and combine scores. **High impact for deep-in-book content** (specific register names, opcodes, exact terms that embeddings spread across semantic neighbours). Especially important as corpus grows (more PDFs, Discord). Options: `BM25Retriever` + `EnsembleRetriever` over the same docs in memory. Chroma has no native BM25; if memory becomes a concern, consider Qdrant or Weaviate.
- [ ] **HyDE (Hypothetical Document Embeddings)** — Embed a LLM-generated hypothetical answer passage instead of the raw question; retrieves docs closer in style/length to the actual chunks. Overlaps with multi-query (Phase 4); revisit if multi-query + hybrid leave a gap. Ref: HyDE paper; rag-from-scratch series.
- [ ] **ColBERT** — Token-level retrieval (MaxSim scoring); avoids compressing a full chunk into one vector. High recall ceiling but requires a ColBERT index (e.g. RAGatouille) and re-indexing. Consider only if reranker + multi-query + hybrid still leave a measurable gap. Ref: ColBERT; rag-from-scratch 14.
- [ ] **MMR (Maximal Marginal Relevance)** — Optional retriever diversity: balance relevance vs diversity to reduce redundant chunks. See notes/mmr-analysis.md. Adds moderate value when rerank is OFF; low when rerank is ON. LangChain Chroma supports `search_type="mmr"`, `fetch_k`, `lambda_mult`.

### Document Intelligence
- [ ] **Document name resolution** — Accept fuzzy doc references in queries (e.g. "pico debug probe manual"), resolve to exact `document_id` from `list_pdfs`; add auto-complete helper.
- [ ] **Query structuring** (optional) — LLM converts natural language into structured metadata filters (Pydantic schema: `document_id`, `tag`, `page_min`, `page_max`). Ref: rag-from-scratch 11; LangChain query analysis.
- [ ] **Document update/re-indexing** — Implement update beyond remove + re-add; document status tracking; version-aware retrieval.

### Advanced MCP & Citations
- [ ] **Advanced MCP Features** — Expose query history; streaming support; advanced filtering options.
- [ ] **Advanced Citations** — Confidence scores per source chunk; link back to document sections; improved citation formatting.

### Evaluation
- [ ] **Pairwise prompt experiments** — Run two prompt variants as separate LangSmith experiments on the golden dataset, then use `langsmith.evaluate(("exp-A-id", "exp-B-id"), evaluators=[ranked_preference])` to have an LLM judge pick the better answer head-to-head (more informative than absolute scores when both produce partially-correct answers). Ref: intro-to-langsmith/notebooks/module_2/pairwise_experiments.ipynb.

### Security & Access Control
- [ ] **Security Features** — Document-level access control, query authentication, audit logging, security review.

### Scalability
- [ ] **Performance Optimization** — Caching strategies, batch indexing parallelism (`ThreadPoolExecutor`), scalability test with hundreds of PDFs.

### Evaluation & Retrieval Polish
*(Deferred from Phase 4 — low impact given current reranking pipeline.)*
- [ ] **RAG Fusion: top-K after merge** — Reciprocal rank fusion to merge per-query result lists; pass only top K to the LLM. Redundant when Cohere reranking is on (reranker already re-scores merged set); consider only if reranking is removed. Ref: rag-from-scratch.
- [ ] **Long-context ordering** — Order retrieved docs so best-ranked chunks appear at the start and end of the context ("lost in the middle" effect). Negligible with ≤10 reranked chunks and modern models; revisit if chunk count grows significantly. Ref: Lost in the middle; rag-from-scratch 18.
- [ ] **Lenient / partial-credit grader** — For the ~2–3 nearly-correct answers marked 0: relax reference expectations or add a partial-credit correctness grader. Improves metrics fairness, not retrieval.

---

## Phase 7: Agentic Reasoning & LangGraph

**Goal:** Move from a fixed retrieve→generate pipeline to a stateful, agentic graph that can route, loop, decompose, and hold conversation context.

### Routing
- [ ] **Logical routing** — LLM with structured output (Pydantic) picks data source or retrieval strategy per query. Ref: rag-from-scratch 10.
- [ ] **Semantic routing** — Embed question + candidate strategy prompts, pick most similar. Ref: rag-from-scratch 10.

### Multi-step Reasoning
- [ ] **Query classification & conditional routing** — Detect query type (factual, conceptual, multi-part); route to different retrieval strategies via LangGraph conditional edges.
- [ ] **Query decomposition** — Decompose multi-part questions into sub-questions; run RAG per sub-question in parallel or sequentially; synthesize final answer.
- [ ] **Iterative retrieval** — Loop: generate partial answer → check if more retrieval needed → retrieve again → refine answer.

### Adaptive Retrieval
*(Deferred from Phase 4 — routing/self-reflection patterns that require a state graph.)*
- [ ] **Step back prompting** — Generate an abstract "step back" question, retrieve on both original and step-back, combine contexts. Better fit for agentic pipeline where LLM decides retrieval strategy per query. Ref: Google step back prompting; rag-from-scratch series.
- [ ] **CRAG (Corrective RAG)** — Route by retrieval confidence; fall back to LLM-only or web search when retrieval quality is low. Natural fit for LangGraph conditional edges. Ref: CRAG paper; rag-from-scratch 16.
- [ ] **Self-RAG** — Adaptive retrieval with self-reflection: decide whether to retrieve, critique retrieved docs and own answer, iterate. High factuality ceiling; high implementation complexity. Requires state graph for the reflection loop. Ref: Self-RAG paper; rag-from-scratch 17.

### Conversation History
- [ ] **Conversation state** — Design state schema; implement context tracking across turns; handle follow-up questions.

### LangGraph Migration
- [ ] **`StateGraph` rewrite** — Wrap RAG pipeline in a LangGraph `StateGraph`; add conditional edges for routing and loops.
- [ ] **State persistence** — Add checkpointer for multi-step reasoning and conversation history.
- [ ] **LangChain Studio integration** — Configure `langgraph.json`; test visualization and debugging in Studio; document setup (`studio/README.md`).

### Future / If Needed
- **Horizontal scaling, load balancing, distributed vector store** — when serving many concurrent users
- **A/B testing, feature flags** — when running experiments in production
- **Multi-lingual support, query intent detection** — when user base requires it

---

## Framework Evaluation

### Phase 1: LangChain + Plain Python RAG ✅ (current)
- PDF processing, chunking, embeddings, vector store with Chroma
- RAG pipeline via plain Python `run_rag()` with `@traceable` (2-step: retrieve → generate)
- LangSmith observability via `@traceable` decorator
- MCP tool integration

### Phase 2: Complete ✅
- Plain Python `run_rag()` replaces LCEL; `ChatPromptTemplate` stays; `RunnablePassthrough`/`RunnableLambda`/`StrOutputParser` removed

### Phase 3: Advanced Configuration & Deployment ✅
- Embedding and LLM configurable (env/config; swap providers, tune chunk settings)
- MCP resources (`pinrag://documents`), MCP prompts (`use_pinrag`)
- Retriever abstraction: `store.as_retriever()`, `run_rag()` accepts optional retriever
- Error handling: corrupted PDFs (user-facing ValueError), zero-retrieval and LLM failure (graceful degradation)
- Testing: integration test for multiple PDFs with document_id/tag filters

### Phase 4: Advanced Retrieval & Evaluation
- Evaluation framework: golden dataset, 7 evaluators, LangSmith experiments
- Baselines: Anthropic 73.3%, OpenAI k=5 53.3%, OpenAI k=10 60.0%
- Re-ranking done (Cohere, 30/30 golden, 10/10 hard-10)
- Next (priority order): multi-query, query preprocessing, prompt iteration, chunk tuning; hybrid search deferred to Phase 6

### Phase 5: Polish & Production Ready
- Deployment, monitoring, security, scalability, documentation
- Deferred retrieval polish: RAG Fusion, long-context ordering, lenient grader
- Optional QoL: CLI tags, recursive dir indexing, top_k/MMR, health check (if CLI/HTTP added)

### Phase 6: Document Intelligence & Advanced Indexing
- OCR, richer PDF structure, advanced chunking (parent-child, RAPTOR), document name resolution, query structuring
- Hybrid search (BM25 + EnsembleRetriever) — high impact for deep-in-book exact-term queries
- Deferred retrieval: HyDE, ColBERT (revisit if Phase 4 techniques leave a gap)

### Phase 7: Agentic Reasoning & LangGraph
- Routing, query classification, multi-step reasoning, conversation history, LangGraph `StateGraph` migration, Studio integration
- Deferred adaptive retrieval: step-back prompting, CRAG, Self-RAG (require state graph)

**Note:** LangChain docs recommend plain Python + `@traceable` for a deterministic 2-step RAG. `AgentMiddleware` + `create_agent` is for agentic RAG (LLM decides when to retrieve). LangGraph `StateGraph` is the right target if the pipeline needs routing, loops, or Studio visibility — deferred to Phase 7.
