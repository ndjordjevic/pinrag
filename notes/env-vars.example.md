# PinRAG environment variables (reference)

Set these in the MCP server `env` object in `mcp.json` (for example `~/.cursor/mcp.json` or `.vscode/mcp.json`), or export them in your shell when running repo scripts (`uv run python scripts/...`). The packaged `pinrag` entry point does not load any dotenv file.

See also the configuration table in the project `README.md`.

---

## LLM (required for generation)
# Provider: openai | anthropic | openrouter (default: openrouter)
# PINRAG_LLM_PROVIDER=openrouter
# Model name (OpenRouter slug or provider-specific id). If unset:
#   openrouter=openrouter/free (free model router), openai=gpt-4o-mini, anthropic=claude-haiku-4-5
# Premium OpenRouter example: PINRAG_LLM_MODEL=anthropic/claude-sonnet-4-6
# PINRAG_LLM_MODEL=openrouter/free
# Optional: comma-separated fallback model slugs for OpenRouter (`models` field); tried if the primary fails (use other :free slugs to stay zero-cost).
# PINRAG_OPENROUTER_MODEL_FALLBACKS=google/gemini-2.0-flash-001:free,meta-llama/llama-3.3-70b-instruct:free
# Optional provider preference: price | throughput | latency
# PINRAG_OPENROUTER_SORT=price
# Optional: comma-separated OpenRouter provider.order (pin inference backend; use exact labels from the model page)
# Example: fast gpt-oss-120b via OpenRouter — PINRAG_LLM_MODEL=openai/gpt-oss-120b + PINRAG_OPENROUTER_PROVIDER_ORDER=Cerebras
# PINRAG_OPENROUTER_PROVIDER_ORDER=Cerebras
# PINRAG_OPENROUTER_MODEL_FALLBACKS lists alternate *model* slugs if the primary fails, not duplicate primary or provider names. Legacy alias: PINRAG_LLM_MODEL_FALLBACKS.

# OpenRouter (required when PINRAG_LLM_PROVIDER=openrouter; get a key at https://openrouter.ai/)
#OPENROUTER_API_KEY=your_openrouter_api_key_here
# Optional attribution (defaults: repo URL + "PinRAG"; see README):
# OPENROUTER_APP_URL=https://your-app.example
# OPENROUTER_APP_TITLE=YourAppName

# OpenAI (required when PINRAG_LLM_PROVIDER=openai)
#OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (required when PINRAG_LLM_PROVIDER=anthropic)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# --- Embeddings (local Nomic; no API key) ---
# Model id. If unset: nomic-embed-text-v1.5 (first run downloads ~270 MB, cached locally)
# PINRAG_EMBEDDING_MODEL=nomic-embed-text-v1.5

# --- Storage & chunking ---
# Persist directory (default: chroma_db, project-local)
# PINRAG_PERSIST_DIR=chroma_db
# Chunk size in characters (default: 1000)
# PINRAG_CHUNK_SIZE=1000
# Chunk overlap in characters (default: 200)
# PINRAG_CHUNK_OVERLAP=200
# Enable structure-aware chunking heuristics for code/table boundaries (default: true)
# PINRAG_STRUCTURE_AWARE_CHUNKING=true
# Collection name (default: pinrag)
# PINRAG_COLLECTION_NAME=pinrag

# --- Retrieval ---
# Number of chunks to retrieve (default: 20)
# PINRAG_RETRIEVE_K=20

# --- Parent-child retrieval ---
# Embed small chunks for precise matching, return larger parent chunks for context (default: false). Requires re-indexing.
# PINRAG_USE_PARENT_CHILD=false
# Parent chunk size in characters (default: 2000)
# PINRAG_PARENT_CHUNK_SIZE=2000
# Child chunk size in characters (default: 800)
# PINRAG_CHILD_CHUNK_SIZE=800

# --- Re-ranking (FlashRank; requires pinrag[rerank]) ---
# Enable FlashRank re-ranking (default: false)
# PINRAG_USE_RERANK=false
# Chunks to fetch before reranking (default: same as PINRAG_RETRIEVE_K)
# PINRAG_RERANK_RETRIEVE_K=20
# Chunks to return after reranking (default: 10)
# PINRAG_RERANK_TOP_N=10

# --- Multi-query retrieval ---
# Generate query variants via LLM, merge results (default: false)
# PINRAG_USE_MULTI_QUERY=false
# Number of alternative queries (default: 4)
# PINRAG_MULTI_QUERY_COUNT=4

# --- Response style ---
# thorough (default) | concise. Used by evaluation target and MCP query default.
# PINRAG_RESPONSE_STYLE=thorough

# --- GitHub indexing ---
# Personal access token (optional for public repos; required for private repos and higher rate limits)
# GITHUB_TOKEN=ghp_...
# Max file size in bytes to index (default: 524288 = 512 KB); larger files are skipped
# PINRAG_GITHUB_MAX_FILE_BYTES=524288
# Default branch when not specified in URL (default: main)
# PINRAG_GITHUB_DEFAULT_BRANCH=main

# --- Plain text indexing ---
# Max file size in bytes to index (default: 524288 = 512 KB); larger files are skipped
# PINRAG_PLAINTEXT_MAX_FILE_BYTES=524288

# --- Web docs indexing ---
# Maximum pages fetched per web indexing run (default: 200)
# PINRAG_WEB_MAX_PAGES=200
# Maximum BFS crawl depth from the seed URL (default: 5); llms.txt / sitemap paths ignore this
# PINRAG_WEB_MAX_DEPTH=5
# Skip pages whose response body exceeds this size (default: 1048576 = 1 MiB)
# PINRAG_WEB_MAX_PAGE_BYTES=1048576
# Per-request HTTP timeout in seconds (default: 20)
# PINRAG_WEB_REQUEST_TIMEOUT=20
# Max concurrent fetches per host (default: 4)
# PINRAG_WEB_CONCURRENCY=4
# Token-bucket refill rate in requests/sec per host (default: 2.0)
# PINRAG_WEB_RATE_LIMIT_PER_HOST=2.0
# Override User-Agent header (default: PinRAGBot/<version> (+https://github.com/ndjordjevic/pinrag))
# PINRAG_WEB_USER_AGENT=PinRAGBot/0.10.0
# Honor robots.txt disallow rules when crawling (default: true)
# PINRAG_WEB_RESPECT_ROBOTS=true
# Try llms.txt / llms-full.txt before sitemap / BFS (default: true)
# PINRAG_WEB_PREFER_LLMS_TXT=true

# --- YouTube transcript proxy (work around IP bans) ---
# Generic HTTP/HTTPS proxy (e.g. Webshare Proxy Server rotating endpoint):
# PINRAG_YT_PROXY_HTTP_URL=http://user:pass@p.webshare.io:80
# PINRAG_YT_PROXY_HTTPS_URL=http://user:pass@p.webshare.io:80

# --- YouTube vision enrichment (optional; off by default) ---
# PINRAG_YT_VISION_ENABLED=true
# Vision API: openai (default) | anthropic | openrouter
#   openai / anthropic: pip install 'pinrag[vision]', ffmpeg on PATH, yt-dlp; per-frame images + OPENAI_API_KEY or ANTHROPIC_API_KEY
#   openrouter: one native video_url call per video (default model google/gemini-2.5-flash); OPENROUTER_API_KEY only; no ffmpeg/scenedetect
# PINRAG_YT_VISION_PROVIDER=openai
# PINRAG_YT_VISION_MODEL=gpt-4o
# OpenRouter example: PINRAG_YT_VISION_PROVIDER=openrouter
# PINRAG_YT_VISION_MODEL=google/gemini-2.5-flash
# PINRAG_YT_VISION_MAX_FRAMES=8
# PINRAG_YT_VISION_MIN_SCENE_SCORE=27.0
# OpenAI-only image resolution for vision (low | high | auto):
# PINRAG_YT_VISION_IMAGE_DETAIL=low

# --- MCP verbose notifications (optional; default off) ---
# When true, tool/resource calls emit detailed phase events to MCP notifications/message.
# Useful for debugging indexing internals (format routing, transcript load, vision steps, chunk upserts).
# PINRAG_VERBOSE_LOGGING=false

# --- Evaluators (LLM-as-judge; used only during evaluation runs) ---
# Provider: openai (default) | anthropic | openrouter — structured JSON output varies by model.
# PINRAG_EVALUATOR_PROVIDER=openai
# Model for correctness/relevance. If unset: openai=gpt-4o, anthropic=claude-sonnet-4-6, openrouter=openrouter/free
# PINRAG_EVALUATOR_MODEL=gpt-4o
# Model for groundedness/retrieval (context-heavy). If unset: openai=gpt-4o-mini, anthropic=claude-haiku-4-5, openrouter=openrouter/free
# PINRAG_EVALUATOR_MODEL_CONTEXT=gpt-4o-mini
# OpenRouter defaults stay free (openrouter/free) but may rotate underlying models; graders use strict json_schema.
# For stable free grading, set EVALUATOR_MODEL* to a specific slug from https://openrouter.ai/models (structured_outputs).
# When PINRAG_EVALUATOR_PROVIDER=openrouter, PINRAG_OPENROUTER_MODEL_FALLBACKS, PINRAG_OPENROUTER_SORT, and PINRAG_OPENROUTER_PROVIDER_ORDER apply to the grader client too.

# --- Optional: LangSmith tracing (see notes/langsmith-setup.md) ---
# LANGSMITH_API_KEY=your_langsmith_api_key_here
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=pinrag
# LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # For EU region only; omit for US
