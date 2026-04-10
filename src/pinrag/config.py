"""Configuration from environment variables."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pinrag.chunking.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

if TYPE_CHECKING:
    from youtube_transcript_api.proxies import GenericProxyConfig

# --- Storage (Chroma) ---
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "pinrag"

# --- LLM (PINRAG_LLM_*) ---
DEFAULT_LLM_PROVIDER = "openrouter"
DEFAULT_LLM_MODEL_OPENROUTER = "openrouter/free"
DEFAULT_LLM_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_LLM_MODEL_ANTHROPIC = "claude-haiku-4-5"
# Default Cerebras model id (see https://inference-docs.cerebras.ai/models/overview).
# Prefer llama3.1-8b over gpt-oss-120b for typical RAG: production tier, fast, and less
# constrained on the free tier than gpt-oss-120b / zai-glm-4.7 per Cerebras docs.
DEFAULT_LLM_MODEL_CEREBRAS = "llama3.1-8b"

# Cerebras Inference OpenAI-compatible API (overridable via PINRAG_CEREBRAS_BASE_URL).
DEFAULT_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# OpenRouter HTTP-Referer / X-Title attribution (overridable via OPENROUTER_APP_* env).
DEFAULT_OPENROUTER_APP_URL = "https://github.com/ndjordjevic/pinrag"
DEFAULT_OPENROUTER_APP_TITLE = "PinRAG"

# --- Evaluator / LLM-as-judge (PINRAG_EVALUATOR_*) ---
DEFAULT_EVALUATOR_PROVIDER = "openai"
DEFAULT_EVALUATOR_MODEL_OPENAI = "gpt-4o"
DEFAULT_EVALUATOR_MODEL_OPENAI_CONTEXT = "gpt-4o-mini"
DEFAULT_EVALUATOR_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_EVALUATOR_MODEL_ANTHROPIC_CONTEXT = "claude-haiku-4-5"
DEFAULT_EVALUATOR_MODEL_OPENROUTER = "openrouter/free"
DEFAULT_EVALUATOR_MODEL_OPENROUTER_CONTEXT = "openrouter/free"

# --- Embeddings (PINRAG_EMBEDDING_*) — local Nomic only ---
DEFAULT_EMBEDDING_MODEL_LOCAL = "nomic-embed-text-v1.5"

# --- Chunking (PINRAG_CHUNK_*): numeric defaults from pinrag.chunking.splitter ---

# --- Retrieval & re-ranking ---
DEFAULT_RETRIEVE_K = 20
DEFAULT_USE_RERANK = False
DEFAULT_RERANK_RETRIEVE_K = 20
DEFAULT_RERANK_TOP_N = 10

# --- Multi-query retrieval (PINRAG_MULTI_QUERY_*) ---
DEFAULT_USE_MULTI_QUERY = False
DEFAULT_MULTI_QUERY_COUNT = 4
DEFAULT_MULTI_QUERY_MAX = 10

# --- Parent-child retrieval (PINRAG_PARENT_*, PINRAG_CHILD_*) ---
DEFAULT_USE_PARENT_CHILD = False
DEFAULT_PARENT_CHUNK_SIZE = 2000
DEFAULT_CHILD_CHUNK_SIZE = 800

# --- RAG response & structure-aware chunking ---
DEFAULT_RESPONSE_STYLE = "thorough"
DEFAULT_STRUCTURE_AWARE_CHUNKING = True
DEFAULT_VERBOSE_LOGGING = False

# --- YouTube vision enrichment (optional, opt-in) ---
DEFAULT_YT_VISION_ENABLED = False
DEFAULT_VISION_PROVIDER = "openai"
DEFAULT_VISION_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_VISION_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_VISION_MODEL_OPENROUTER = "google/gemini-2.5-flash"
DEFAULT_YT_VISION_MAX_FRAMES = 8
DEFAULT_YT_VISION_MIN_SCENE_SCORE = 27.0
DEFAULT_YT_VISION_IMAGE_DETAIL = "low"

# --- Indexing limits (GitHub, plain text files) ---
DEFAULT_MAX_INDEX_FILE_BYTES = 524288  # 512 KB
DEFAULT_GITHUB_MAX_FILE_BYTES = DEFAULT_MAX_INDEX_FILE_BYTES
DEFAULT_GITHUB_DEFAULT_BRANCH = "main"
DEFAULT_PLAINTEXT_MAX_FILE_BYTES = DEFAULT_MAX_INDEX_FILE_BYTES

# --- Web docs indexing ---
DEFAULT_WEB_MAX_PAGES = 200
DEFAULT_WEB_MAX_DEPTH = 5
DEFAULT_WEB_MAX_PAGE_BYTES = 1_048_576  # 1 MiB
DEFAULT_WEB_REQUEST_TIMEOUT = 20.0
DEFAULT_WEB_CONCURRENCY = 4
DEFAULT_WEB_RATE_LIMIT_PER_HOST = 2.0
DEFAULT_WEB_USER_AGENT = "pinrag/0.10 (+https://github.com/ndjordjevic/pinrag)"
DEFAULT_WEB_RESPECT_ROBOTS = True
DEFAULT_WEB_PREFER_LLMS_TXT = True


# --- LLM ---
def get_llm_provider() -> str:
    """Return LLM provider from PINRAG_LLM_PROVIDER env (openai | anthropic | openrouter | cerebras)."""
    val = os.environ.get("PINRAG_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    p = (val or "").strip().lower()
    if p in ("openai", "anthropic", "openrouter", "cerebras"):
        return p
    return DEFAULT_LLM_PROVIDER


def get_cerebras_base_url() -> str:
    """Return Cerebras API base URL for OpenAI-compatible clients (trailing slash stripped)."""
    val = os.environ.get("PINRAG_CEREBRAS_BASE_URL")
    if val is not None and str(val).strip():
        return str(val).strip().rstrip("/")
    return DEFAULT_CEREBRAS_BASE_URL.rstrip("/")


def get_llm_model() -> str:
    """Return LLM model name from PINRAG_LLM_MODEL env, or provider default."""
    val = os.environ.get("PINRAG_LLM_MODEL")
    if val and str(val).strip():
        return str(val).strip()
    provider = get_llm_provider()
    if provider == "anthropic":
        return DEFAULT_LLM_MODEL_ANTHROPIC
    if provider == "openrouter":
        return DEFAULT_LLM_MODEL_OPENROUTER
    if provider == "cerebras":
        return DEFAULT_LLM_MODEL_CEREBRAS
    return DEFAULT_LLM_MODEL_OPENAI


def get_openrouter_app_url() -> str | None:
    """Return OpenRouter ``HTTP-Referer`` attribution URL (env or PinRAG default)."""
    val = os.environ.get("OPENROUTER_APP_URL")
    if val is not None and str(val).strip():
        return str(val).strip()
    return DEFAULT_OPENROUTER_APP_URL


def get_openrouter_app_title() -> str | None:
    """Return OpenRouter app title (``X-Title``) for attribution (env or PinRAG default)."""
    val = os.environ.get("OPENROUTER_APP_TITLE")
    if val is not None and str(val).strip():
        return str(val).strip()
    return DEFAULT_OPENROUTER_APP_TITLE


def sync_openrouter_sdk_attribution_env() -> None:
    """Copy PinRAG-resolved ``OPENROUTER_APP_*`` into ``OPENROUTER_HTTP_REFERER`` /
    ``OPENROUTER_X_OPEN_ROUTER_TITLE`` for code that talks to OpenRouter outside
    ``langchain-openrouter`` (that stack uses ``app_url`` / ``app_title`` on
    ``ChatOpenRouter`` directly, with PinRAG passing :func:`get_openrouter_app_url`
    / :func:`get_openrouter_app_title`).
    """
    url = get_openrouter_app_url()
    title = get_openrouter_app_title()
    if url:
        os.environ["OPENROUTER_HTTP_REFERER"] = url
    else:
        os.environ.pop("OPENROUTER_HTTP_REFERER", None)
    if title:
        os.environ["OPENROUTER_X_OPEN_ROUTER_TITLE"] = title
    else:
        os.environ.pop("OPENROUTER_X_OPEN_ROUTER_TITLE", None)


def get_llm_model_fallbacks() -> list[str] | None:
    """Return OpenRouter fallback model IDs from ``PINRAG_OPENROUTER_MODEL_FALLBACKS`` (comma-separated).

    Passed to OpenRouter as the ``models`` request field so the gateway tries the next
    model when the primary (``PINRAG_LLM_MODEL``) fails (rate limits, downtime, etc.).
    Only used when ``PINRAG_LLM_PROVIDER=openrouter``.

    Legacy: if unset or empty, ``PINRAG_LLM_MODEL_FALLBACKS`` is read (same format).
    """
    val = os.environ.get("PINRAG_OPENROUTER_MODEL_FALLBACKS")
    if val is None or not str(val).strip():
        val = os.environ.get("PINRAG_LLM_MODEL_FALLBACKS")
    if val is None or not str(val).strip():
        return None
    out = [p.strip() for p in str(val).split(",") if p.strip()]
    return out or None


def get_openrouter_sort() -> str | None:
    """Return OpenRouter ``provider.sort`` from ``PINRAG_OPENROUTER_SORT`` if set.

    Valid values: ``price``, ``throughput``, ``latency``. Invalid or empty env returns
    ``None`` (OpenRouter default routing). Only used when ``PINRAG_LLM_PROVIDER=openrouter``.
    """
    val = (os.environ.get("PINRAG_OPENROUTER_SORT") or "").strip().lower()
    return val if val in ("price", "throughput", "latency") else None


def get_openrouter_provider_order() -> list[str] | None:
    """Return OpenRouter ``provider.order`` from ``PINRAG_OPENROUTER_PROVIDER_ORDER`` (comma-separated).

    Provider slugs are tried in sequence before OpenRouter's default selection. For example
    ``PINRAG_OPENROUTER_PROVIDER_ORDER=Cerebras`` with ``PINRAG_LLM_MODEL=openai/gpt-oss-120b``
    prefers Cerebras-backed inference on OpenRouter. Use exact labels from the model's
    provider list on OpenRouter if a name fails validation.

    See https://openrouter.ai/docs/guides/routing/provider-selection. Only used when
    ``PINRAG_LLM_PROVIDER=openrouter``.
    """
    val = os.environ.get("PINRAG_OPENROUTER_PROVIDER_ORDER")
    if val is None or not str(val).strip():
        return None
    out = [p.strip() for p in str(val).split(",") if p.strip()]
    return out or None


# --- Evaluator ---
def get_evaluator_provider() -> str:
    """Return evaluator (LLM-as-judge) provider from PINRAG_EVALUATOR_PROVIDER env (openai | anthropic | openrouter)."""
    val = os.environ.get("PINRAG_EVALUATOR_PROVIDER", DEFAULT_EVALUATOR_PROVIDER)
    p = (val or "").strip().lower()
    if p in ("openai", "anthropic", "openrouter"):
        return p
    return DEFAULT_EVALUATOR_PROVIDER


def get_evaluator_model(*, context_heavy: bool = False) -> str:
    """Return evaluator model for correctness (context_heavy=False) or groundedness (context_heavy=True)."""
    val = os.environ.get(
        "PINRAG_EVALUATOR_MODEL_CONTEXT" if context_heavy else "PINRAG_EVALUATOR_MODEL"
    )
    if val and str(val).strip():
        return str(val).strip()
    provider = get_evaluator_provider()
    if provider == "anthropic":
        return (
            DEFAULT_EVALUATOR_MODEL_ANTHROPIC_CONTEXT
            if context_heavy
            else DEFAULT_EVALUATOR_MODEL_ANTHROPIC
        )
    if provider == "openrouter":
        return (
            DEFAULT_EVALUATOR_MODEL_OPENROUTER_CONTEXT
            if context_heavy
            else DEFAULT_EVALUATOR_MODEL_OPENROUTER
        )
    return (
        DEFAULT_EVALUATOR_MODEL_OPENAI_CONTEXT
        if context_heavy
        else DEFAULT_EVALUATOR_MODEL_OPENAI
    )


# --- Embeddings ---
def get_embedding_model_name() -> str:
    """Return Nomic embedding model id from PINRAG_EMBEDDING_MODEL env, or default local model."""
    val = os.environ.get("PINRAG_EMBEDDING_MODEL")
    if val and str(val).strip():
        return str(val).strip()
    return DEFAULT_EMBEDDING_MODEL_LOCAL


# --- Chroma storage ---
def get_persist_dir() -> str:
    """Return Chroma persist directory from PINRAG_PERSIST_DIR env var, or chroma_db (project-local)."""
    raw = os.environ.get("PINRAG_PERSIST_DIR")
    if raw is None:
        return DEFAULT_PERSIST_DIR
    s = str(raw).strip()
    return s if s else DEFAULT_PERSIST_DIR


def get_collection_name() -> str:
    """Return Chroma collection name.

    If PINRAG_COLLECTION_NAME is set, use it. Otherwise return DEFAULT_COLLECTION_NAME.
    Use a single collection per persist dir; if you switch embedding models,
    re-index or use a separate PINRAG_PERSIST_DIR / PINRAG_COLLECTION_NAME
    to avoid dimension mismatches.
    """
    env_name = os.environ.get("PINRAG_COLLECTION_NAME")
    if env_name and str(env_name).strip():
        return str(env_name).strip()
    return DEFAULT_COLLECTION_NAME


# --- Chunking ---
def get_chunk_size() -> int:
    """Return chunk size from PINRAG_CHUNK_SIZE env var, or default (1000)."""
    val = os.environ.get("PINRAG_CHUNK_SIZE")
    if val is None:
        return DEFAULT_CHUNK_SIZE
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_CHUNK_SIZE
        return n
    except ValueError:
        return DEFAULT_CHUNK_SIZE


def get_chunk_overlap() -> int:
    """Return chunk overlap from PINRAG_CHUNK_OVERLAP env var, or default (200).

    If the requested overlap is >= chunk size, clamps to chunk_size - 1 so splitters
    always see overlap < chunk size.
    """
    val = os.environ.get("PINRAG_CHUNK_OVERLAP")
    if val is None:
        return DEFAULT_CHUNK_OVERLAP
    try:
        n = int(val)
        if n < 0:
            return DEFAULT_CHUNK_OVERLAP
        chunk_size = get_chunk_size()
        if n >= chunk_size:
            return max(0, chunk_size - 1)
        return n
    except ValueError:
        return DEFAULT_CHUNK_OVERLAP


# --- Retrieval & re-ranking ---
def get_retrieve_k() -> int:
    """Return how many chunks to retrieve (default 20). Used for both rerank on and off."""
    val = os.environ.get("PINRAG_RETRIEVE_K")
    if val is None:
        return DEFAULT_RETRIEVE_K
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_RETRIEVE_K
        return n
    except ValueError:
        return DEFAULT_RETRIEVE_K


def get_use_rerank() -> bool:
    """Return whether to use FlashRank re-ranking. Requires pinrag[rerank]."""
    val = os.environ.get("PINRAG_USE_RERANK")
    if val is None or not str(val).strip():
        return DEFAULT_USE_RERANK
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return DEFAULT_USE_RERANK


def get_rerank_retrieve_k() -> int:
    """Return how many chunks the base retriever fetches before reranking. Falls back to PINRAG_RETRIEVE_K when unset."""
    val = os.environ.get("PINRAG_RERANK_RETRIEVE_K")
    if val is None:
        return get_retrieve_k()
    try:
        n = int(val)
        if n < 1:
            return get_retrieve_k()
        return n
    except ValueError:
        return get_retrieve_k()


def get_rerank_top_n() -> int:
    """Return how many chunks the reranker returns to the LLM (default 10).

    Capped by rerank retrieve k so top_n does not exceed the retrieved pool.
    """
    val = os.environ.get("PINRAG_RERANK_TOP_N")
    cap = get_rerank_retrieve_k()
    if val is None:
        return min(DEFAULT_RERANK_TOP_N, cap)
    try:
        n = int(val)
        if n < 1:
            return min(DEFAULT_RERANK_TOP_N, cap)
        return min(n, cap)
    except ValueError:
        return min(DEFAULT_RERANK_TOP_N, cap)


# --- Multi-query retrieval ---
def get_use_multi_query() -> bool:
    """Return whether to use multi-query retrieval (merged variants; see PINRAG_MULTI_QUERY_COUNT)."""
    val = os.environ.get("PINRAG_USE_MULTI_QUERY")
    if val is None or not str(val).strip():
        return DEFAULT_USE_MULTI_QUERY
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return DEFAULT_USE_MULTI_QUERY


def get_multi_query_count() -> int:
    """Return number of alternative queries for multi-query retrieval (default 4, max 10)."""
    val = os.environ.get("PINRAG_MULTI_QUERY_COUNT")
    if val is None:
        return DEFAULT_MULTI_QUERY_COUNT
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_MULTI_QUERY_COUNT
        if n > DEFAULT_MULTI_QUERY_MAX:
            return DEFAULT_MULTI_QUERY_MAX
        return n
    except ValueError:
        return DEFAULT_MULTI_QUERY_COUNT


# --- Parent-child retrieval ---
def get_use_parent_child() -> bool:
    """Return whether to use parent-child retrieval (embed small, return large chunks)."""
    val = os.environ.get("PINRAG_USE_PARENT_CHILD")
    if val is None or not str(val).strip():
        return DEFAULT_USE_PARENT_CHILD
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return DEFAULT_USE_PARENT_CHILD


def get_parent_chunk_size() -> int:
    """Return parent chunk size in chars for parent-child retrieval (default 2000)."""
    val = os.environ.get("PINRAG_PARENT_CHUNK_SIZE")
    if val is None:
        return DEFAULT_PARENT_CHUNK_SIZE
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_PARENT_CHUNK_SIZE
        return n
    except ValueError:
        return DEFAULT_PARENT_CHUNK_SIZE


def get_child_chunk_size() -> int:
    """Return child chunk size in chars for parent-child retrieval (default 800).

    Never larger than the parent chunk size so parent-child indexing stays consistent.
    """
    parent = get_parent_chunk_size()
    val = os.environ.get("PINRAG_CHILD_CHUNK_SIZE")
    if val is None:
        return min(DEFAULT_CHILD_CHUNK_SIZE, parent)
    try:
        n = int(val)
        if n < 1:
            return min(DEFAULT_CHILD_CHUNK_SIZE, parent)
        return min(n, parent)
    except ValueError:
        return min(DEFAULT_CHILD_CHUNK_SIZE, parent)


# --- RAG response & chunking UX ---
def get_response_style() -> str:
    """Return RAG response style from PINRAG_RESPONSE_STYLE (thorough | concise)."""
    val = os.environ.get("PINRAG_RESPONSE_STYLE")
    if val is None or not str(val).strip():
        return DEFAULT_RESPONSE_STYLE
    v = str(val).strip().lower()
    if v in ("thorough", "concise"):
        return v
    return DEFAULT_RESPONSE_STYLE


def get_structure_aware_chunking() -> bool:
    """Return whether to apply structure-aware chunking heuristics (default: true)."""
    val = os.environ.get("PINRAG_STRUCTURE_AWARE_CHUNKING")
    if val is None or not str(val).strip():
        return DEFAULT_STRUCTURE_AWARE_CHUNKING
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return DEFAULT_STRUCTURE_AWARE_CHUNKING


def get_verbose_logging() -> bool:
    """Return whether verbose MCP notifications are enabled (default: false)."""
    val = os.environ.get("PINRAG_VERBOSE_LOGGING")
    if val is None or not str(val).strip():
        return DEFAULT_VERBOSE_LOGGING
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return DEFAULT_VERBOSE_LOGGING


# --- GitHub indexing ---
def get_github_token() -> str | None:
    """Return GitHub personal access token from GITHUB_TOKEN env (optional)."""
    val = os.environ.get("GITHUB_TOKEN")
    if val and str(val).strip():
        return str(val).strip()
    return None


def get_github_max_file_bytes() -> int:
    """Return max file size in bytes for GitHub indexing (default 512 KB)."""
    val = os.environ.get("PINRAG_GITHUB_MAX_FILE_BYTES")
    if val is None:
        return DEFAULT_GITHUB_MAX_FILE_BYTES
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_GITHUB_MAX_FILE_BYTES
        return n
    except ValueError:
        return DEFAULT_GITHUB_MAX_FILE_BYTES


def get_github_default_branch() -> str:
    """Return default branch for GitHub URLs when not specified (default main)."""
    val = os.environ.get("PINRAG_GITHUB_DEFAULT_BRANCH")
    if val and str(val).strip():
        return str(val).strip()
    return DEFAULT_GITHUB_DEFAULT_BRANCH


# --- Web docs indexing ---
def _get_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or not str(val).strip():
        return default
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _get_int_env(name: str, default: int, *, min_value: int = 1) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        n = int(val)
        if n < min_value:
            return default
        return n
    except ValueError:
        return default


def _get_float_env(name: str, default: float, *, min_value: float = 0.0) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        n = float(val)
        if n <= min_value:
            return default
        return n
    except ValueError:
        return default


def get_web_max_pages() -> int:
    """Max pages indexed from a single web docs seed (default 200)."""
    return _get_int_env("PINRAG_WEB_MAX_PAGES", DEFAULT_WEB_MAX_PAGES)


def get_web_max_depth() -> int:
    """Max BFS depth from a seed URL when sitemap/llms.txt unavailable (default 5)."""
    return _get_int_env("PINRAG_WEB_MAX_DEPTH", DEFAULT_WEB_MAX_DEPTH)


def get_web_max_page_bytes() -> int:
    """Max bytes fetched per page (default 1 MiB)."""
    return _get_int_env("PINRAG_WEB_MAX_PAGE_BYTES", DEFAULT_WEB_MAX_PAGE_BYTES)


def get_web_request_timeout() -> float:
    """HTTP timeout per request in seconds (default 20)."""
    return _get_float_env("PINRAG_WEB_REQUEST_TIMEOUT", DEFAULT_WEB_REQUEST_TIMEOUT)


def get_web_concurrency() -> int:
    """Number of concurrent page fetches (default 4)."""
    return _get_int_env("PINRAG_WEB_CONCURRENCY", DEFAULT_WEB_CONCURRENCY)


def get_web_rate_limit_per_host() -> float:
    """Max requests per second per host (default 2.0)."""
    return _get_float_env(
        "PINRAG_WEB_RATE_LIMIT_PER_HOST", DEFAULT_WEB_RATE_LIMIT_PER_HOST
    )


def get_web_user_agent() -> str:
    """User-Agent string for web docs fetches."""
    val = os.environ.get("PINRAG_WEB_USER_AGENT")
    if val and str(val).strip():
        return str(val).strip()
    return DEFAULT_WEB_USER_AGENT


def get_web_respect_robots() -> bool:
    """Whether to respect robots.txt (default True). Disable only for whitelisted use."""
    return _get_bool_env("PINRAG_WEB_RESPECT_ROBOTS", DEFAULT_WEB_RESPECT_ROBOTS)


def get_web_prefer_llms_txt() -> bool:
    """Whether to try ``/llms.txt`` discovery before ``sitemap.xml`` (default True)."""
    return _get_bool_env("PINRAG_WEB_PREFER_LLMS_TXT", DEFAULT_WEB_PREFER_LLMS_TXT)


# --- Plain text indexing ---
def get_plaintext_max_file_bytes() -> int:
    """Return max file size in bytes for plain text indexing (default 512 KB)."""
    val = os.environ.get("PINRAG_PLAINTEXT_MAX_FILE_BYTES")
    if val is None:
        return DEFAULT_PLAINTEXT_MAX_FILE_BYTES
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_PLAINTEXT_MAX_FILE_BYTES
        return n
    except ValueError:
        return DEFAULT_PLAINTEXT_MAX_FILE_BYTES


# --- YouTube transcripts ---
def get_yt_proxy_config() -> GenericProxyConfig | None:
    """Return YouTube transcript API proxy config from env, or None if not configured.

    When PINRAG_YT_PROXY_HTTP_URL or PINRAG_YT_PROXY_HTTPS_URL is set, uses
    GenericProxyConfig. Otherwise returns None (no proxy).
    """
    from youtube_transcript_api.proxies import GenericProxyConfig

    http_url = os.environ.get("PINRAG_YT_PROXY_HTTP_URL", "").strip()
    https_url = os.environ.get("PINRAG_YT_PROXY_HTTPS_URL", "").strip()
    if http_url or https_url:
        return GenericProxyConfig(
            http_url=http_url or None,
            https_url=https_url or None,
        )

    return None


def get_yt_vision_enabled() -> bool:
    """Return whether YouTube vision enrichment is enabled (default: false)."""
    val = os.environ.get("PINRAG_YT_VISION_ENABLED")
    if val is None or not str(val).strip():
        return DEFAULT_YT_VISION_ENABLED
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return DEFAULT_YT_VISION_ENABLED


def get_vision_provider() -> str:
    """Return YouTube vision provider from PINRAG_YT_VISION_PROVIDER (openai | anthropic | openrouter).

    Legacy ``PINRAG_VISION_PROVIDER`` is still read if the YT-prefixed variable is unset.
    """
    val = os.environ.get("PINRAG_YT_VISION_PROVIDER") or os.environ.get(
        "PINRAG_VISION_PROVIDER", DEFAULT_VISION_PROVIDER
    )
    p = (val or "").strip().lower()
    if p in ("openai", "anthropic", "openrouter"):
        return p
    return DEFAULT_VISION_PROVIDER


def get_vision_model() -> str:
    """Return YouTube vision model from PINRAG_YT_VISION_MODEL, or provider default.

    Legacy ``PINRAG_VISION_MODEL`` is still read if the YT-prefixed variable is unset.
    """
    val = os.environ.get("PINRAG_YT_VISION_MODEL") or os.environ.get("PINRAG_VISION_MODEL")
    if val and str(val).strip():
        return str(val).strip()
    provider = get_vision_provider()
    if provider == "anthropic":
        return DEFAULT_VISION_MODEL_ANTHROPIC
    if provider == "openrouter":
        return DEFAULT_VISION_MODEL_OPENROUTER
    return DEFAULT_VISION_MODEL_OPENAI


def get_yt_vision_max_frames() -> int:
    """Return max extracted frames per video for vision enrichment."""
    val = os.environ.get("PINRAG_YT_VISION_MAX_FRAMES")
    if val is None:
        return DEFAULT_YT_VISION_MAX_FRAMES
    try:
        n = int(val)
        if n < 1:
            return DEFAULT_YT_VISION_MAX_FRAMES
        return n
    except ValueError:
        return DEFAULT_YT_VISION_MAX_FRAMES


def get_yt_vision_min_scene_score() -> float:
    """Return min scene score for PySceneDetect AdaptiveDetector (default 27.0)."""
    val = os.environ.get("PINRAG_YT_VISION_MIN_SCENE_SCORE")
    if val is None:
        return DEFAULT_YT_VISION_MIN_SCENE_SCORE
    try:
        n = float(val)
        if n <= 0:
            return DEFAULT_YT_VISION_MIN_SCENE_SCORE
        return n
    except ValueError:
        return DEFAULT_YT_VISION_MIN_SCENE_SCORE


def get_yt_vision_image_detail() -> str:
    """OpenAI vision image detail: low | high | auto (default low).

    Use PINRAG_YT_VISION_IMAGE_DETAIL. ``high`` uses more image tokens and reads
    small on-screen code better; ignored for Anthropic (full image is sent).
    """
    val = os.environ.get("PINRAG_YT_VISION_IMAGE_DETAIL")
    if val is None or not str(val).strip():
        return DEFAULT_YT_VISION_IMAGE_DETAIL
    d = str(val).strip().lower()
    if d in ("low", "high", "auto"):
        return d
    return DEFAULT_YT_VISION_IMAGE_DETAIL
