"""Plain-text bodies for MCP resources (documents list, server config)."""

from __future__ import annotations

import os

from pinrag import __version__ as pinrag_version
from pinrag import config
from pinrag.core import list_documents


def _env_set(name: str) -> bool:
    """Return True if the environment variable is set and non-empty."""
    v = os.environ.get(name)
    return v is not None and str(v).strip() != ""


def format_documents_list() -> str:
    """Sync helper: fetch and format documents list for documents_resource."""
    result = list_documents(
        persist_dir=config.get_persist_dir(),
        collection=config.get_collection_name(),
    )
    docs = result.get("documents", [])
    total = result.get("total_chunks", 0)
    details = result.get("document_details") or {}
    if not docs:
        return "No documents indexed."

    def _sort_key(doc_id: str) -> tuple[str, str]:
        info = details.get(doc_id, {}) or {}
        if info.get("document_type") == "youtube" and info.get("title"):
            primary = str(info["title"]).casefold()
        else:
            primary = str(doc_id).casefold()
        return (primary, str(doc_id).casefold())

    lines = [f"Indexed documents ({total} chunks total):", ""]
    for d in sorted(docs, key=_sort_key):
        info = details.get(d, {})
        extra: list[str] = []
        if info.get("pages") is not None:
            extra.append(f"{info['pages']} pages")
        if info.get("messages") is not None:
            extra.append(f"{info['messages']} messages")
        if info.get("segments") is not None:
            extra.append(f"{info['segments']} segments")
        if info.get("bytes") is not None and "messages" not in info:
            b = info["bytes"]
            size = f"{b / 1024:.1f} KB" if b >= 1024 else f"{b} B"
            extra.append(size)
        if info.get("tag"):
            extra.append(f"tag: {info['tag']}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        # For YouTube with title, show title prominently with video ID
        if info.get("document_type") == "youtube" and info.get("title"):
            display_name = f"{info['title']} ({d})"
        else:
            display_name = d
        lines.append(f"  - {display_name}{suffix}")
    return "\n".join(lines)


def format_server_config() -> str:
    """Build config string for server_config_resource.

    Runtime-only contract (OSS + Cloud):
    - If env var exists in os.environ (non-empty): explicitly set
    - Otherwise: default
    - Always show effective value from config getters.
    """
    config_items = [
        ("PINRAG_PERSIST_DIR", config.get_persist_dir),
        ("PINRAG_COLLECTION_NAME", config.get_collection_name),
        ("PINRAG_LLM_PROVIDER", config.get_llm_provider),
        ("PINRAG_LLM_MODEL", config.get_llm_model),
        (
            "PINRAG_OPENROUTER_MODEL_FALLBACKS",
            lambda: ",".join(config.get_llm_model_fallbacks() or []),
        ),
        ("PINRAG_OPENROUTER_SORT", lambda: config.get_openrouter_sort() or ""),
        (
            "PINRAG_OPENROUTER_PROVIDER_ORDER",
            lambda: ",".join(config.get_openrouter_provider_order() or []),
        ),
        ("PINRAG_EVALUATOR_PROVIDER", config.get_evaluator_provider),
        (
            "PINRAG_EVALUATOR_MODEL",
            lambda: config.get_evaluator_model(context_heavy=False),
        ),
        (
            "PINRAG_EVALUATOR_MODEL_CONTEXT",
            lambda: config.get_evaluator_model(context_heavy=True),
        ),
        ("PINRAG_EMBEDDING_MODEL", config.get_embedding_model_name),
        ("PINRAG_CHUNK_SIZE", lambda: str(config.get_chunk_size())),
        ("PINRAG_CHUNK_OVERLAP", lambda: str(config.get_chunk_overlap())),
        (
            "PINRAG_STRUCTURE_AWARE_CHUNKING",
            lambda: str(config.get_structure_aware_chunking()),
        ),
        ("PINRAG_RETRIEVE_K", lambda: str(config.get_retrieve_k())),
        ("PINRAG_USE_RERANK", lambda: str(config.get_use_rerank()).lower()),
        ("PINRAG_RERANK_RETRIEVE_K", lambda: str(config.get_rerank_retrieve_k())),
        ("PINRAG_RERANK_TOP_N", lambda: str(config.get_rerank_top_n())),
        ("PINRAG_USE_MULTI_QUERY", lambda: str(config.get_use_multi_query()).lower()),
        ("PINRAG_MULTI_QUERY_COUNT", lambda: str(config.get_multi_query_count())),
        ("PINRAG_USE_PARENT_CHILD", lambda: str(config.get_use_parent_child()).lower()),
        ("PINRAG_PARENT_CHUNK_SIZE", lambda: str(config.get_parent_chunk_size())),
        ("PINRAG_CHILD_CHUNK_SIZE", lambda: str(config.get_child_chunk_size())),
        ("PINRAG_RESPONSE_STYLE", config.get_response_style),
        ("PINRAG_VERBOSE_LOGGING", lambda: str(config.get_verbose_logging()).lower()),
        (
            "PINRAG_GITHUB_MAX_FILE_BYTES",
            lambda: str(config.get_github_max_file_bytes()),
        ),
        ("PINRAG_GITHUB_DEFAULT_BRANCH", config.get_github_default_branch),
        (
            "PINRAG_PLAINTEXT_MAX_FILE_BYTES",
            lambda: str(config.get_plaintext_max_file_bytes()),
        ),
        (
            "PINRAG_YT_VISION_ENABLED",
            lambda: str(config.get_yt_vision_enabled()).lower(),
        ),
        ("PINRAG_YT_VISION_PROVIDER", config.get_vision_provider),
        ("PINRAG_YT_VISION_MODEL", config.get_vision_model),
        (
            "PINRAG_YT_VISION_MAX_FRAMES",
            lambda: str(config.get_yt_vision_max_frames()),
        ),
        (
            "PINRAG_YT_VISION_MIN_SCENE_SCORE",
            lambda: str(config.get_yt_vision_min_scene_score()),
        ),
        ("PINRAG_YT_VISION_IMAGE_DETAIL", config.get_yt_vision_image_detail),
    ]
    set_items: list[str] = []
    default_items: list[str] = []
    for var, getter in config_items:
        val = getter()
        line = f"  {var}: {val}"
        if _env_set(var):
            set_items.append(line)
        else:
            default_items.append(line)

    lines = [
        "PinRAG MCP Server Configuration",
        "=" * 40,
        "",
        f"PINRAG_VERSION: {pinrag_version}",
        "",
        "--- Explicitly set (runtime env) ---",
        *set_items,
        "",
        "--- Defaults (not set in env) ---",
        *default_items,
        "",
        "--- API keys (status only) ---",
        f"  OPENROUTER_API_KEY: {'set' if _env_set('OPENROUTER_API_KEY') else 'not set'}",
        f"  OPENAI_API_KEY: {'set' if _env_set('OPENAI_API_KEY') else 'not set'}",
        f"  CEREBRAS_API_KEY: {'set' if _env_set('CEREBRAS_API_KEY') else 'not set'}",
        f"  ANTHROPIC_API_KEY: {'set' if _env_set('ANTHROPIC_API_KEY') else 'not set'}",
        f"  GITHUB_TOKEN: {'set' if _env_set('GITHUB_TOKEN') else 'not set'}",
        f"  PINRAG_YT_PROXY_HTTP_URL: {'set' if _env_set('PINRAG_YT_PROXY_HTTP_URL') else 'not set'}",
        f"  PINRAG_YT_PROXY_HTTPS_URL: {'set' if _env_set('PINRAG_YT_PROXY_HTTPS_URL') else 'not set'}",
        "",
        "--- Optional: OpenRouter attribution & OpenAI / Cerebras client ---",
        f"  OPENROUTER_APP_URL (effective): {config.get_openrouter_app_url()}",
        f"  OPENROUTER_APP_TITLE (effective): {config.get_openrouter_app_title()}",
        f"  OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL') or 'not set'}",
        f"  PINRAG_CEREBRAS_BASE_URL (effective): {config.get_cerebras_base_url()}",
        "",
        "--- LangSmith observability ---",
        f"  LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING', 'not set')}",
        f"  LANGSMITH_ENDPOINT: {os.environ.get('LANGSMITH_ENDPOINT', 'not set')}",
        f"  LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT', 'not set')}",
        f"  LANGSMITH_API_KEY: {'set' if _env_set('LANGSMITH_API_KEY') else 'not set'}",
    ]
    return "\n".join(lines)
