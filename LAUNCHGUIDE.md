# PinRAG

## Tagline
MCP RAG server: index PDFs, GitHub Repos, YouTube, Discord, Plain Text, Web Docs; query with citations.

## Description
PinRAG is a retrieval-augmented generation (RAG) MCP server built with LangChain and Chroma. Index PDFs, plain text, Discord exports, YouTube transcripts (with optional on-screen vision enrichment), GitHub repositories, and web documentation sites, then ask questions in Cursor, VS Code (Copilot), or any MCP-capable client. Answers include citations (pages, timestamps, paths). Install via PyPI (`pipx`, `uv tool`, or `uvx --refresh pinrag`). Configure API keys in your editor’s MCP `env` block. For YouTube vision, see the main README (`pinrag[vision]`, ffmpeg, `PINRAG_YT_VISION_*`).

## Setup Requirements

- `OPENAI_API_KEY` (required when using OpenAI for the LLM): Default chat model is OpenAI; embeddings are local (Nomic) and need no key. https://platform.openai.com/api-keys
- `PINRAG_PERSIST_DIR` (optional): Absolute path to the Chroma data directory; defaults to `chroma_db` relative to the server process if unset.

Use `env` keys in this order in `mcp.json` (example):

```json
{
  "OPENAI_API_KEY": "your-openai-api-key-here",
  "PINRAG_PERSIST_DIR": "your-pinrag-persist-dir-here"
}
```

### MCP client block (`mcp.json`)

PinRAG is started with **`uvx --refresh pinrag`** (or the same idea: `"command": "uvx"`, `"args": ["--refresh", "pinrag"]`). Add something like this under `mcpServers` (the object key can be a slug such as `io-github-ndjordjevic-pinrag` in Cursor):

```json
"io-github-ndjordjevic-pinrag": {
  "command": "uvx",
  "args": ["--refresh", "pinrag"],
  "env": {
    "OPENAI_API_KEY": "your-openai-api-key-here",
    "PINRAG_PERSIST_DIR": "your-pinrag-persist-dir-here"
  }
}
```

If your IDE’s MCP UI cannot find `uvx` on `PATH`, set `command` to the full path from `which uvx` (e.g. `/opt/homebrew/bin/uvx` on Apple Silicon Homebrew).

On startup, `pinrag` validates API keys for the active embedding and LLM providers (`require_api_keys_for_server`). Additional providers and options are documented in the README.

## Category
Developer Tools

## Use Cases
Coding assistants, documentation Q&A, research, onboarding, technical support, content analysis

## Features
- Index PDFs, directories, plain text, Discord exports, YouTube (URL/playlist/ID), GitHub repos, web documentation sites
- Query with source citations and optional metadata filters (tags, doc type, page range, paths)
- MCP tools: add documents, add URL, query, list, remove
- MCP resources and `use_pinrag` prompt for guided workflows
- Configurable LLM and embeddings (OpenAI defaults; other providers in README)

## Getting Started
- "Index the PDFs in ./docs and summarize the deployment steps."
- "What does this repo say about authentication?" (after indexing a GitHub URL)
- Tool: `query_tool` — Ask questions over indexed documents with citations
- Tool: `add_document_tool` — Index local files, directories, or URLs (PDF, YouTube, GitHub, Discord export, web documentation sites, etc.)
- Tool: `list_documents_tool` — See what is indexed
- Tool: `remove_document_tool` — Remove a document by ID from `list_documents_tool`

## Tags
rag, mcp, langchain, chromadb, pdf, github, youtube, discord, web, scraping, embeddings, openai, cursor, vscode, retrieval, citations

## Documentation URL
https://github.com/ndjordjevic/pinrag#readme

## Health Check URL
Not applicable — stdio MCP (local `uvx` / `pipx`); no hosted HTTP endpoint.
