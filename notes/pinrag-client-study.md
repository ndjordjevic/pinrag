# PinRAG Client — Implementation Decisions

---

## Transport Decision

**Keep stdio as default; add HTTP via `pinrag server` subcommand.**

| Invocation | Transport | Who uses it |
|---|---|---|
| `pinrag` (default) | stdio | Editors (Cursor, VS Code, Claude Code) via `mcp.json` command entry |
| `pinrag server` | streamable-http on `localhost:8765` | pinrag-cli, multi-client, future web clients |

Rationale: Dropping stdio breaks all existing editor configs, marketplace listings, and zero-config UX. HTTP is additive for the CLI use case.

---

## Repository Structure

| Repository | Contents |
|---|---|
| **`pinrag`** (existing) | Core logic + MCP server (stdio default + HTTP subcommand) |
| **`pinrag-cli`** (new) | Standalone interactive CLI client |

---

## Tech Stack

| Concern | Choice |
|---|---|
| CLI entry point (args, flags) | **Typer** or **argparse** |
| Interactive REPL loop | **prompt_toolkit** (`PromptSession`) |
| Output rendering | **Rich** (`Console`, `Markdown`, `Live`) |
| Slash commands | Convention-based dispatcher (method `cmd_foo()` becomes `/foo`) |
| Streaming display | **Rich.Live** |
| Input history | **prompt_toolkit.FileHistory** |
| Conversation history | JSON files (Phase 1) |
| MCP client | `mcp` SDK (`streamable_http_client`, `ClientSession`) |
| ASGI server | **Uvicorn** |

---

## Phase 0 — Core Extraction (pinrag repo refactor)

Extract transport-agnostic business logic out of `mcp/tools.py` into `pinrag.core` so any consumer (CLI, MCP server, scripts) can import cleanly.

- Create `src/pinrag/core/operations.py` — move `query()`, `add_file()`, `add_files()`, `list_documents()`, `remove_document()` (same signatures, same behavior)
- Create `src/pinrag/core/format_detection.py` — move format detection, path resolution, GitHub URL validation helpers
- Create `src/pinrag/core/__init__.py` — clean public API exports
- Slim `mcp/tools.py` to backward-compatible re-exports from `pinrag.core` (including legacy `_resolve_*` / `_detect_*` aliases for imports that still target `pinrag.mcp.tools`)
- No new dependencies, no behavioral changes; tests that mock internals via `patch("pinrag.mcp.tools.*")` were updated to `pinrag.core.operations` (and `pathlib` where tests used `mcp_tools.Path`)

After this phase, `pinrag-cli` (and future consumers) can `from pinrag.core import query, add_files, list_documents, remove_document`.

## Phase 1 — Minimal Viable CLI

- Interactive REPL: plain text = query, `/command` = action
- Slash commands: `/add`, `/list`, `/remove`, `/status`, `/help`, `/exit`
- Backend: **direct Python import** of pinrag core (same env, behind `BackendClient` abstraction)
- Output: Rich markdown, tables, citations
- Input: prompt_toolkit with file history

## Phase 2 — Server Mode & Streaming

- Add `pinrag server` subcommand (streamable-http on `localhost:8765`)
- CLI switches to **MCP streamable-http client** (decoupled from pinrag env)
- Streaming response display (Rich.Live)
- Persistent conversation history (JSON files)
- Collection switching (`/switch`)

## Phase 3 — Polish & Expansion

- Consider **Textual** TUI upgrade (split panes, scrollable history)
- Multi-collection awareness
- Session management (resume previous conversations)
- Config file (`~/.config/pinrag-cli/config.toml`)

---

## Open Questions

1. Streaming over MCP — how to stream RAG responses?
2. Auth for local HTTP transport
3. Conversation history format — JSON vs SQLite?
4. Collection management UX — discovery and switching
5. Server lifecycle — should CLI auto-start the server?
6. Server startup — foreground process or daemon?
