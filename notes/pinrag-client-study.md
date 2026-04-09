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
| Conversation history | JSON files under `~/.pinrag-cli/history/` |
| Session follow-ups (planned) | Rolling recent Q/A prepended to the next query (Phase 2.5) |
| MCP client | `mcp` SDK (`streamable_http_client`, `ClientSession`) |
| ASGI server | **Uvicorn** |

---

## Phase 0 — Core Extraction *(done)*

Extract transport-agnostic business logic out of `mcp/tools.py` into `pinrag.core` so any consumer (CLI, MCP server, scripts) can import cleanly.

- Create `src/pinrag/core/operations.py` — move `query()`, `add_file()`, `add_files()`, `list_documents()`, `remove_document()` (same signatures, same behavior)
- Create `src/pinrag/core/format_detection.py` — move format detection, path resolution, GitHub URL validation helpers
- Create `src/pinrag/core/__init__.py` — clean public API exports
- Slim `mcp/tools.py` to backward-compatible re-exports from `pinrag.core` (including legacy `_resolve_*` / `_detect_*` aliases for imports that still target `pinrag.mcp.tools`)
- No new dependencies, no behavioral changes; tests that mock internals via `patch("pinrag.mcp.tools.*")` were updated to `pinrag.core.operations` (and `pathlib` where tests used `mcp_tools.Path`)

After this phase, `pinrag-cli` (and future consumers) can `from pinrag.core import query, add_files, list_documents, remove_document`.

## Phase 1 — Minimal Viable CLI *(done)*

- Interactive REPL: plain text = query, `/command` = action
- Slash commands: `/add`, `/list`, `/remove`, `/status`, `/help`, `/exit`
- Backend: **direct Python import** of pinrag core (same env, behind `BackendClient` abstraction)
- Output: Rich markdown, tables, citations
- Input: prompt_toolkit with file history

## Phase 2 — Server Mode & Streaming *(implemented)*

- **`pinrag server`** — streamable-http MCP (default bind e.g. `127.0.0.1:8765`).
- **`pinrag-cli --server <url>`** — REPL uses the HTTP MCP client; LLM/embeddings env live on the server process.
- **Rich.Live** — streaming progress while querying and for tool-backed slash commands.
- **Session history** — JSON turns under `~/.pinrag-cli/history/`; **`/history`** in the REPL.
- **`/switch`** — list or select Chroma collection name for the active backend.

## Phase 2.5 — Session conversational memory *(planned)*

Follow-ups in one session (“elaborate”, “what about page 5?”) without restating context: **pinrag-cli** keeps a short in-memory **rolling Q/A window** and folds it into the next query (bounded length). JSON `/history` unchanged. Optional **`/clear`** and env to turn off. Cross-session resume and core/API chat history stay Phase 3+.

## Phase 3 — Polish & Expansion

Split into three CLI polish tracks with a clear dependency order. **Phase 4** (below) is the Textual TUI and ships after 3a–3c are stable.

| Action | Name | Scope | Depends On | Priority |
|--------|------|-------|------------|----------|
| **3a** | Config file support | Small-Medium | None | **Do first** |
| **3b** | Session resume | Medium | 3a (to persist last session) | Second |
| **3c** | Multi-collection UX polish | Medium | 3a (to persist default collection) | Third |

### Phase 3a — Config file (`~/.config/pinrag-cli/config.toml`)

Foundation for 3b and 3c — both need to persist state.

- Parse TOML (`tomllib`, stdlib since Python 3.11)
- Persist defaults: collection, persist-dir, server URL, LLM provider/model
- CLI preferences: memory on/off, memory turns, response style
- Per-project overrides via `.pinrag-cli.toml` in CWD
- Precedence chain: CLI flags > env vars > project config > user config > defaults
- `/config` slash command to view/edit from REPL
- Zero behavior change for existing env-var users

### Phase 3b — Session management (resume previous conversations)

Infrastructure already exists (`ConversationStore`, `list_sessions()`, JSON files) — needs UX.

- **`/sessions`** command — list previous sessions (date, collection, turn count, last query preview)
- **`/resume [session-id | index]`** — reload a prior session's history into memory window
- **`--resume` CLI flag** — start the REPL with a previous session loaded
- Resume primes the conversational memory window (not full Rich output reconstruction)
- Optional session naming/aliasing

### Phase 3c — Multi-collection UX polish

Builds on config (persistent default collection) and session resume (sessions remember their collection).

- `/switch` persists the choice to config file
- Collection metadata in `/switch` output: description, creation date, document count
- Better `/list` filtering when multiple collections exist
- Session files tagged with collection context for resume
- Consider cross-collection queries (query multiple collections, merge results) — may be later scope beyond 3c

## Phase 4 — Textual TUI upgrade

Full UI-layer rewrite for **pinrag-cli** (~2–3× the scope of 3a+3b+3c combined). Depends on 3a–3c being stable so new commands and flows are not built twice.

- Replace `prompt_toolkit` + `Rich` rendering with **Textual** full-screen app
- Split-pane layout: input area, scrollable conversation history, sidebar for collections/documents
- Textual has its own event loop, widget system, CSS-like styling — fundamentally different paradigm
- Risk: Textual apps behave differently in SSH, tmux, limited terminals — consider keeping classic REPL as `--no-tui` fallback
- Needs own test strategy (Textual `pilot` for snapshot testing)
- Rich rendering helpers from `output.py` partially reusable; orchestration changes completely
- Implement after 3b and 3c: those phases add commands/flows that would otherwise need dual implementation

---

## Open Questions

1. Streaming over MCP — how to stream RAG responses?
2. Auth for local HTTP transport
3. Conversation history format — JSON vs SQLite?
4. Collection management UX — defaults, discovery, naming (beyond `/switch`)
5. Server lifecycle — should CLI auto-start the server?
6. Server startup — foreground process or daemon?
