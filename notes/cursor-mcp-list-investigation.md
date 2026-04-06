# Investigation: Adding PinRAG MCP to Cursor's MCP Server List

**Date:** March 2025  
**Updated:** March 2026 — [cursor/mcp-servers](https://github.com/cursor/mcp-servers) deprecated; official listings use [Cursor Directory](https://cursor.directory).  
**Status:** Investigation complete  
**Related:** [implementation-checklist.md](../implementation-checklist.md) — "Investigate how to add pinrag mcp to cursor mcp server list"

---

## Executive Summary

Cursor does **not** have a single built-in MCP marketplace like VS Code's Extensions view. Discovery happens through **Cursor Directory** ([cursor.directory](https://cursor.directory)), **third-party directories**, and README install links. The old GitHub list **[cursor/mcp-servers](https://github.com/cursor/mcp-servers)** is **deprecated** (README redirects to Cursor Directory).

| Platform | Effort | Process | Listing / protocol install |
|----------|--------|---------|---------------------------|
| **Cursor Directory** (official) | Low | [Submit a Plugin](https://cursor.directory/plugins/new) — Manual + **MCP Server** component (or Auto if repo has Open Plugins layout) | Yes — **Add to Cursor** on listing; also `cursor.com/en/install-mcp` |
| **cursor.store** | Low | Web form at cursor.store/mcp/new | Yes |
| **mcp-marketplace.io** | Low | Sign up, submit at /submit | Yes |
| **cursormcp.dev** | Unknown | No public submission; may auto-discover from GitHub | N/A |

**Recommendation:** Submit to **Cursor Directory** first, then **cursor.store** and **mcp-marketplace.io** for broader reach.

---

## 1. Cursor Directory (replaces deprecated cursor/mcp-servers)

### 1.1 Overview

- **Site:** [cursor.directory](https://cursor.directory) — submissions at [cursor.directory/plugins/new](https://cursor.directory/plugins/new).
- **Historical note:** [github.com/cursor/mcp-servers](https://github.com/cursor/mcp-servers) is deprecated; do not use the removed GitHub issue template.
- **Model:** Listings follow the **[Open Plugins](https://open-plugins.com)** standard. A "plugin" can include an **MCP Server** component (`.mcp.json`–style config). Pure PyPI MCP servers often use **Manual** submit because **Auto (GitHub)** expects files like `mcp.json`, `rules/*.mdc`, or `skills/*/SKILL.md` at the repo root.
- **Visibility:** Approved listings show **Add to Cursor**; users still configure API keys in `mcp.json` after install.

**PinRAG:** Submitted Mar 2026 (pending approval). Icon: [`docs/pinrag-icon.svg`](../docs/pinrag-icon.svg).

### 1.2 Submission process

1. Open [Submit a Plugin](https://cursor.directory/plugins/new) and sign in.
2. Choose **Manual** (unless you add root `mcp.json` / Open Plugins layout for Auto scan).
3. Fill **Name**, **Description**, **Repository URL**, optional **Homepage** (e.g. PyPI), **Keywords**, upload **Logo** (square SVG).
4. Under **Components**, add **Type: MCP Server**, **Name** (e.g. `pinrag`), paste MCP install JSON in **Content** (see below — often wrapped as `{ "mcpServers": { "pinrag": { ... } } }` if the form expects a full `.mcp.json` blob).
5. **Publish** — listing may show **Under review** until approved.

### 1.3 Configuration JSON for PinRAG

**Option A — `pinrag` on PATH (user must `pipx install pinrag` first):**

```json
{
  "command": "pinrag",
  "env": {
    "OPENAI_API_KEY": "",
    "PINRAG_PERSIST_DIR": ""
  }
}
```

**Option B — uvx (no prior install; requires `uv` on PATH):**

```json
{
  "command": "uvx",
  "args": ["--refresh", "pinrag"],
  "env": {
    "OPENAI_API_KEY": "",
    "PINRAG_PERSIST_DIR": ""
  }
}
```

**Recommendation:** Use **Option B (uvx)** as primary — fewer setup steps. Use **`uvx --refresh pinrag`** (`"command": "uvx"`, `"args": ["--refresh", "pinrag"]`) so each MCP launch resolves the latest PyPI build (slower cold start than omitting `--refresh`). Document that users need [uv](https://docs.astral.sh/uv/) installed. PinRAG reads only the process environment (MCP `env` in `mcp.json` or exported shell vars); there is no built-in `.env` loader.

### 1.4 Cursor MCP install URL format

Cursor uses: `https://cursor.com/en/install-mcp?name={name}&config={base64}`

Example for PinRAG (uvx — user fills `OPENAI_API_KEY`; optional absolute `PINRAG_PERSIST_DIR`):

```javascript
const config = { command: "uvx", args: ["--refresh", "pinrag"], env: { OPENAI_API_KEY: "", PINRAG_PERSIST_DIR: "" } };
const b64 = btoa(JSON.stringify(config));
const url = `https://cursor.com/en/install-mcp?name=pinrag&config=${encodeURIComponent(b64)}`;
```

**Note:** `btoa` in browser; in Node use `Buffer.from(JSON.stringify(config)).toString('base64')`.

### 1.5 Listing expectations

Treat like a public product listing: valid MCP implementation, maintained repo, clear install/docs, developer-focused use case, professional description and icon. Rejected or low-quality submissions are at Cursor Directory’s discretion (same practical bar as the old GitHub list).

**PinRAG fit:** Developer-focused RAG (PDFs, GitHub repos, YouTube, Discord exports, citations).

### 1.6 Icon

Use a **square SVG** (e.g. [`docs/pinrag-icon.svg`](../docs/pinrag-icon.svg), 128×128 viewBox) for the Directory logo field.

---

## 2. cursor.store

### 2.1 Overview

- **URL:** [cursor.store](https://www.cursor.store/)
- **Submit:** [List your MCP (free)](https://www.cursor.store/mcp/new)
- **Type:** Curated marketplace; free listing, optional paid featured placement ($49/mo)

### 2.2 Requirements (from [rules](https://www.cursor.store/rules))

- **Metadata:** name, slug, short/long description, category, tags, author, repo URL, install snippet, permissions
- **Install snippet:** Must work in Cursor; use secure defaults; no real credentials
- **Permission level:** Declare `low` | `medium` | `high`
  - PinRAG: **medium** (uses `OPENAI_API_KEY` by default; optional Anthropic if configured for LLM; reads/writes local files for Chroma)
- **Quality:** README with install/usage; valid MCP responses; graceful errors

### 2.3 Categories

PinRAG fits: **AI / ML Helpers**, **Data & APIs**, or **Developer Tools**.

### 2.4 Install Snippet for Listing

```json
{
  "mcpServers": {
    "pinrag": {
      "command": "uvx",
      "args": ["--refresh", "pinrag"],
      "env": {
        "OPENAI_API_KEY": "<YOUR_OPENAI_KEY>",
        "PINRAG_PERSIST_DIR": "<absolute/path/to/chroma_db>"
      }
    }
  }
}
```

---

## 3. mcp-marketplace.io

### 3.1 Overview

- **URL:** [mcp-marketplace.io](https://mcp-marketplace.io/)
- **Submit:** [Submit Your Tool](https://mcp-marketplace.io/submit) (requires sign-in: GitHub or email)
- **Features:** Security scanning, in-directory install, 2400+ tools
- **Categories:** Developer Tools, Data & Analytics, Productivity, Content & Media, etc.

### 3.2 Submission

1. Create account (GitHub or email)
2. Go to /submit
3. Provide tool details; platform scans and lists

**PyPI packages:** They discover MCP servers from PyPI. PinRAG is already on PyPI as `pinrag`; consider ensuring the package has `-mcp` in the name or clear MCP metadata for discoverability (optional; they may also accept manual submission).

---

## 4. cursormcp.dev

### 4.1 Overview

- **URL:** [cursormcp.dev](https://cursormcp.dev/)
- **Size:** 1,544 servers, 1,236 developers
- **Submission:** **No public submission form.** Verified March 2026: `/submit`, `/contact`, `/add` all return 404. No contact email in footer (only Privacy Policy link). No public GitHub repo for contributions.
- **Source:** Aggregates from **GitHub star counts** (primary ranking) and likely the **Anthropic/official MCP Registry**. All listed servers link to GitHub repos and show star counts. Crawl/refresh appears infrequent (many entries show "updated 7 months ago").

### 4.2 Action

- **Primary path:** Grow GitHub stars and ensure PinRAG stays listed in the **Anthropic official MCP Registry** (the marketplace listing already says "Imported from the Official MCP Registry" — that's the likely feed cursormcp.dev uses).
- **Direct outreach:** Find the site owner via Twitter/X or WHOIS and pitch PinRAG directly — there is no self-serve form.
- **Timeline:** Manual curation means no guaranteed pickup window; organic discovery via star growth is the more reliable long-term path.

---

## 5. Cursor Install URL (Direct Use)

Even without being listed, you can provide a Cursor MCP install link in the PinRAG README:

```
https://cursor.com/en/install-mcp?name=pinrag&config=eyJjb21tYW5kIjoidXZ4IiwiYXJncyI6WyItLXJlZnJlc2giLCJwaW5yYWciXSwiZW52Ijp7fX0%3D
```

(Decodes to `{"command":"uvx","args":["--refresh","pinrag"],"env":{}}` — user should add keys to `env` in `mcp.json` or export them before starting the client.)

**Generate dynamically:**

```javascript
const config = { command: "uvx", args: ["--refresh", "pinrag"], env: {} };
const url = `https://cursor.com/en/install-mcp?name=pinrag&config=${encodeURIComponent(btoa(JSON.stringify(config)))}`;
```

---

## 6. Cursor MCP Extension API (Programmatic Registration)

For enterprise/MDM use, Cursor exposes [vscode.cursor.mcp.registerServer](https://cursor.com/docs/context/mcp-extension-api). This is for **extensions** that register MCP servers programmatically, not for getting listed in directories. PinRAG could be wrapped in a Cursor extension (similar to VS Code) that calls `registerServer` — see [vscode-marketplace-investigation.md](vscode-marketplace-investigation.md) for the extension approach.

---

## 7. Recommended Action Plan

### Phase 1: Official list (1–2 hours)

1. **Create PinRAG icon** — Square SVG (e.g. [`docs/pinrag-icon.svg`](../docs/pinrag-icon.svg))
2. **Submit to Cursor Directory** via [Submit a Plugin](https://cursor.directory/plugins/new) — Manual + MCP Server component, uvx config, attach SVG (**done** Mar 2026, pending approval)
3. **Add Cursor install link to README** — Link to `cursor.com/en/install-mcp?name=pinrag&config=...` in Cursor section (already in README)

### Phase 2: Other directories (30 min each)

1. **cursor.store** — Submit at [mcp/new](https://www.cursor.store/mcp/new) with metadata and install snippet
2. **mcp-marketplace.io** — Sign up, submit at [submit](https://mcp-marketplace.io/submit)

### Phase 3: Monitor

- Watch for cursormcp.dev listing after Directory approval / GitHub visibility
- If Cursor Directory shows **Under review**, wait for approval or edit listing as needed

---

## 8. References

- [Cursor Directory](https://cursor.directory) — [Submit a Plugin](https://cursor.directory/plugins/new)
- [Open Plugins](https://open-plugins.com) — [MCP servers component](https://open-plugins.com/agent-builders/components/mcp-servers)
- [cursor/mcp-servers](https://github.com/cursor/mcp-servers) — Deprecated (README → Cursor Directory)
- [cursor.store](https://www.cursor.store/) — [List MCP](https://www.cursor.store/mcp/new), [Rules](https://www.cursor.store/rules)
- [mcp-marketplace.io](https://mcp-marketplace.io/) — [Submit](https://mcp-marketplace.io/submit)
- [cursormcp.dev](https://cursormcp.dev/) — Directory (no clear submit)
- [Cursor MCP Extension API](https://cursor.com/docs/context/mcp-extension-api)
