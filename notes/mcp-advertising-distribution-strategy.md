# PinRAG MCP: Advertising & Distribution Strategy

---

## Executive Summary

Two complementary tracks:

1. **Distribution** — Get PinRAG listed in directories, marketplaces, and IDE galleries so people can *find and install* it.
2. **Advertising** — Drive awareness so people actually *visit* those listings, star the repo, and try PinRAG.

Neither works alone. Listings without promotion = buried among 17,000+ servers. Promotion without listings = nowhere for interested users to land with a README install link.

**Two-tier distribution:** (1) **MCP-only** — PyPI package `pinrag` with the **`pinrag`** MCP CLI entrypoint; works in every MCP client. (2) **Claude Code plugin** — separate repo [pinrag-plugin](https://github.com/ndjordjevic/pinrag-plugin) with `.claude-plugin/`, `.mcp.json`, and `skills/use-pinrag/` for [Claude Code](https://code.claude.com/docs/en/plugins.md) only (marketplace or `claude --plugin-dir`).

---

## 1. Distribution Channels

PinRAG reaches users through two mechanisms:

1. **MCP-only** — the PyPI package (`pinrag`) added to any MCP client's config via `uvx --refresh pinrag` (`"command": "uvx"`, `"args": ["--refresh", "pinrag"]`). Works in every MCP client.
2. **Claude Code plugin** — the [pinrag-plugin](https://github.com/ndjordjevic/pinrag-plugin) repo ships `.claude-plugin/`, `.mcp.json`, and `skills/use-pinrag/` for **Claude Code** only (not a multi-editor Open Plugins bundle).

### 1.1 Status Overview

| # | Channel | Type | Status |
|---|---------|------|--------|
| 1 | [Official MCP Registry](https://registry.modelcontextprotocol.io/?q=pinrag) | MCP | **Done** |
| 2 | [mcp-marketplace.io](https://mcp-marketplace.io/server/io-github-ndjordjevic-pinrag) | MCP | **Done** |
| 3 | [cursor.store](https://www.cursor.store/mcp/ndjordjevic/pinrag) - Still in Beta | MCP | **Done** |
| 4 | [Cursor Directory](https://cursor.directory) | MCP | **Done** |
| 5 | [Goose Agent Skills PR](https://github.com/block/agent-skills/pull/18) | Plugin | **PR pending** ([#18](https://github.com/block/agent-skills/pull/18)) |
| 6 | [mcp.so](https://mcp.so) | MCP | **Submitted** ([web form](https://mcp.so/submit); pending listing) |
| 7 | [MCPRepository](https://mcprepository.com/ndjordjevic/pinrag) | MCP | **Done** ([PinRAG](https://mcprepository.com/ndjordjevic/pinrag); [`mcp-index` CLI](https://github.com/mcprepository/mcp-index)) |
| 8 | [Awesome MCP Servers](https://mcpservers.org/servers/ndjordjevic/pinrag) | Visibility | **Done** — [PinRAG on mcpservers.org](https://mcpservers.org/servers/ndjordjevic/pinrag) (submitted [mcpservers.org/submit](https://mcpservers.org/submit), Mar 2026). |
| 9 | [Claude Code marketplace](https://claude.ai/settings/plugins/submit) | Plugin | **Submitted** — [platform.claude.com/plugins/submit](https://platform.claude.com/plugins/submit) (Mar 2026); pending review |
| 10 | [MCP Market](https://mcpmarket.com/) | MCP | **Done** — [PinRAG on MCP Market](https://mcpmarket.com/server/pinrag) (submitted [mcpmarket.com/submit](https://mcpmarket.com/submit) with `https://github.com/ndjordjevic/pinrag`, Mar 2026). |
| 11 | [MCPCentral](https://mcpcentral.io/) | MCP | **Done** — [registry search `pinrag`](https://mcpcentral.io/registry?q=pinrag) returns PinRAG (`github.com/ndjordjevic/pinrag`; Mar 2026 browser verification). Updates via `mcp-publisher` + `-registry https://registry.mcpcentral.io` ([submit-server](https://mcpcentral.io/submit-server)). |
| 12 | [Cursor Marketplace](https://cursor.com/marketplace/publish) (official plugin) | Plugin | **N/A / superseded** — [pinrag-plugin](https://github.com/ndjordjevic/pinrag-plugin) is Claude Code–only; Cursor uses MCP-only paths (README, registry, cursor.store). Prior publisher application (Mar 2026) is moot unless repointed at a Cursor-specific bundle. |
| 13 | [Glama](https://glama.ai/mcp/servers/ndjordjevic/pinrag) | MCP | **Done** — [PinRAG on Glama](https://glama.ai/mcp/servers/ndjordjevic/pinrag); [`glama.json`](../glama.json) in repo. |
| 14 | [cursormcp.dev](https://cursormcp.dev/) | Directory | **Unknown** — no public “add server” / submit UI; unclear how PinRAG could be listed |

### 1.2 Tool → Channel Matrix

**Bold** = PinRAG is listed on that channel **or** the install path works for that tool today (README / plugin / registry).

| Tool | Live channels | Planned channels |
|------|--------------|-----------------|
| **Cursor** | **MCP Registry**, **mcp-marketplace.io**, **cursor.store**, **Cursor Directory**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **Awesome MCP Servers** ([listing](https://mcpservers.org/servers/ndjordjevic/pinrag)), **README Quick Start (install links)**, **manual config** | mcp.so (submitted; pending — §1.1 row 6), [cursormcp.dev](https://cursormcp.dev/) (§1.1 row 14; no submit path known) |
| **VS Code Copilot** | **MCP Registry**, **mcp-marketplace.io**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **README Quick Start (install links)**, **manual config** | mcp.so (§1.1 row 6), VS Code Extension, VS Code Copilot marketplace (not in §1.1) |
| **Claude Code** | **MCP Registry**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config**, **[pinrag-plugin](https://github.com/ndjordjevic/pinrag-plugin)** (Claude Code plugin) | Claude Code marketplace (submitted Mar 2026; pending — §1.1 row 9), mcp.so (§1.1 row 6) |
| **JetBrains + Copilot** | **MCP Registry**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config** | mcp.so |
| **Windsurf** | **MCP Registry**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config** | windsurf.run — no working self-serve listing (not a §1.1 row); mcp.so (§1.1 row 6) |
| **Zed** | **MCP Registry**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config** | mcp.so |
| **OpenCode** | **MCP Registry**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config** | mcp.so |
| **Amp** | **MCP Registry**, **MCPRepository** (listed), **Glama**, **MCP Market** ([listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config** | — |
| **Goose** | **MCP Market** ([MCP server listing](https://mcpmarket.com/server/pinrag)), **MCPCentral** ([registry](https://mcpcentral.io/registry?q=pinrag)), **manual config** (stdio MCP from main `pinrag` repo) | Goose Agent Skills (PR [#18](https://github.com/block/agent-skills/pull/18) — §1.1 row 5) |

---

## 2. Advertising

- **Reddit** — e.g. r/MCP, r/langchain: one post when you are ready; follow each sub’s rules.
- **MCP Discord** — MCP-focused servers: share PinRAG in the right channel, answer questions.
