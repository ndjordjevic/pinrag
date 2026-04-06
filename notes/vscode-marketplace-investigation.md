# Investigation: Adding PinRAG MCP to VS Code Extensions Marketplace

**Date:** March 2025  
**Status:** Investigation complete  
**Related:** [implementation-checklist.md](../implementation-checklist.md) — "Investigate to add pinrag mcp to visual studio code extensions marketplace"

---

## Executive Summary

There are **three viable approaches** to get PinRAG MCP into VS Code’s ecosystem:

| Approach | Effort | Discovery | Pros | Cons |
|----------|--------|-----------|------|------|
| **A. VS Code Extension** | High | Extensions view (`@mcp pinrag`) | Marketplace install, auto-updates, discoverable | Requires TypeScript extension + Python runtime handling |
| **B. Protocol install URL** | Low | README / website link | No extension build, works today | Not in Extensions view; user must open the link |
| **C. MCP Server Gallery (curated)** | Unknown | Possibly `code.visualstudio.com/mcp` | Official listing | Process unclear; may require Microsoft approval |

**Recommendation:** Start with **Approach B** (protocol install URL) for immediate value. Pursue **Approach A** if you want marketplace presence and discoverability.

---

## 1. How VS Code Discovers MCP Servers

### 1.1 MCP Server Gallery (Extensions View)

When users search `@mcp` in the Extensions view (⇧⌘X), they see an **MCP server gallery**. Servers there come from:

1. **VS Code extensions** that register MCP servers via `mcpServerDefinitionProviders`
2. Extensions are published to the [Visual Studio Marketplace](https://marketplace.visualstudio.com/)

Examples:

- **Azure MCP Server** (`ms-azuretools.vscode-azure-mcp-server`) — 600k+ installs
- **VSCode MCP Server** (`SemanticWorkbenchTeam.mcp-server-vscode`) — 18k+ installs
- **Playwright MCP** — installable via `@mcp playwright`

To appear in this gallery, PinRAG must be packaged as a **VS Code extension**.

### 1.2 Manual Configuration (`mcp.json`)

Users can add MCP servers by editing `mcp.json` (user profile or `.vscode/mcp.json`). PinRAG already documents this:

```json
{
  "servers": {
    "pinrag": {
      "command": "pinrag",
      "env": { "OPENAI_API_KEY": "...", "ANTHROPIC_API_KEY": "..." }
    }
  }
}
```

This works today but requires users to install `pinrag` (pipx/uv) and edit config manually.

### 1.3 Protocol install URL (`vscode:mcp/install`)

VS Code supports a URL handler for installing MCP servers:

```
vscode:mcp/install?{url-encoded-json-config}
```

When opened (browser, `xdg-open`, etc.), VS Code adds the server to the user’s `mcp.json` and prompts to start it. **No extension required.**

### 1.4 Command-Line

```bash
code --add-mcp '{"name":"pinrag","command":"pinrag","env":{...}}'
```

---

## 2. Approach A: VS Code Extension (Marketplace)

### 2.1 Architecture

An extension that **wraps** the existing Python MCP server:

1. **Extension** (TypeScript) — contributes `mcpServerDefinitionProviders`, registers the PinRAG MCP server
2. **MCP server** — the existing `pinrag` CLI, invoked as a subprocess

The extension does **not** reimplement the MCP server. It spawns `pinrag` (or `uvx --refresh pinrag` / `python -m pinrag.cli`) and connects via stdio.

### 2.2 Key API: `vscode.lm.registerMcpServerDefinitionProvider`

From the [VS Code MCP Developer Guide](https://code.visualstudio.com/api/extension-guides/mcp):

```typescript
// package.json
{
  "contributes": {
    "mcpServerDefinitionProviders": [
      { "id": "pinrag.mcpServer", "label": "PinRAG MCP" }
    ]
  }
}

// extension.ts
vscode.lm.registerMcpServerDefinitionProvider('pinrag.mcpServer', {
  provideMcpServerDefinitions: async () => {
    return [
      new vscode.McpStdioServerDefinition({
        label: 'PinRAG',
        command: 'uvx',  // or 'pinrag' if on PATH
        args: ['--refresh', 'pinrag'],
        env: { /* optional: pass PINRAG_PERSIST_DIR etc */ },
        version: '0.8.5'
      })
    ];
  },
  resolveMcpServerDefinition: async (server) => {
    // Optional: prompt for API keys via vscode.window.showInputBox
    return server;
  }
});
```

### 2.3 Python Runtime Handling

PinRAG is a Python package. Options:

| Option | Command | Pros | Cons |
|--------|---------|------|------|
| **uvx** | `uvx --refresh pinrag` | No user install; uv fetches latest PyPI build each launch | Requires `uv` on PATH; slower cold start than omitting `--refresh` |
| **pipx** | `pipx run --spec pinrag pinrag` | Common for Python tools | Requires pipx |
| **pinrag** | `pinrag` | Simple if already installed | User must `pipx install pinrag` first |
| **Bundled Python** | Extension ships Python + venv | No external deps | Large package, complex build |

**Recommended:** Use `uvx --refresh pinrag` (or `uv tool run --from pinrag pinrag`) as the default so the extension tracks latest PyPI releases. Document that `uv` must be installed. Fallback: check for `pinrag` on PATH and use that if present.

### 2.4 Extension Project Structure

```
pinrag-vscode/
├── package.json          # publisher, name, contributes.mcpServerDefinitionProviders
├── src/
│   └── extension.ts      # registerMcpServerDefinitionProvider
├── tsconfig.json
├── .vscodeignore
└── README.md
```

### 2.5 Publishing Steps

1. **Scaffold:** `npm install -g yo generator-code && yo code` (or copy [mcp-extension-sample](https://github.com/microsoft/vscode-extension-samples/tree/main/mcp-extension-sample))
2. **Implement:** Add `mcpServerDefinitionProviders` and spawn `uvx --refresh pinrag` (or `pinrag` on PATH)
3. **Package:** `npm install -g @vscode/vsce && vsce package`
4. **Publish:** Create publisher at [marketplace.visualstudio.com/manage](https://marketplace.visualstudio.com/manage), then `vsce publish`
5. **Ongoing:** Bump version in `package.json`, run `vsce publish patch`

### 2.6 Effort Estimate

- **Initial:** 1–2 days (scaffold, implement, test, publish)
- **Maintenance:** Keep extension version in sync with PyPI releases; consider automation (e.g. GitHub Action on tag)

---

## 3. Approach B: Protocol install URL (no extension)

### 3.1 How It Works

VS Code’s URL handler: `vscode:mcp/install?{json}`

Create a URL with the MCP server config (JSON-stringified and URL-encoded). When the user opens it, VS Code adds the server to `mcp.json`.

### 3.2 PinRAG Config for URL

```javascript
const config = {
  name: "pinrag",
  command: "uvx",
  args: ["--refresh", "pinrag"],
  env: {}  // User adds keys via input variables or env file
};
const url = `vscode:mcp/install?${encodeURIComponent(JSON.stringify(config))}`;
// Result: vscode:mcp/install?%7B%22name%22%3A%22pinrag%22%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22--refresh%22%2C%22pinrag%22%5D%7D
```

### 3.3 With Input Variables (API Keys)

For sensitive data, use `inputs` in `mcp.json`. The install URL can reference `${input:openai-key}` if the schema supports it. The [MCP configuration reference](https://code.visualstudio.com/docs/copilot/reference/mcp-configuration) shows `inputs` as a separate top-level array. The URL format may only support the `servers` object—verify with a minimal test.

**Simpler approach:** Use a URL that installs the server with empty `env`; document that users must add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` via `mcp.json` inputs or `envFile`.

### 3.4 Implementation

1. Add a section to the README: "Install in VS Code (protocol link)"
2. Provide a link, e.g. `[Install PinRAG MCP in VS Code](vscode:mcp/install?...)`
3. Optionally host a small HTML page that generates the URL (e.g. for different `command` options: `uvx` vs `pinrag`)

### 3.5 Effort Estimate

- **Initial:** 1–2 hours (create URL, test, document)
- **Maintenance:** None

---

## 4. Approach C: VS Code Curated MCP List

VS Code has a curated MCP page: [code.visualstudio.com/mcp](https://code.visualstudio.com/mcp) (fetch failed during investigation; URL may have changed).

**Process:** Unclear. Likely requires:

- Submitting PinRAG for review
- Meeting quality/security bar
- Possibly having an extension or a stable install path

**Recommendation:** Treat this as a follow-up after Approach A or B. Check the current MCP page and any "Submit" or "Contribute" links for the exact process.

---

## 5. Comparison: Extension vs Install URL

| Criterion | Extension (A) | Install URL (B) |
|-----------|--------------|------------------|
| Discoverability | High (Extensions view, `@mcp`) | Low (README, docs, website) |
| User flow | Install extension → server auto-registered | Click link → config added → user adds keys |
| Python dependency | Extension uses `uvx` or documents `pipx` | Same; user needs `uv` or `pipx` |
| API keys | Can use `resolveMcpServerDefinition` to prompt | User configures in `mcp.json` |
| Updates | Extension version separate from PyPI | User runs `pipx upgrade pinrag` |
| Maintenance | Sync extension with PyPI releases | Minimal |
| Effort | 1–2 days | 1–2 hours |

---

## 6. Recommended Action Plan

### Phase 1: Quick Win (1–2 hours)

1. **Create a protocol install URL** for PinRAG.
2. **Document in README** under "VS Code" or "Install (protocol link)".
3. **Test** in VS Code (and VS Code Insiders if needed).

Example README addition:

```markdown
### Install in VS Code (protocol link)

Click to add PinRAG to VS Code's MCP configuration: [Install PinRAG MCP](vscode:mcp/install?%7B%22name%22%3A%22pinrag%22%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22--refresh%22%2C%22pinrag%22%5D%7D)

Requires [uv](https://docs.astral.sh/uv/) on your PATH. Alternatively, install PinRAG first (`pipx install pinrag`) and use [this config](vscode:mcp/install?%7B%22name%22%3A%22pinrag%22%2C%22command%22%3A%22pinrag%22%7D) with `command: "pinrag"` only.
```

### Phase 2: Marketplace Extension (1–2 days)

1. Create `pinrag-vscode` repo (or `vscode/` in pinrag monorepo).
2. Implement extension with `mcpServerDefinitionProviders` spawning `uvx --refresh pinrag`.
3. Publish to Visual Studio Marketplace.
4. Add CI to bump and publish extension on PinRAG releases.

### Phase 3: Curated List (Optional)

1. Revisit [code.visualstudio.com/mcp](https://code.visualstudio.com/mcp).
2. Submit PinRAG if a process exists and requirements are met.

---

## 7. References

- [VS Code MCP Developer Guide](https://code.visualstudio.com/api/extension-guides/mcp)
- [Add and manage MCP servers](https://code.visualstudio.com/docs/copilot/customization/mcp-servers)
- [MCP configuration reference](https://code.visualstudio.com/docs/copilot/reference/mcp-configuration)
- [Publishing Extensions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension)
- [Adding an MCP Server to a VS Code Extension (Ken Muse)](https://www.kenmuse.com/blog/adding-mcp-server-to-vs-code-extension/)
- [mcp-extension-sample](https://github.com/microsoft/vscode-extension-samples/tree/main/mcp-extension-sample)
- [Azure MCP Server](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azure-mcp-server)
- [VSCode MCP Server (Semantic Workbench)](https://marketplace.visualstudio.com/items?itemName=SemanticWorkbenchTeam.mcp-server-vscode)
