"""Unit tests for web docs URL utilities, discovery, loader, indexer, and detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from langchain_core.embeddings import Embeddings

from pinrag.core.format_detection import detect_source_format, is_web_docs_url
from pinrag.indexing import (
    CrawlLimits,
    WebIndexResult,
    index_web,
    load_web_docs_as_documents,
)
from pinrag.indexing.web_loader import (
    _parse_llms_txt_links,
    _seed_prefix,
    is_noise_url,
    normalize_url,
    same_scope,
)


# ---------------------------------------------------------------------------
# Mock embeddings (no API key, no network)
# ---------------------------------------------------------------------------


class _MockEmbeddings(Embeddings):
    """Returns fixed-dim vectors (1536) for testing without API key."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 1536


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


def test_normalize_url_drops_fragment_and_tracking() -> None:
    assert (
        normalize_url("https://Docs.Example.COM/a/b#section")
        == "https://docs.example.com/a/b"
    )
    assert (
        normalize_url("https://x.com/p?utm_source=foo&x=1&fbclid=z")
        == "https://x.com/p?x=1"
    )
    assert normalize_url("https://x.com//a///b") == "https://x.com/a/b"


def test_normalize_url_preserves_path_case() -> None:
    # URLs are case-sensitive after the host; don't lowercase the path.
    assert (
        normalize_url("https://x.com/CaseSensitive/Path")
        == "https://x.com/CaseSensitive/Path"
    )


def test_seed_prefix_file_and_directory() -> None:
    assert _seed_prefix("https://x.com/") == ("x.com", "/")
    assert _seed_prefix("https://x.com/docs/") == ("x.com", "/docs/")
    assert _seed_prefix("https://x.com/docs/page.html") == ("x.com", "/docs/")
    assert _seed_prefix("https://x.com") == ("x.com", "/")


def test_same_scope_enforces_host_and_prefix() -> None:
    assert same_scope("https://x.com/docs/a", "x.com", "/docs/") is True
    # Different subdomain => reject.
    assert same_scope("https://api.x.com/docs/a", "x.com", "/docs/") is False
    # Same host but outside prefix => reject.
    assert same_scope("https://x.com/blog/a", "x.com", "/docs/") is False
    # Non-http scheme => reject.
    assert same_scope("ftp://x.com/docs/a", "x.com", "/docs/") is False


def test_is_noise_url() -> None:
    assert is_noise_url("https://x.com/file.zip") is True
    assert is_noise_url("https://x.com/IMG.PNG") is True
    assert is_noise_url("https://x.com/style.css") is True
    assert is_noise_url("https://x.com/docs/page.html") is False
    assert is_noise_url("https://x.com/docs/page") is False


# ---------------------------------------------------------------------------
# llms.txt parser
# ---------------------------------------------------------------------------


def test_parse_llms_txt_links_extracts_title_and_url() -> None:
    body = """# Docs

## Pages

- [Intro](https://docs.example.com/intro.md): welcome
- [API](https://docs.example.com/api.md)
- [Other](relative/path.md): skipped because not absolute
"""
    links = _parse_llms_txt_links(body)
    urls = [u for u, _ in links]
    assert "https://docs.example.com/intro.md" in urls
    assert "https://docs.example.com/api.md" in urls
    # relative URLs are skipped
    assert all(u.startswith("http") for u in urls)
    titles = dict(links)
    assert titles["https://docs.example.com/intro.md"] == "Intro"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_is_web_docs_url() -> None:
    assert is_web_docs_url("https://docs.langchain.com/") is True
    assert is_web_docs_url("http://example.com/foo") is True
    assert is_web_docs_url("not-a-url") is False
    assert is_web_docs_url("ftp://example.com/foo") is False
    assert is_web_docs_url("javascript:alert(1)") is False
    assert is_web_docs_url("") is False


def test_detect_source_format_web_precedence() -> None:
    # GitHub still wins for github.com.
    assert detect_source_format("https://github.com/foo/bar") == "github"
    # github.io pages host is NOT github.com → routes to web.
    assert detect_source_format("https://picocomputer.github.io/") == "web"
    # YouTube still wins.
    assert (
        detect_source_format("https://www.youtube.com/watch?v=abcdefghijk") == "youtube"
    )
    # Generic docs URL → web.
    assert detect_source_format("https://docs.langchain.com/") == "web"
    assert detect_source_format("https://docs.crewai.com/") == "web"


# ---------------------------------------------------------------------------
# Loader with mocked HTTP
# ---------------------------------------------------------------------------


def _mock_transport(
    routes: dict[str, tuple[int, str, dict[str, str] | None]],
) -> httpx.MockTransport:
    """Build a MockTransport that serves ``routes``: url -> (status, body, headers)."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        # Also allow matching on path-only for same-host routing.
        if url in routes:
            status, body, headers = routes[url]
            return httpx.Response(
                status, text=body, headers=headers or {"content-type": "text/html"}
            )
        # Fallback: strip trailing slash variant.
        alt = url.rstrip("/")
        if alt in routes:
            status, body, headers = routes[alt]
            return httpx.Response(
                status, text=body, headers=headers or {"content-type": "text/html"}
            )
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _AsyncClientFactory:
    """Monkeypatch target: swap httpx.AsyncClient for a mock-transport variant."""

    def __init__(self, transport: httpx.MockTransport):
        self._transport = transport

    def __call__(self, *args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=self._transport, **kwargs)


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, tuple[int, str, dict[str, str] | None]],
) -> None:
    transport = _mock_transport(routes)
    monkeypatch.setattr(
        "pinrag.indexing.web_loader.httpx.AsyncClient",
        _AsyncClientFactory(transport),
    )


def test_loader_bfs_crawl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded BFS with no llms.txt / sitemap: crawl from seed, follow scoped links."""
    routes = {
        "https://site.example/robots.txt": (404, "", {"content-type": "text/plain"}),
        "https://site.example/llms.txt": (404, "", {"content-type": "text/plain"}),
        "https://site.example/sitemap.xml": (404, "", {"content-type": "text/xml"}),
        "https://site.example/sitemap_index.xml": (
            404,
            "",
            {"content-type": "text/xml"},
        ),
        "https://site.example/": (
            200,
            '<html><body><a href="/a.html">A</a><a href="/b.html">B</a>'
            '<a href="https://other.example/x">off</a></body></html>',
            {"content-type": "text/html"},
        ),
        "https://site.example/a.html": (
            200,
            "<html><head><title>Page A</title></head><body>"
            "<main><h1>Page A</h1><p>"
            + ("Alpha content. " * 40)
            + "</p></main></body></html>",
            {"content-type": "text/html"},
        ),
        "https://site.example/b.html": (
            200,
            "<html><head><title>Page B</title></head><body>"
            "<main><h1>Page B</h1><p>"
            + ("Beta content. " * 40)
            + "</p></main></body></html>",
            {"content-type": "text/html"},
        ),
    }
    _install_mock_transport(monkeypatch, routes)

    limits = CrawlLimits(
        max_pages=10,
        max_depth=2,
        rate_limit_per_host=100.0,
        respect_robots=False,
        prefer_llms_txt=False,
    )
    result = load_web_docs_as_documents("https://site.example/", limits=limits)
    assert result.discovery == "crawl"
    assert result.host == "site.example"
    assert result.path_prefix == "/"
    assert result.document_id == "site.example/"
    urls = {d.metadata["source_url"] for d in result.documents}
    # BFS finds root + a + b; off-scope is dropped.
    assert "https://site.example/a.html" in urls
    assert "https://site.example/b.html" in urls
    assert not any("other.example" in u for u in urls)
    for d in result.documents:
        assert d.metadata["document_type"] == "web"
        assert d.metadata["document_id"] == "site.example/"


def test_loader_sitemap_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """sitemap.xml URLs are used when llms.txt is absent."""
    sitemap = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://s.example/docs/a</loc></url>
  <url><loc>https://s.example/docs/b</loc></url>
  <url><loc>https://s.example/blog/off</loc></url>
  <url><loc>https://other.example/docs/nope</loc></url>
</urlset>
"""
    routes = {
        "https://s.example/robots.txt": (404, "", {"content-type": "text/plain"}),
        "https://s.example/llms.txt": (404, "", {"content-type": "text/plain"}),
        "https://s.example/sitemap.xml": (
            200,
            sitemap,
            {"content-type": "application/xml"},
        ),
        "https://s.example/docs/a": (
            200,
            "<html><body><main><h1>A</h1><p>"
            + ("content " * 30)
            + "</p></main></body></html>",
            {"content-type": "text/html"},
        ),
        "https://s.example/docs/b": (
            200,
            "<html><body><main><h1>B</h1><p>"
            + ("content " * 30)
            + "</p></main></body></html>",
            {"content-type": "text/html"},
        ),
    }
    _install_mock_transport(monkeypatch, routes)
    limits = CrawlLimits(
        max_pages=20,
        rate_limit_per_host=100.0,
        respect_robots=False,
        prefer_llms_txt=False,
    )
    result = load_web_docs_as_documents("https://s.example/docs/", limits=limits)
    assert result.discovery == "sitemap"
    urls = {d.metadata["source_url"] for d in result.documents}
    assert urls == {"https://s.example/docs/a", "https://s.example/docs/b"}


def test_loader_llms_txt_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """llms.txt returns markdown URLs; each is fetched as-is (content-type markdown)."""
    llms_txt = """# Docs

- [Alpha](https://l.example/docs/alpha.md): intro
- [Beta](https://l.example/docs/beta.md)
- [Off-scope](https://l.example/blog/x.md)
- [Other host](https://other.example/docs/y.md)
"""
    routes = {
        "https://l.example/robots.txt": (404, "", {"content-type": "text/plain"}),
        "https://l.example/llms.txt": (
            200,
            llms_txt,
            {"content-type": "text/markdown"},
        ),
        "https://l.example/docs/alpha.md": (
            200,
            "# Alpha\n\nAlpha markdown body.",
            {"content-type": "text/markdown"},
        ),
        "https://l.example/docs/beta.md": (
            200,
            "# Beta\n\nBeta markdown body.",
            {"content-type": "text/markdown"},
        ),
    }
    _install_mock_transport(monkeypatch, routes)
    limits = CrawlLimits(
        max_pages=20,
        rate_limit_per_host=100.0,
        respect_robots=False,
    )
    result = load_web_docs_as_documents("https://l.example/docs/", limits=limits)
    assert result.discovery == "llms_txt"
    urls = {d.metadata["source_url"] for d in result.documents}
    assert urls == {
        "https://l.example/docs/alpha.md",
        "https://l.example/docs/beta.md",
    }
    # llms.txt title overrides html title extraction.
    titles = {d.metadata["source_url"]: d.metadata.get("doc_title") for d in result.documents}
    assert titles["https://l.example/docs/alpha.md"] == "Alpha"
    # Markdown body is returned unchanged (not re-run through trafilatura).
    bodies = {d.metadata["source_url"]: d.page_content for d in result.documents}
    assert "Alpha markdown body" in bodies["https://l.example/docs/alpha.md"]


def test_loader_subdomain_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-subdomain URLs in llms.txt are dropped."""
    llms_txt = "- [ok](https://main.example/docs/a.md)\n- [bad](https://sub.main.example/docs/b.md)\n"
    routes = {
        "https://main.example/robots.txt": (404, "", {"content-type": "text/plain"}),
        "https://main.example/llms.txt": (
            200,
            llms_txt,
            {"content-type": "text/markdown"},
        ),
        "https://main.example/docs/a.md": (
            200,
            "# A\n\nbody",
            {"content-type": "text/markdown"},
        ),
    }
    _install_mock_transport(monkeypatch, routes)
    limits = CrawlLimits(
        max_pages=20, rate_limit_per_host=100.0, respect_robots=False
    )
    result = load_web_docs_as_documents("https://main.example/docs/", limits=limits)
    urls = {d.metadata["source_url"] for d in result.documents}
    assert urls == {"https://main.example/docs/a.md"}


# ---------------------------------------------------------------------------
# Indexer end-to-end with mocked loader
# ---------------------------------------------------------------------------


def test_index_web_upserts_into_chroma(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """index_web() wires loader → chunker → chroma and writes web metadata."""
    routes = {
        "https://idx.example/robots.txt": (404, "", {"content-type": "text/plain"}),
        "https://idx.example/llms.txt": (404, "", {"content-type": "text/plain"}),
        "https://idx.example/sitemap.xml": (
            404,
            "",
            {"content-type": "text/xml"},
        ),
        "https://idx.example/sitemap_index.xml": (
            404,
            "",
            {"content-type": "text/xml"},
        ),
        "https://idx.example/": (
            200,
            "<html><head><title>Home</title></head><body>"
            '<a href="/p1">p1</a><main><h1>Home</h1><p>'
            + ("home content " * 40)
            + "</p></main></body></html>",
            {"content-type": "text/html"},
        ),
        "https://idx.example/p1": (
            200,
            "<html><head><title>Page1</title></head><body><main><h1>P1</h1><p>"
            + ("page1 content " * 40)
            + "</p></main></body></html>",
            {"content-type": "text/html"},
        ),
    }
    _install_mock_transport(monkeypatch, routes)
    # Reduce crawl limits via env so _crawl_limits_from_config picks them up.
    monkeypatch.setenv("PINRAG_WEB_MAX_PAGES", "5")
    monkeypatch.setenv("PINRAG_WEB_MAX_DEPTH", "2")
    monkeypatch.setenv("PINRAG_WEB_RATE_LIMIT_PER_HOST", "100")
    monkeypatch.setenv("PINRAG_WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("PINRAG_WEB_PREFER_LLMS_TXT", "0")

    persist = str(tmp_path / "chroma")
    result: WebIndexResult = index_web(
        "https://idx.example/",
        persist_directory=persist,
        collection_name="test_web",
        embedding=_MockEmbeddings(),
    )
    assert result.host == "idx.example"
    assert result.document_id == "idx.example/"
    assert result.pages_indexed >= 1
    assert result.total_chunks > 0
    assert result.discovery == "crawl"

    from pinrag.vectorstore import get_chroma_store

    store = get_chroma_store(
        persist_directory=persist,
        collection_name="test_web",
        embedding=_MockEmbeddings(),
    )
    data = store._collection.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    assert metas, "expected at least one chunk written"
    assert all(m.get("document_type") == "web" for m in metas)
    assert all(m.get("document_id") == "idx.example/" for m in metas)
    assert any("idx.example" in (m.get("source_url") or "") for m in metas)

    # Re-index: chunk count should stay stable (upsert, not append).
    count_before = len(data.get("ids") or [])
    result2 = index_web(
        "https://idx.example/",
        persist_directory=persist,
        collection_name="test_web",
        embedding=_MockEmbeddings(),
    )
    data2 = store._collection.get(include=[])
    count_after = len(data2.get("ids") or [])
    assert count_after == count_before
    assert result2.total_chunks == result.total_chunks
