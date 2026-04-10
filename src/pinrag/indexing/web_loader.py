"""Load web documentation sites into LangChain Documents for RAG indexing.

v1 scope: pure-Python, `httpx` + `beautifulsoup4` + `trafilatura`, no JS rendering,
no pluggable backend. Discovery is tried in order: ``llms.txt`` / ``llms-full.txt``,
then ``sitemap.xml`` (+ ``robots.txt`` sitemap hints), then scoped BFS from the seed.

Scope boundary is locked to the seed's host + directory path prefix. Subdomains are
dropped. Anything outside the prefix is dropped. See
``notes/web-docs-indexing-research.md`` for the rationale.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from markdownify import markdownify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

_NOISE_SUFFIXES = (
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".rar",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp4",
    ".webm",
    ".mp3",
    ".wav",
    ".ogg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".css",
    ".js",
    ".json",
    ".xml",
    ".csv",
    ".exe",
    ".dmg",
    ".iso",
)

# Tracking params we strip from every URL before dedup.
_NOISY_QUERY_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    }
)


def normalize_url(url: str) -> str:
    """Return a canonical form of ``url`` for scope checks and deduplication.

    - Drops the fragment (``#section``).
    - Strips tracking query params.
    - Lowercases the scheme and host.
    - Collapses ``//`` runs in the path.
    - Leaves the path case intact (URLs are case-sensitive after the host).
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip()
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    query = parsed.query
    if query:
        kept = [
            kv
            for kv in query.split("&")
            if kv and kv.split("=", 1)[0] not in _NOISY_QUERY_PARAMS
        ]
        query = "&".join(kept)
    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def _seed_prefix(seed_url: str) -> tuple[str, str]:
    """Return ``(host, dir_prefix)`` for a seed URL.

    ``dir_prefix`` is the path up to and including the last ``/``. For a seed that
    already points at a directory (ends in ``/``) we use it directly; for a seed
    that points at a file (``.../pico-series.html``) we drop the file name. The
    empty path is treated as ``/``.
    """
    parsed = urlparse(seed_url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or "/"
    if not path.endswith("/"):
        # Strip to last slash. If no slash, treat as root.
        idx = path.rfind("/")
        path = path[: idx + 1] if idx >= 0 else "/"
    if not path:
        path = "/"
    return host, path


def same_scope(candidate: str, seed_host: str, seed_prefix: str) -> bool:
    """Return True if ``candidate`` is same-host and under ``seed_prefix``.

    No subdomain crossing: ``candidate`` host must equal ``seed_host`` exactly.
    """
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.netloc or "").lower()
    if host != seed_host:
        return False
    path = parsed.path or "/"
    return path.startswith(seed_prefix)


def is_noise_url(url: str) -> bool:
    """Return True for URLs that should never be indexed (binary assets, archives)."""
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    for suffix in _NOISE_SUFFIXES:
        if path.endswith(suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebLoadResult:
    """Result of discovering and fetching pages from a documentation site."""

    seed_url: str
    host: str
    path_prefix: str
    documents: list[Document]
    discovery: str  # "llms_txt" | "sitemap" | "crawl"
    failed_pages: list[dict[str, str]] = field(default_factory=list)

    @property
    def document_id(self) -> str:
        return f"{self.host}{self.path_prefix}"


@dataclass
class CrawlLimits:
    """Runtime caps for discovery and fetching."""

    max_pages: int = 200
    max_depth: int = 5
    max_page_bytes: int = 1_048_576
    request_timeout: float = 20.0
    concurrency: int = 4
    rate_limit_per_host: float = 2.0
    user_agent: str = "pinrag/0.10 (+https://github.com/ndjordjevic/pinrag)"
    respect_robots: bool = True
    prefer_llms_txt: bool = True


# ---------------------------------------------------------------------------
# robots.txt + rate limiting
# ---------------------------------------------------------------------------


class _HostPolicy:
    """Per-host robots.txt check + simple token-bucket rate limit."""

    def __init__(self, host: str, rate_per_sec: float, respect_robots: bool):
        self.host = host
        self.rate = max(0.1, rate_per_sec)
        self.respect_robots = respect_robots
        self._min_gap = 1.0 / self.rate
        self._last_request_ts = 0.0
        self._robots: RobotFileParser | None = None
        self._robots_loaded = False
        self._lock = asyncio.Lock()

    async def load_robots(self, client: httpx.AsyncClient, scheme: str) -> list[str]:
        """Fetch ``/robots.txt`` once. Return ``Sitemap:`` URLs found in it."""
        if self._robots_loaded:
            return []
        self._robots_loaded = True
        if not self.respect_robots:
            return []
        url = f"{scheme}://{self.host}/robots.txt"
        try:
            resp = await client.get(url, timeout=10.0, follow_redirects=True)
        except Exception as e:
            logger.debug("robots.txt fetch failed for %s: %s", self.host, e)
            return []
        if resp.status_code != 200:
            return []
        body = resp.text
        rp = RobotFileParser()
        rp.parse(body.splitlines())
        self._robots = rp
        sitemaps: list[str] = []
        for line in body.splitlines():
            if line.lower().startswith("sitemap:"):
                val = line.split(":", 1)[1].strip()
                if val:
                    sitemaps.append(val)
        return sitemaps

    def allowed(self, url: str, user_agent: str) -> bool:
        if not self.respect_robots or self._robots is None:
            return True
        return self._robots.can_fetch(user_agent, url)

    async def throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_gap - (now - self._last_request_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.monotonic()


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------


@dataclass
class FetchedPage:
    """HTTP response payload after size cap enforcement."""

    url: str
    status: int
    content_type: str
    text: str
    byte_size: int


async def _fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: _HostPolicy,
    limits: CrawlLimits,
) -> FetchedPage | None:
    """Fetch a URL with rate limit + size cap. Returns ``None`` on any error."""
    if policy.respect_robots and not policy.allowed(url, limits.user_agent):
        logger.info("robots.txt disallow: %s", url)
        return None
    await policy.throttle()
    try:
        resp = await client.get(url, timeout=limits.request_timeout, follow_redirects=True)
    except Exception as e:
        logger.debug("fetch failed: %s: %s", url, e)
        return None
    if resp.status_code != 200:
        logger.debug("non-200 for %s: %d", url, resp.status_code)
        return None
    content = resp.content or b""
    if len(content) > limits.max_page_bytes:
        logger.info(
            "skipping oversized page (%d bytes > cap): %s", len(content), url
        )
        return None
    ct = (resp.headers.get("content-type") or "").lower()
    try:
        text = resp.text
    except Exception:
        text = content.decode("utf-8", errors="replace")
    return FetchedPage(
        url=str(resp.url),
        status=resp.status_code,
        content_type=ct,
        text=text,
        byte_size=len(content),
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            return title
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return None


def _extract_markdown(page: FetchedPage) -> tuple[str, str | None]:
    """Return ``(markdown, title)`` for a fetched page, or ``("", None)`` on empty.

    Pages served as ``text/markdown`` (e.g. Mintlify's ``.md`` URLs, llms-full.txt
    sections) are returned as-is — trafilatura would just re-process markdown badly.
    """
    ct = page.content_type
    if "text/markdown" in ct or "text/plain" in ct and page.url.endswith(".md"):
        text = page.text.strip()
        # Use first heading line as title if any.
        title: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("# ").strip() or None
                break
        return text, title

    # HTML path.
    md = trafilatura.extract(
        page.text,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        include_formatting=True,
    )
    if md and md.strip():
        return md.strip(), _extract_title(page.text)

    # Fallback: try to find <main>/<article> and markdownify it.
    try:
        soup = BeautifulSoup(page.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
        )
        if main is not None:
            md2 = markdownify(str(main), heading_style="ATX").strip()
            if md2:
                return md2, _extract_title(page.text)
    except Exception as e:
        logger.debug("fallback markdownify failed for %s: %s", page.url, e)

    return "", None


# ---------------------------------------------------------------------------
# Discovery: llms.txt / llms-full.txt
# ---------------------------------------------------------------------------


_LLMS_LINK_RE = re.compile(r"\[(?P<title>[^\]]+)\]\((?P<url>[^)\s]+)\)")


def _parse_llms_txt_links(body: str) -> list[tuple[str, str]]:
    """Pull ``(url, title)`` tuples from an llms.txt markdown link list."""
    out: list[tuple[str, str]] = []
    for m in _LLMS_LINK_RE.finditer(body):
        url = m.group("url").strip()
        title = m.group("title").strip()
        if url.startswith("http://") or url.startswith("https://"):
            out.append((url, title))
    return out


async def _try_llms_txt(
    client: httpx.AsyncClient,
    *,
    scheme: str,
    host: str,
    seed_prefix: str,
    policy: _HostPolicy,
    limits: CrawlLimits,
) -> list[tuple[str, str]] | None:
    """Try ``/llms.txt``; return scoped ``[(url, title), ...]`` or ``None`` if absent."""
    url = f"{scheme}://{host}/llms.txt"
    page = await _fetch(client, url, policy=policy, limits=limits)
    if page is None:
        return None
    links = _parse_llms_txt_links(page.text)
    if not links:
        return None
    scoped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for link_url, title in links:
        norm = normalize_url(link_url)
        if norm in seen:
            continue
        if not same_scope(norm, host, seed_prefix):
            continue
        if is_noise_url(norm):
            continue
        seen.add(norm)
        scoped.append((norm, title))
    return scoped or None


# ---------------------------------------------------------------------------
# Discovery: sitemap.xml (+ robots.txt Sitemap: hints)
# ---------------------------------------------------------------------------


async def _fetch_sitemap(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: _HostPolicy,
    limits: CrawlLimits,
    depth: int = 0,
) -> list[str]:
    """Fetch one sitemap URL and return the ``<loc>`` URLs it contains.

    Recurses into sitemap index files (``<sitemapindex>``). Bounded by depth=3 to
    avoid sitemap bombs.
    """
    if depth > 3:
        return []
    page = await _fetch(client, url, policy=policy, limits=limits)
    if page is None:
        return []
    try:
        soup = BeautifulSoup(page.text, "lxml-xml")
    except Exception:
        soup = BeautifulSoup(page.text, "xml")
    # Sitemap index?
    sitemaps = soup.find_all("sitemap")
    if sitemaps:
        out: list[str] = []
        for sm in sitemaps:
            loc = sm.find("loc")
            if loc and loc.get_text(strip=True):
                sub = loc.get_text(strip=True)
                out.extend(
                    await _fetch_sitemap(
                        client, sub, policy=policy, limits=limits, depth=depth + 1
                    )
                )
        return out
    # Regular sitemap.
    urls: list[str] = []
    for url_tag in soup.find_all("url"):
        loc = url_tag.find("loc")
        if loc and loc.get_text(strip=True):
            urls.append(loc.get_text(strip=True))
    return urls


async def _try_sitemap(
    client: httpx.AsyncClient,
    *,
    scheme: str,
    host: str,
    seed_prefix: str,
    extra_sitemaps: list[str],
    policy: _HostPolicy,
    limits: CrawlLimits,
) -> list[str] | None:
    """Try ``sitemap.xml`` / ``sitemap_index.xml`` plus any hints from robots.txt."""
    candidates: list[str] = list(extra_sitemaps) + [
        f"{scheme}://{host}/sitemap.xml",
        f"{scheme}://{host}/sitemap_index.xml",
    ]
    seen_loc: set[str] = set()
    all_urls: list[str] = []
    for sm_url in candidates:
        urls = await _fetch_sitemap(client, sm_url, policy=policy, limits=limits)
        for u in urls:
            norm = normalize_url(u)
            if norm in seen_loc:
                continue
            if not same_scope(norm, host, seed_prefix):
                continue
            if is_noise_url(norm):
                continue
            seen_loc.add(norm)
            all_urls.append(norm)
    return all_urls or None


# ---------------------------------------------------------------------------
# Discovery: BFS fallback
# ---------------------------------------------------------------------------


def _extract_links(html: str, base_url: str) -> list[str]:
    """Return absolute ``<a href>`` targets from ``html``, resolved against ``base_url``."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        out.append(urljoin(base_url, href))
    return out


async def _bfs_crawl(
    client: httpx.AsyncClient,
    *,
    seed_url: str,
    host: str,
    seed_prefix: str,
    policy: _HostPolicy,
    limits: CrawlLimits,
    prefetched_pages: dict[str, FetchedPage],
) -> list[str]:
    """Scoped BFS from ``seed_url``. Populates ``prefetched_pages`` as a side-effect."""
    queue: list[tuple[str, int]] = [(normalize_url(seed_url), 0)]
    seen: set[str] = {queue[0][0]}
    out: list[str] = []
    while queue and len(out) < limits.max_pages:
        url, depth = queue.pop(0)
        page = await _fetch(client, url, policy=policy, limits=limits)
        if page is None:
            continue
        prefetched_pages[url] = page
        out.append(url)
        if depth >= limits.max_depth:
            continue
        for link in _extract_links(page.text, url):
            norm = normalize_url(link)
            if norm in seen:
                continue
            if not same_scope(norm, host, seed_prefix):
                continue
            if is_noise_url(norm):
                continue
            seen.add(norm)
            queue.append((norm, depth + 1))
    return out


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


async def _load_web_docs_async(
    seed_url: str,
    *,
    limits: CrawlLimits,
) -> WebLoadResult:
    norm_seed = normalize_url(seed_url)
    parsed = urlparse(norm_seed)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid web URL: {seed_url!r}")
    scheme = parsed.scheme
    host, seed_prefix = _seed_prefix(norm_seed)

    policy = _HostPolicy(
        host=host,
        rate_per_sec=limits.rate_limit_per_host,
        respect_robots=limits.respect_robots,
    )

    headers = {
        "User-Agent": limits.user_agent,
        "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(headers=headers) as client:
        # Always load robots.txt first (for Sitemap: hints even if we don't enforce).
        extra_sitemaps = await policy.load_robots(client, scheme)

        discovery: str = "crawl"
        urls_to_fetch: list[str] = []
        titles: dict[str, str] = {}
        prefetched: dict[str, FetchedPage] = {}

        # 1. llms.txt fast path
        if limits.prefer_llms_txt:
            llms = await _try_llms_txt(
                client,
                scheme=scheme,
                host=host,
                seed_prefix=seed_prefix,
                policy=policy,
                limits=limits,
            )
            if llms:
                discovery = "llms_txt"
                for url, title in llms[: limits.max_pages]:
                    urls_to_fetch.append(url)
                    if title:
                        titles[url] = title

        # 2. sitemap.xml (+ robots hints)
        if not urls_to_fetch:
            sitemap_urls = await _try_sitemap(
                client,
                scheme=scheme,
                host=host,
                seed_prefix=seed_prefix,
                extra_sitemaps=extra_sitemaps,
                policy=policy,
                limits=limits,
            )
            if sitemap_urls:
                discovery = "sitemap"
                urls_to_fetch = sitemap_urls[: limits.max_pages]

        # 3. BFS fallback (also populates prefetched pages).
        if not urls_to_fetch:
            discovery = "crawl"
            urls_to_fetch = await _bfs_crawl(
                client,
                seed_url=norm_seed,
                host=host,
                seed_prefix=seed_prefix,
                policy=policy,
                limits=limits,
                prefetched_pages=prefetched,
            )

        if not urls_to_fetch:
            return WebLoadResult(
                seed_url=norm_seed,
                host=host,
                path_prefix=seed_prefix,
                documents=[],
                discovery=discovery,
                failed_pages=[{"url": norm_seed, "error": "no pages discovered"}],
            )

        # Fetch + extract. Concurrency bounded by semaphore.
        semaphore = asyncio.Semaphore(max(1, limits.concurrency))
        document_id = f"{host}{seed_prefix}"
        documents: list[Document] = []
        failed: list[dict[str, str]] = []
        doc_lock = asyncio.Lock()

        async def process(url: str) -> None:
            async with semaphore:
                page = prefetched.get(url)
                if page is None:
                    page = await _fetch(client, url, policy=policy, limits=limits)
                if page is None:
                    async with doc_lock:
                        failed.append({"url": url, "error": "fetch failed"})
                    return
                md, html_title = _extract_markdown(page)
                if not md:
                    async with doc_lock:
                        failed.append({"url": url, "error": "empty extraction"})
                    return
                title = titles.get(url) or html_title
                meta = {
                    "document_id": document_id,
                    "document_type": "web",
                    "source": url,
                    "source_url": url,
                    "doc_bytes": page.byte_size,
                }
                if title:
                    meta["doc_title"] = title
                    meta["page_title"] = title
                async with doc_lock:
                    documents.append(Document(page_content=md, metadata=meta))

        await asyncio.gather(*(process(u) for u in urls_to_fetch))

    return WebLoadResult(
        seed_url=norm_seed,
        host=host,
        path_prefix=seed_prefix,
        documents=documents,
        discovery=discovery,
        failed_pages=failed,
    )


def load_web_docs_as_documents(
    seed_url: str,
    *,
    limits: CrawlLimits | None = None,
) -> WebLoadResult:
    """Discover, fetch, and extract pages from a documentation site.

    Blocking wrapper around the async implementation. Runs its own asyncio loop if
    called from sync code; uses ``asyncio.run`` which fails cleanly if called from
    inside a running loop — the indexer is sync by convention (matching github).
    """
    if limits is None:
        limits = CrawlLimits()
    try:
        return asyncio.run(_load_web_docs_async(seed_url, limits=limits))
    except RuntimeError as e:
        # Fallback: if already inside an event loop, run in a fresh thread.
        if "running event loop" in str(e).lower() or "cannot be called" in str(e).lower():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    asyncio.run, _load_web_docs_async(seed_url, limits=limits)
                )
                return fut.result()
        raise
