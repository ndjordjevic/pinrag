"""Unit tests for GitHub repo loader, indexer, and MCP tool detection."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

from pinrag.indexing import index_github, load_github_repo_as_documents
from pinrag.indexing.github_loader import _parse_github_url
from pinrag.mcp.tools import _detect_source_format, _is_github_url


class _MockEmbeddings(Embeddings):
    """Returns fixed-dim vectors (1536) for testing without API key."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 1536


def test_is_github_url_detection() -> None:
    """Valid and invalid GitHub URLs."""
    assert _is_github_url("https://github.com/owner/repo") is True
    assert _is_github_url("https://github.com/owner/repo/tree/main") is True
    assert _is_github_url("github.com/owner/repo") is True
    assert _is_github_url("http://github.com/foo/bar") is True
    assert _is_github_url("https://www.github.com/foo/bar") is True
    assert _is_github_url("https://youtu.be/dQw4w9WgXcQ") is False
    assert _is_github_url("/path/to/file.pdf") is False
    assert _is_github_url("") is False
    assert _is_github_url("   ") is False
    assert _is_github_url("https://evil.com/https://github.com/foo/bar") is False
    assert _is_github_url("javascript:alert(1)") is False
    assert _is_github_url("https://github.com/owner") is False
    assert _is_github_url("https://github.com/../foo/bar") is False


def test_detect_source_format_github() -> None:
    """_detect_source_format returns 'github' for GitHub URLs."""
    assert _detect_source_format("https://github.com/owner/repo") == "github"
    assert (
        _detect_source_format("https://github.com/ndjordjevic/pinrag/tree/main")
        == "github"
    )
    assert _detect_source_format("github.com/foo/bar") == "github"


def test_parse_github_url() -> None:
    """Parse owner, repo, branch from GitHub URLs."""
    assert _parse_github_url("https://github.com/owner/repo") == (
        "owner",
        "repo",
        "main",
    )
    assert _parse_github_url("https://github.com/owner/repo/tree/develop") == (
        "owner",
        "repo",
        "develop",
    )
    assert _parse_github_url("github.com/foo/bar") == ("foo", "bar", "main")
    assert _parse_github_url("https://github.com/foo/repo.git") == (
        "foo",
        "repo",
        "main",
    )
    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        _parse_github_url("https://youtube.com/watch?v=abc")
    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        _parse_github_url("not-a-url")


def test_parse_github_url_branch_with_slash() -> None:
    """Branch names may contain slashes (e.g. feature/foo)."""
    assert _parse_github_url("https://github.com/owner/repo/tree/feature/foo") == (
        "owner",
        "repo",
        "feature/foo",
    )
    assert _parse_github_url(
        "https://github.com/owner/repo/tree/release/v1.0/patches"
    ) == (
        "owner",
        "repo",
        "release/v1.0/patches",
    )


def test_parse_github_url_branch_url_encoded() -> None:
    """Path segments are URL-decoded (e.g. feature%2Ffoo -> feature/foo)."""
    assert _parse_github_url(
        "https://github.com/owner/repo/tree/feature%2Ffoo%2Fbar"
    ) == (
        "owner",
        "repo",
        "feature/foo/bar",
    )


def test_is_default_text_file_special_basenames() -> None:
    """Dockerfile and .env.example match without relying on naive extension parsing."""
    from pinrag.indexing.github_loader import _is_default_text_file

    assert _is_default_text_file("Dockerfile") is True
    assert _is_default_text_file("path/to/Dockerfile") is True
    assert _is_default_text_file("path/to/.env.example") is True
    assert _is_default_text_file("foo.env.example") is False
    assert _is_default_text_file("README.md") is True


def test_parse_github_url_uses_default_branch_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PINRAG_GITHUB_DEFAULT_BRANCH is used when URL has no branch."""
    monkeypatch.setenv("PINRAG_GITHUB_DEFAULT_BRANCH", "develop")
    assert _parse_github_url("https://github.com/owner/repo") == (
        "owner",
        "repo",
        "develop",
    )
    monkeypatch.delenv("PINRAG_GITHUB_DEFAULT_BRANCH", raising=False)


def _mock_resp(status: int, json_data: object) -> object:
    """Create a mock response with status_code, json(), raise_for_status."""

    class _Resp:
        def __init__(self) -> None:
            self.status_code = status
            self._data = json_data

        def json(self) -> object:
            return self._data

        def raise_for_status(self) -> None:
            pass

    return _Resp()


@patch("pinrag.indexing.github_loader.requests.get")
def test_load_github_repo_documents_mocked(mock_get: object) -> None:
    """Mock GitHub API; verify Document metadata."""
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "tree123"}}})
    tree_resp = _mock_resp(
        200,
        {
            "tree": [
                {"type": "blob", "path": "README.md"},
                {"type": "blob", "path": "src/main.py"},
            ],
        },
    )
    readme_content = base64.b64encode(b"# Hello World").decode()
    main_content = base64.b64encode(b'print("hi")').decode()
    readme_file = _mock_resp(
        200, {"type": "file", "size": 20, "content": readme_content}
    )
    main_file = _mock_resp(200, {"type": "file", "size": 15, "content": main_content})

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "README.md" in url:
            return readme_file
        if "main.py" in url:
            return main_file
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    result = load_github_repo_as_documents("https://github.com/owner/repo")

    assert result.owner == "owner"
    assert result.repo == "repo"
    assert result.branch == "main"
    assert result.files_loaded == 2
    assert result.failed_files == []
    assert len(result.documents) == 2
    doc_ids = {d.metadata["document_id"] for d in result.documents}
    assert "owner/repo" in doc_ids
    sources = {d.metadata["source"] for d in result.documents}
    assert any("README.md" in s for s in sources)
    assert any("main.py" in s for s in sources)
    for d in result.documents:
        assert d.metadata["document_type"] == "github"
        assert d.metadata["repo"] == "owner/repo"
        assert d.metadata["branch"] == "main"
        assert "source" in d.metadata


@patch("pinrag.indexing.github_loader.requests.get")
def test_load_github_repo_excludes_binary_files(mock_get: object) -> None:
    """Tree with .png; verify excluded."""
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(
        200,
        {
            "tree": [
                {"type": "blob", "path": "img.png"},
                {"type": "blob", "path": "README.md"},
            ]
        },
    )
    readme_resp = _mock_resp(
        200, {"type": "file", "size": 10, "content": base64.b64encode(b"# Hi").decode()}
    )

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "README.md" in url:
            return readme_resp
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    result = load_github_repo_as_documents("https://github.com/x/y")

    assert result.files_loaded == 1
    assert "README.md" in result.documents[0].metadata["source"]


@patch("pinrag.indexing.github_loader.requests.get")
def test_load_github_repo_skips_oversized_files(mock_get: object) -> None:
    """Large file exceeds max_file_bytes; verify skipped."""
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(200, {"tree": [{"type": "blob", "path": "huge.bin"}]})
    huge_resp = _mock_resp(
        200,
        {
            "type": "file",
            "size": 1_000_000,
            "content": base64.b64encode(b"x" * 1000).decode(),
        },
    )

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "huge.bin" in url:
            return huge_resp
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    result = load_github_repo_as_documents("https://github.com/x/y", max_file_bytes=500)

    assert result.files_loaded == 0
    assert len(result.documents) == 0


@patch("pinrag.indexing.github_loader.requests.get")
def test_load_github_repo_records_per_file_fetch_failures(mock_get: object) -> None:
    """Non-200 per-file responses are listed in failed_files; other files still load."""
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(
        200,
        {
            "tree": [
                {"type": "blob", "path": "ok.py"},
                {"type": "blob", "path": "missing.py"},
            ],
        },
    )
    ok_resp = _mock_resp(
        200,
        {"type": "file", "size": 5, "content": base64.b64encode(b"x=1").decode()},
    )

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "ok.py" in url:
            return ok_resp
        if "missing.py" in url:
            return _mock_resp(500, {})
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    result = load_github_repo_as_documents("https://github.com/x/y")

    assert result.files_loaded == 1
    assert len(result.documents) == 1
    assert "ok.py" in result.documents[0].metadata["source"]
    assert len(result.failed_files) == 1
    assert result.failed_files[0]["path"] == "missing.py"
    assert "500" in result.failed_files[0]["error"]


@patch("pinrag.indexing.github_loader.requests.get")
def test_index_github_smoke(mock_get: object, tmp_path: Path) -> None:
    """Index mocked repo; verify Chroma store populated."""
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(200, {"tree": [{"type": "blob", "path": "README.md"}]})
    file_resp = _mock_resp(
        200,
        {"type": "file", "size": 20, "content": base64.b64encode(b"# PinRAG").decode()},
    )

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "README.md" in url:
            return file_resp
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    result = index_github(
        "https://github.com/ndjordjevic/pinrag",
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_github",
        embedding=_MockEmbeddings(),
    )

    assert result.owner == "ndjordjevic"
    assert result.repo == "pinrag"
    assert result.files_indexed == 1
    assert result.total_chunks > 0

    from pinrag.vectorstore import get_chroma_store

    store = get_chroma_store(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_github",
        embedding=_MockEmbeddings(),
    )
    docs = store.similarity_search("PinRAG", k=2)
    assert len(docs) > 0
    assert docs[0].metadata.get("document_type") == "github"
    assert docs[0].metadata.get("document_id") == "ndjordjevic/pinrag"
    assert "README.md" in (docs[0].metadata.get("source") or "")


@patch("pinrag.indexing.github_loader.requests.get")
def test_index_github_replaces_on_reindex(mock_get: object, tmp_path: Path) -> None:
    """Index same repo twice; verify no duplicate chunks."""
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(200, {"tree": [{"type": "blob", "path": "a.py"}]})
    file_resp = _mock_resp(
        200,
        {"type": "file", "size": 10, "content": base64.b64encode(b"x = 1").decode()},
    )

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "a.py" in url:
            return file_resp
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    persist = str(tmp_path / "chroma")
    coll = "test_replace"
    emb = _MockEmbeddings()

    r1 = index_github(
        "https://github.com/x/y",
        persist_directory=persist,
        collection_name=coll,
        embedding=emb,
    )
    from pinrag.vectorstore import get_chroma_store

    store = get_chroma_store(
        persist_directory=persist, collection_name=coll, embedding=emb
    )
    data1 = store._collection.get(include=[])
    count1 = len(data1.get("ids") or [])

    r2 = index_github(
        "https://github.com/x/y",
        persist_directory=persist,
        collection_name=coll,
        embedding=emb,
    )
    data2 = store._collection.get(include=[])
    count2 = len(data2.get("ids") or [])

    assert count2 == count1
    assert r2.total_chunks == r1.total_chunks


@patch("pinrag.indexing.github_loader.requests.get")
def test_index_github_parent_child_reindex_docstore_stable(
    mock_get: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-indexing in parent-child mode does not accumulate orphan parent docstore keys."""
    monkeypatch.setenv("PINRAG_USE_PARENT_CHILD", "true")

    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(200, {"tree": [{"type": "blob", "path": "a.py"}]})
    file_resp = _mock_resp(
        200,
        {"type": "file", "size": 10, "content": base64.b64encode(b"x = 1").decode()},
    )

    def side_effect(url: str, **kwargs: object) -> object:
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "a.py" in url:
            return file_resp
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    persist = str(tmp_path / "chroma")
    coll = "test_pc_reindex"
    emb = _MockEmbeddings()

    index_github(
        "https://github.com/x/y",
        persist_directory=persist,
        collection_name=coll,
        embedding=emb,
    )
    from pinrag.vectorstore.docstore import get_parent_docstore

    ds = get_parent_docstore(persist, coll)
    n1 = len(list(ds.yield_keys()))

    index_github(
        "https://github.com/x/y",
        persist_directory=persist,
        collection_name=coll,
        embedding=emb,
    )
    n2 = len(list(ds.yield_keys()))

    assert n2 == n1
    assert n1 > 0


@patch("pinrag.indexing.github_loader.requests.get")
def test_index_github_empty_documents_deletes_old_chunks(
    mock_get: object, tmp_path: Path
) -> None:
    """When all files are filtered out, re-index deletes existing chunks (no stale data)."""
    from pinrag.indexing.github_loader import GitHubLoadResult
    from pinrag.vectorstore import get_chroma_store

    # First call: return one file
    commit_resp = _mock_resp(200, {"commit": {"tree": {"sha": "t1"}}})
    tree_resp = _mock_resp(200, {"tree": [{"type": "blob", "path": "a.py"}]})
    file_resp = _mock_resp(
        200,
        {"type": "file", "size": 10, "content": base64.b64encode(b"x = 1").decode()},
    )

    call_count = 0

    def side_effect(url: str, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if "commits/" in url:
            return commit_resp
        if "git/trees/" in url:
            return tree_resp
        if "a.py" in url:
            return file_resp
        return _mock_resp(404, {})

    mock_get.side_effect = side_effect

    persist = str(tmp_path / "chroma")
    coll = "test_empty"
    emb = _MockEmbeddings()

    r1 = index_github(
        "https://github.com/x/y",
        persist_directory=persist,
        collection_name=coll,
        embedding=emb,
    )
    assert r1.total_chunks > 0

    # Second call: patch loader to return empty (repo now has no indexable files)
    with patch(
        "pinrag.indexing.github_indexer.load_github_repo_as_documents",
        return_value=GitHubLoadResult(
            owner="x",
            repo="y",
            branch="main",
            documents=[],
            files_loaded=0,
            total_chars=0,
        ),
    ):
        r2 = index_github(
            "https://github.com/x/y",
            persist_directory=persist,
            collection_name=coll,
            embedding=emb,
        )

    assert r2.files_indexed == 0
    assert r2.total_chunks == 0

    store = get_chroma_store(
        persist_directory=persist, collection_name=coll, embedding=emb
    )
    data = store._collection.get(where={"document_id": "x/y"}, include=[])
    assert len(data.get("ids") or []) == 0
