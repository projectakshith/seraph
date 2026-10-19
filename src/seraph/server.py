"""Local stdio MCP server for coding agents."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .context_tools import (
    change_impact as build_change_impact,
)
from .context_tools import (
    prepare_change_context as build_change_context,
)
from .context_tools import (
    search_worktree as search_worktree_index,
)
from .context_tools import (
    working_tree_status as read_worktree_status,
)
from .index import VersionedIndex
from .service import compare_versions as compare_index
from .service import find_dependencies as lookup_dependencies
from .service import find_symbol as lookup_symbol
from .service import search_index
from .working_tree import snapshot as working_tree_snapshot

mcp = FastMCP("Seraph")


def _repo(repo: str | None = None) -> Path:
    # Agents should pass the workspace path on every call. The env/cwd fallback
    # keeps manual MCP clients and older configurations usable.
    if repo:
        return Path(repo).expanduser().resolve()
    configured = os.environ.get("SERAPH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0:
        raise ValueError("not inside a Git repository; set SERAPH_REPO to the repository to search")
    return Path(top.stdout.strip())


def _store(repo: str | None = None) -> VersionedIndex:
    db = os.environ.get("SERAPH_DB")
    return VersionedIndex(_repo(repo), Path(db) if db else None)


def _results(query: str, version: str, limit: int, include_history: bool, repo: str | None) -> dict:
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    with _store(repo) as store:
        out = search_index(store, query, version, limit, include_history)
    for r in out["results"]:
        r["text"] = r["text"][:4000]
    return out


@mcp.tool()
def search_code(query: str, limit: int = 5, repo: str | None = None) -> dict:
    """Search code in a local Git repository. Pass its absolute path as repo on every call."""
    return _results(query, "HEAD", limit, False, repo)


@mcp.tool()
def search_at_version(query: str, version: str, limit: int = 5, repo: str | None = None) -> dict:
    """Find code at a Git commit, branch, or tag. Pass the local repository path as repo."""
    return _results(query, version, limit, False, repo)


@mcp.tool()
def search_history(query: str, limit: int = 5, repo: str | None = None) -> dict:
    """Search indexed history. Pass the local repository path as repo."""
    return _results(query, "HEAD", limit, True, repo)


@mcp.tool()
def find_symbol(name: str, version: str = "HEAD", repo: str | None = None) -> dict:
    """Look up a function, class or method. Pass the local repository path as repo."""
    with _store(repo) as store:
        out = lookup_symbol(store, name, version)
    for r in out["results"]:
        r["text"] = r["text"][:4000]
    return out


@mcp.tool()
def find_dependencies(
    symbol: str, version: str = "HEAD", direction: str = "out", max_depth: int = 3, limit: int = 20,
    repo: str | None = None,
) -> dict:
    """Walk the call/import/inheritance graph from a symbol; direction is out, in, or both."""
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be out, in, or both")
    if not 1 <= max_depth <= 6:
        raise ValueError("max_depth must be between 1 and 6")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    with _store(repo) as store:
        return lookup_dependencies(store, symbol, version, direction, max_depth=max_depth, limit=limit)


@mcp.tool()
def compare_versions(from_version: str, to_version: str, limit: int = 50, repo: str | None = None) -> dict:
    """Diff Git versions. Pass the local repository path as repo."""
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    with _store(repo) as store:
        return compare_index(store, from_version, to_version, limit)


@mcp.tool()
def index_repository(version: str = "HEAD", repo: str | None = None) -> dict:
    """Index one Git version. Pass the local repository path as repo."""
    with _store(repo) as store:
        start = time.perf_counter()
        stats = asdict(store.index_commit(version))
    return {**stats, "ms": round((time.perf_counter() - start) * 1000, 1)}


@mcp.tool()
def get_working_tree_status(repo: str | None = None) -> dict:
    """Show staged, unstaged and untracked source changes in the selected local repository."""
    with _store(repo) as store:
        return read_worktree_status(store)


@mcp.tool()
def search_worktree(query: str, limit: int = 5, repo: str | None = None) -> dict:
    """Search current on-disk source, including uncommitted and untracked files, using configured retrieval."""
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    with _store(repo) as store:
        return search_worktree_index(store, query, limit)


@mcp.tool()
def change_impact(
    symbols: list[str] | None = None,
    paths: list[str] | None = None,
    scope: str = "worktree",
    limit: int = 30,
    task: str | None = None,
    repo: str | None = None,
) -> dict:
    """Show statically resolved callers, importers and tests affected by symbols or changed paths."""
    if scope not in ("head", "worktree"):
        raise ValueError("scope must be 'head' or 'worktree'")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    with _store(repo) as store:
        return build_change_impact(store, symbols, paths, scope, limit, task)


@mcp.tool()
def prepare_change_context(
    task: str,
    scope: str = "worktree",
    max_chars: int = 12000,
    repo: str | None = None,
) -> dict:
    """Assemble task-relevant code, repo map, change impact and Git history; the agent manages token use."""
    if scope not in ("head", "worktree"):
        raise ValueError("scope must be 'head' or 'worktree'")
    if not 1000 <= max_chars <= 30000:
        raise ValueError("max_chars must be between 1000 and 30000")
    with _store(repo) as store:
        return build_change_context(store, task, scope, max_chars)


@mcp.tool()
def get_repo_map(
    scope: str = "head",
    version: str = "HEAD",
    path_prefix: str | None = None,
    max_files: int = 30,
    max_chars: int = 3000,
    repo: str | None = None,
) -> dict:
    """Get a compact file-to-symbol map. Expand listed subdirectories with path_prefix or request larger limits."""
    if scope not in ("head", "worktree"):
        raise ValueError("scope must be 'head' or 'worktree'")
    if scope == "worktree" and version != "HEAD":
        raise ValueError("version selection is only supported with scope='head'")
    if not 1 <= max_files <= 200:
        raise ValueError("max_files must be between 1 and 200")
    if not 200 <= max_chars <= 16000:
        raise ValueError("max_chars must be between 200 and 16000")
    if path_prefix and (Path(path_prefix).is_absolute() or ".." in Path(path_prefix).parts):
        raise ValueError("path_prefix must be a repository-relative directory")
    prefix = path_prefix.strip("/") if path_prefix else None

    with _store(repo) as store:
        view = working_tree_snapshot(store) if scope == "worktree" else None
        commit = view.base_commit if view else store.index_commit(version).commit
        map_version = view.version if view else commit
        files: dict[str, set[str]] = {}
        subdirectories: set[str] = set()
        for chunk in (view.chunks if view else store.iter_chunks(commit)):
            if prefix and not (chunk.path == prefix or chunk.path.startswith(prefix + "/")):
                continue
            files.setdefault(chunk.path, set())
            if chunk.symbol:
                files[chunk.path].add(f"{chunk.kind} {chunk.symbol}")
            relative = chunk.path[len(prefix) + 1:] if prefix else chunk.path
            parts = relative.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                directory = "/".join(parts[:i])
                subdirectories.add(f"{prefix}/{directory}" if prefix else directory)

    lines = [
        f"{path}: {', '.join(sorted(symbols)) if symbols else '(file-level code)'}"
        for path, symbols in sorted(files.items())
    ]
    total_files = len(lines)
    selected = lines[:max_files]
    output: list[str] = []
    used = 0
    for line in selected:
        extra = len(line) + (1 if output else 0)
        if used + extra > max_chars:
            break
        output.append(line)
        used += extra
    shown = len(output)
    return {
        "resolved_commit": commit,
        "working_tree_snapshot": map_version if view else None,
        "scope": scope,
        "path_prefix": prefix,
        "map": "\n".join(output),
        "subdirectories": sorted(subdirectories)[:100],
        "subdirectories_truncated": len(subdirectories) > 100,
        "total_files": total_files,
        "shown_files": shown,
        "truncated": shown < total_files,
        "expand_hint": (
            "Call get_repo_map with a narrower path_prefix (choose one of subdirectories) "
            "or increase max_files/max_chars."
            if shown < total_files else None
        ),
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
