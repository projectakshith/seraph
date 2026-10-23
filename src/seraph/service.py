"""Rank chunks from a `VersionedIndex` with the retrieval `Pipeline` (used by the CLI and MCP server)."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from seraph import index as vindex
from seraph import tiers
from seraph.config import SeraphConfig, load_config
from seraph.types import ChangeSet, Chunk

HISTORY = "*"


def to_chunk(c: vindex.Chunk, lineage_id: str | None = None) -> Chunk:
    """Ranking text is prefixed with path and symbol; results show the original `vindex.Chunk.text`."""
    return Chunk(
        chunk_hash=c.occurrence_id,
        text=f"{c.path} {c.symbol or ''}\n{c.text}",
        lang=c.language,
        file=c.path,
        symbol=c.symbol,
        kind=c.kind,
        start_line=c.start_line,
        end_line=c.end_line,
        version_id=c.commit,
        lineage_id=lineage_id,
    )


class IndexChunkStore:
    """`ChunkStore` over a snapshot of index chunks; `head` is the commit that "HEAD" refers to."""

    def __init__(
        self, chunks: Iterable[vindex.Chunk], head: str, lineage: dict[str, str] | None = None
    ) -> None:
        self.head = head
        self._source: dict[str, vindex.Chunk] = {}
        self._chunks: dict[str, Chunk] = {}
        self._by_commit: dict[str, list[str]] = {}
        lineage = lineage or {}
        for c in chunks:
            self._source[c.occurrence_id] = c
            self._chunks[c.occurrence_id] = to_chunk(c, lineage.get(c.occurrence_id))
            self._by_commit.setdefault(c.commit, []).append(c.occurrence_id)

    @classmethod
    def from_index(cls, index: vindex.VersionedIndex, ref: str = "HEAD", include_history: bool = False) -> IndexChunkStore:
        head = index.resolve(ref)
        lineage: dict[str, str] = {}
        for commit in index.indexed_versions() if include_history else [head]:
            lineage.update(index.lineage_ids(commit))
        return cls(index.iter_chunks(ref, include_history), head, lineage)

    def _commit(self, version: str) -> str:
        if version == "HEAD":
            return self.head
        matches = [c for c in self._by_commit if c.startswith(version)]
        if len(matches) != 1:
            raise KeyError(f"version {version!r} is not in this snapshot")
        return matches[0]

    def iter_chunks(self, version: str = "HEAD") -> Iterator[Chunk]:
        if version == HISTORY:
            yield from self._chunks.values()
            return
        for oid in self._by_commit.get(self._commit(version), []):
            yield self._chunks[oid]

    def get(self, chunk_hash: str) -> Chunk:
        return self._chunks[chunk_hash]

    def source(self, chunk_hash: str) -> vindex.Chunk:
        return self._source[chunk_hash]

    def changed_since(self, version: str) -> ChangeSet:
        old = {self._source[o].content_hash for o in self._by_commit.get(self._commit(version), [])}
        head = self._by_commit.get(self.head, [])
        new = {self._source[o].content_hash for o in head}
        return ChangeSet(
            added=sorted(o for o in head if self._source[o].content_hash not in old),
            removed=sorted(old - new),
            unchanged=sorted(o for o in head if self._source[o].content_hash in old),
        )

    def __len__(self) -> int:
        return len(self._chunks)


def _project_root(config_path: Path) -> Path:
    for parent in config_path.resolve().parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return config_path.resolve().parent


def _anchor_paths(cfg: SeraphConfig, config_path: Path) -> SeraphConfig:
    """Relative paths in a YAML config resolve against its project root, not the server's cwd."""
    root = _project_root(config_path)

    def anchored(value: str | None) -> str | None:
        return str(root / value) if value and not Path(value).is_absolute() else value

    cfg.cache_dir = anchored(cfg.cache_dir)
    cfg.retrieval.structural_model_path = anchored(cfg.retrieval.structural_model_path)
    cfg.retrieval.weight_model_path = anchored(cfg.retrieval.weight_model_path)
    return cfg


def service_config() -> SeraphConfig:
    """`SERAPH_CONFIG` (YAML) wins; otherwise the user's tier (`SERAPH_TIER` or `seraph setup`), else BM25."""
    path = os.environ.get("SERAPH_CONFIG")
    if path:
        return _anchor_paths(load_config(path), Path(path))
    return tiers.tier_config(tiers.effective_tier().effective)


def tier_notice() -> dict[str, str]:
    """Tells the agent when search is keyword-only and why, so it can ask the user."""
    if os.environ.get("SERAPH_CONFIG"):
        return {}
    state = tiers.effective_tier()
    notice: dict[str, str] = {"tier": state.effective}
    if state.requested != state.effective:
        notice["notice"] = (
            f"Tier '{state.requested}' is selected but unavailable ({'; '.join(state.problems)}); "
            f"using keyword search. Fix: {tiers.setup_command(state.requested)}"
        )
    elif state.source == "default":
        notice["notice"] = (
            "Keyword-only search (tier 'lite'). Embedding search is available: ask the user whether "
            "they want 'fast' (62 MB, instant) or 'quality' (about 6 GB, slow first search), then "
            "call set_retrieval_tier. Skip this if they only want keyword search."
        )
    return notice


_PIPELINES: dict[tuple, tuple[Any, IndexChunkStore]] = {}
_GRAPHS: dict[tuple[str, str], Any] = {}


def repo_graph(index: vindex.VersionedIndex, ref: str = "HEAD"):
    """The code graph of an indexed commit (indexing it first if needed); a commit's graph never changes."""
    from seraph.graph.builder import build_graph

    commit = index.index_commit(ref).commit
    key = (str(index.db_path), commit)
    if key not in _GRAPHS:
        _GRAPHS[key] = build_graph(index.iter_chunks(commit), index.iter_refs(commit))
    return _GRAPHS[key]


def _pipeline(index: vindex.VersionedIndex, commit: str, include_history: bool, cfg: SeraphConfig):
    from seraph.lineage import IndexVersionStore
    from seraph.retrieval.pipeline import Pipeline

    # Indexing an older commit can re-link lineage, so every indexed version is part of the key.
    scope = tuple(index.indexed_versions())
    key = (str(index.repo), commit, scope, include_history, hashlib.sha1(cfg.model_dump_json().encode()).hexdigest())
    if key not in _PIPELINES:
        store = IndexChunkStore.from_index(index, commit, include_history)
        pipe = None
        if len(store):
            version = HISTORY if include_history else "HEAD"
            graph = repo_graph(index, commit) if cfg.retrieval.use_graph_expansion else None
            pipe = Pipeline.from_config(cfg, store, version, graph=graph, versions=IndexVersionStore(index))
        _PIPELINES[key] = (pipe, store)
    return _PIPELINES[key]


def find_symbol(index: vindex.VersionedIndex, name: str, ref: str = "HEAD") -> dict[str, Any]:
    """Definitions matching `name` (`path::symbol`, `Class.method`, or a bare name) in one version."""
    graph = repo_graph(index, ref)
    commit = index.index_commit(ref).commit
    results = []
    for sid in graph.resolve(name):
        oid = graph.chunk_for_symbol(sid)
        if oid is None:
            continue
        results.append({"id": sid, **asdict(index.get_chunk(oid))})
    return {"query": name, "resolved_commit": commit, "results": results}


def find_dependencies(
    index: vindex.VersionedIndex,
    symbol: str,
    ref: str = "HEAD",
    direction: str = "out",
    kinds: list[str] | None = None,
    max_depth: int = 3,
    limit: int = 50,
) -> dict[str, Any]:
    """Dependency chains from each symbol matching `symbol` in one version."""
    from seraph.graph.traversal import find_dependencies as chains_from
    from seraph.types import Direction, EdgeKind

    graph = repo_graph(index, ref)
    commit = index.index_commit(ref).commit
    kind_set = {EdgeKind(k) for k in kinds} if kinds else None
    results = []
    for sid in graph.resolve(symbol):
        chains = chains_from(graph, sid, Direction(direction), kind_set, max_depth, limit)
        results.append({
            "symbol": sid,
            "chains": [
                {
                    "symbols": c.symbols,
                    "score": round(c.score, 4),
                    "edges": [{"src": e.src, "dst": e.dst, "kind": str(e.kind)} for e in c.edges],
                }
                for c in chains
            ],
        })
    return {"query": symbol, "resolved_commit": commit, "direction": direction, "results": results}


_CHANGE_ORDER = ("added", "modified", "renamed", "moved", "deleted")


def _trim_diff(old: str, new: str, a: str, b: str, max_lines: int) -> str:
    lines = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True), a, b))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... {len(lines) - max_lines} more diff lines\n"]
    return "".join(lines)


def compare_versions(
    index: vindex.VersionedIndex, a: str, b: str, limit: int = 200, max_diff_lines: int = 40
) -> dict[str, Any]:
    """Symbol, dependency and commit differences between two versions (each indexed if needed)."""
    from seraph.lineage import match_symbols, symbol_ref
    from seraph.types import EdgeKind

    ca, cb = index.index_commit(a).commit, index.index_commit(b).commit
    matches = [m for m in match_symbols(list(index.iter_chunks(ca)), list(index.iter_chunks(cb)))
               if m.change_type.value != "unchanged"]
    matches.sort(key=lambda m: (_CHANGE_ORDER.index(m.change_type.value), symbol_ref(m.new or m.old)))
    symbols = []
    for m in matches:
        c = m.new or m.old
        entry: dict[str, Any] = {"change_type": m.change_type.value, "symbol": symbol_ref(c), "path": c.path,
                                 "start_line": c.start_line, "end_line": c.end_line}
        if m.old is not None and m.new is not None:
            if symbol_ref(m.old) != symbol_ref(m.new):
                entry["previous"] = symbol_ref(m.old)
                entry["similarity"] = m.similarity
            if m.old.content_hash != m.new.content_hash:
                entry["diff"] = _trim_diff(m.old.text, m.new.text, f"{ca[:7]}:{symbol_ref(m.old)}",
                                           f"{cb[:7]}:{symbol_ref(m.new)}", max_diff_lines)
        symbols.append(entry)

    renamed = {symbol_ref(m.old): symbol_ref(m.new) for m in matches if m.old is not None and m.new is not None}
    old_edges = {(renamed.get(s, s), renamed.get(d, d), k) for s, d, k in repo_graph(index, ca).edges()
                 if k != EdgeKind.DEFINES}
    new_edges = {e for e in repo_graph(index, cb).edges() if e[2] != EdgeKind.DEFINES}

    def edge_list(edges: set) -> list[dict[str, str]]:
        return [{"src": s, "dst": d, "kind": str(k)} for s, d, k in sorted(edges)][:limit]

    by_path: dict[str, list[str]] = {}
    for entry in symbols:
        by_path.setdefault(entry["path"], []).append(entry["symbol"])
    commits = index.commits_between(ca, cb)
    for commit in commits:
        commit["symbols"] = [s for f in commit["files"] for s in by_path.get(f, [])]
    return {
        "from": ca,
        "to": cb,
        "summary": {t: sum(1 for m in matches if m.change_type.value == t) for t in _CHANGE_ORDER},
        "symbols": symbols[:limit],
        "dependencies": {"added": edge_list(new_edges - old_edges), "removed": edge_list(old_edges - new_edges)},
        "commits": commits,
    }


_IMPLEMENTATION_CUES = re.compile(
    r"\b(implement(?:ation|ed|s|ing)?|actual\s+(?:code|implementation)|source\s+code|"
    r"definition|handler|entry\s+point)\b",
    re.I,
)
_TEST_CUES = re.compile(r"\b(tests?|testing|coverage|fixtures?|assertions?)\b", re.I)
_TEST_PATH = re.compile(
    r"(^|/)(tests?|specs?)(/|$)|(^|/)(test_|.*_test\.|.*\.test\.|.*\.spec\.)",
    re.I,
)


def _search_intent(query: str) -> str | None:
    """Return an explicit test or implementation intent; leave broad behavior questions alone."""
    wants_tests = bool(_TEST_CUES.search(query))
    wants_implementation = bool(_IMPLEMENTATION_CUES.search(query))
    if wants_tests == wants_implementation:
        return None
    return "tests" if wants_tests else "implementation"


def _is_test_result(result) -> bool:
    path = result.file
    symbol = result.symbol or ""
    return bool(_TEST_PATH.search(path) or re.match(r"^(test|spec)[A-Z_]", symbol, re.I))


def _apply_intent_ranking(query: str, results: list) -> str | None:
    intent = _search_intent(query)
    if intent:
        results.sort(
            key=lambda result: (
                _is_test_result(result) != (intent == "tests"),
                -result.score,
            )
        )
    return intent


def _retrieval_mode(cfg: SeraphConfig) -> str:
    retrieval = cfg.retrieval
    views = []
    if retrieval.use_bm25:
        views.append("BM25")
    if retrieval.use_dense:
        views.append(f"dense:{retrieval.dense_model}")
    if retrieval.use_structural:
        views.append("structural")
    if retrieval.use_graph_expansion:
        views.append("graph")
    if retrieval.use_evolution:
        views.append("evolution")
    if retrieval.use_reranker:
        views.append(f"rerank:{retrieval.reranker_model}")
    if len(views) > 1:
        views.append(f"fusion:{retrieval.fusion}")
    return "+".join(views) if views else "no retrieval views enabled"


def search_index(
    index: vindex.VersionedIndex,
    query: str,
    ref: str = "HEAD",
    limit: int = 10,
    include_history: bool = False,
    cfg: SeraphConfig | None = None,
) -> dict[str, Any]:
    """Search an indexed version (indexing it first if needed) and return JSON-ready results."""
    from seraph.retrieval.dedup import lineage_history

    t0 = time.perf_counter()
    stats = index.index_commit(ref)
    explicit_cfg = cfg
    cfg = cfg or service_config()
    pipe, store = _pipeline(index, stats.commit, include_history, cfg)
    t1 = time.perf_counter()
    out: dict[str, Any] = {
        "query": query,
        "requested_version": ref,
        "resolved_commit": stats.commit,
        "index": {**asdict(stats), "ms": round((t1 - t0) * 1000, 1)},
        "results": [],
    }
    out["retrieval_mode"] = _retrieval_mode(cfg)
    if explicit_cfg is None:
        out.update(tier_notice())
    out["ranking_intent"] = _search_intent(query)
    if pipe is None or not query.strip() or limit < 1:
        return out
    intent = out["ranking_intent"]
    candidate_limit = limit * 4 if include_history or intent else limit
    resp = pipe.search(query, top_k=candidate_limit, include_history=include_history)
    _apply_intent_ranking(query, resp.results)
    results: list[dict[str, Any]] = []
    by_content: dict[str, dict[str, Any]] = {}
    for r in resp.results:
        source = store.source(r.chunk_hash)
        seen = by_content.get(source.content_hash) if include_history else None
        if seen is not None:
            seen["versions"].append(source.commit)
            continue
        hit = {"score": round(r.score, 4), **asdict(source), "lineage_id": r.lineage_id,
               "retrieval_scores": r.retrieval_scores}
        if include_history:
            hit["versions"] = [source.commit]
            hit["history"] = lineage_history(pipe.versions, r.lineage_id)
            by_content[source.content_hash] = hit
        if len(results) < limit:
            results.append(hit)
    out["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    out["query_type"] = resp.query_type
    out["results"] = results
    return out
