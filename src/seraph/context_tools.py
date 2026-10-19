"""Working-tree search and evidence-backed change context helpers for MCP."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from seraph.graph.builder import build_graph
from seraph.index import VersionedIndex
from seraph.service import IndexChunkStore, repo_graph, search_index, service_config
from seraph.types import Direction, EdgeKind
from seraph.working_tree import WorktreeSnapshot, snapshot

_TEXT_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".xml"}
_SKIP_DIRS = {".git", ".seraph", ".venv", "node_modules", "dist", "build", "vendor"}


def _text_candidates(index: VersionedIndex, terms: list[str], limit: int = 12) -> tuple[list[dict], list[dict]]:
    stop_words = {"about", "after", "agent", "also", "and", "are", "can", "code", "does", "for",
                  "from", "how", "into", "its", "new", "the", "this", "use", "using", "with"}
    tokens = []
    for value in terms:
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", split))
    terms = list(dict.fromkeys(t.casefold() for t in tokens if t.casefold() not in stop_words))[:30]
    if not terms:
        return [], []
    raw = index._git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode("utf-8", "replace")
    docs: list[dict] = []
    configs: list[dict] = []
    inspected = 0
    for rel in raw.split("\0"):
        if not rel or Path(rel).suffix.lower() not in _TEXT_SUFFIXES or _SKIP_DIRS.intersection(Path(rel).parts):
            continue
        inspected += 1
        if inspected > 1000:
            break
        path = index.repo / rel
        try:
            path.resolve().relative_to(index.repo)
            if path.stat().st_size > 256_000:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        matches = []
        line_numbers = []
        for number, line in enumerate(lines, 1):
            folded = line.casefold()
            found = [term for term in terms if term.casefold() in folded]
            if found:
                matches.extend(found)
                line_numbers.append(number)
        if not matches:
            continue
        item = {"path": rel, "lines": line_numbers[:8], "matching_terms": sorted(set(matches))[:8],
                "evidence": "text mention; candidate to inspect, not a resolved dependency"}
        if Path(rel).suffix.lower() in {".md", ".mdx", ".rst", ".txt"}:
            docs.append(item)
        else:
            configs.append(item)
    def rank(item: dict[str, Any]) -> tuple[int, str]:
        return -len(item["matching_terms"]), item["path"]

    return sorted(configs, key=rank)[:limit], sorted(docs, key=rank)[:limit]


_SNAPSHOT_PIPELINES: dict[tuple[str, str, str], Any] = {}


def _search_snapshot(
    index: VersionedIndex, view: WorktreeSnapshot, query: str, limit: int
) -> list[dict[str, Any]]:
    if not query.strip() or not view.chunks:
        return []
    from seraph.retrieval.pipeline import Pipeline

    cfg = service_config()
    config_hash = hashlib.sha1(cfg.model_dump_json().encode()).hexdigest()
    key = (str(index.repo), view.version, config_hash)
    if key not in _SNAPSHOT_PIPELINES:
        store = IndexChunkStore(view.chunks, view.version)
        graph = build_graph(view.chunks, view.refs) if cfg.retrieval.use_graph_expansion else None
        _SNAPSHOT_PIPELINES[key] = Pipeline.from_config(cfg, store, "HEAD", graph=graph)
        while len(_SNAPSHOT_PIPELINES) > 8:
            _SNAPSHOT_PIPELINES.pop(next(iter(_SNAPSHOT_PIPELINES)))
    pipe = _SNAPSHOT_PIPELINES[key]
    hits = pipe.search(query, top_k=limit).results
    source = {c.occurrence_id: c for c in view.chunks}
    return [
        {
            "score": round(hit.score, 4),
            **asdict(source[hit.chunk_hash]),
            "retrieval_scores": hit.retrieval_scores,
        }
        for hit in hits if hit.chunk_hash in source
    ]


def working_tree_status(index: VersionedIndex) -> dict[str, Any]:
    view = snapshot(index)
    return {
        "base_commit": view.base_commit,
        "snapshot": view.version,
        "changed_paths": view.changed_paths,
        "status": view.status,
        "indexed_files": len({c.path for c in view.chunks}),
        "mode": "ephemeral overlay for changed files; the overlay is not persisted; base HEAD uses the regular Seraph index",
    }


def search_worktree(index: VersionedIndex, query: str, limit: int = 5) -> dict[str, Any]:
    view = snapshot(index)
    return {
        "query": query,
        "base_commit": view.base_commit,
        "snapshot": view.version,
        "changed_paths": view.changed_paths,
        "retrieval_mode": _retrieval_mode(),
        "results": _search_snapshot(index, view, query, limit),
    }


def _retrieval_mode() -> str:
    retrieval = service_config().retrieval
    enabled = []
    if retrieval.use_bm25:
        enabled.append("BM25")
    if retrieval.use_dense:
        enabled.append(f"dense:{retrieval.dense_model}")
    if retrieval.use_structural:
        enabled.append("structural")
    if retrieval.use_graph_expansion:
        enabled.append("graph")
    if retrieval.use_evolution:
        enabled.append("evolution")
    if retrieval.use_reranker:
        enabled.append(f"rerank:{retrieval.reranker_model}")
    if len(enabled) > 1:
        enabled.append(f"fusion:{retrieval.fusion}")
    return "+".join(enabled) if enabled else "no retrieval views enabled"


def _impact(view: WorktreeSnapshot | None, index: VersionedIndex, symbols: list[str], paths: list[str],
            limit: int, extra_terms: list[str] | None = None):
    if view is None:
        commit = index.index_commit("HEAD").commit
        graph = repo_graph(index, commit)
        chunks = list(index.iter_chunks(commit))
    else:
        commit, graph, chunks = view.version, build_graph(view.chunks, view.refs), view.chunks

    changed_symbols: set[str] = set()
    unresolved: list[str] = []
    for name in symbols:
        resolved = graph.resolve(name)
        if resolved:
            changed_symbols.update(resolved)
        else:
            unresolved.append(name)
    for path in paths:
        changed_symbols.update(
            edge.dst for edge in graph.neighbors(path, {EdgeKind.DEFINES}, Direction.OUT)
        )

    by_id = {c.occurrence_id: c for c in chunks}
    related: dict[str, dict[str, Any]] = {}
    direct_edges = {EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS, EdgeKind.REFERENCES}
    for symbol_id in sorted(changed_symbols):
        for edge in graph.neighbors(symbol_id, direct_edges, Direction.IN):
            chunk_id = graph.chunk_for_symbol(edge.src)
            chunk = by_id.get(chunk_id) if chunk_id else None
            source_path = chunk.path if chunk else edge.src.split("::", 1)[0]
            if source_path in paths or source_path == symbol_id.split("::", 1)[0]:
                continue
            entry = related.setdefault(source_path, {"path": source_path, "relationships": set(), "evidence": "resolved static graph edge"})
            entry["relationships"].add(edge.kind.value)

    tests, dependents = [], []
    for path, entry in sorted(related.items()):
        item = {**entry, "relationships": sorted(entry["relationships"])}
        if re.search(r"(^|/)(tests?|specs?)(/|$)|(^|/)(test_|.*_test\.|.*\.test\.|.*\.spec\.)", path, re.I):
            tests.append(item)
        else:
            dependents.append(item)

    definitions = []
    public_candidates = []
    for symbol_id in sorted(changed_symbols):
        chunk_id = graph.chunk_for_symbol(symbol_id)
        chunk = by_id.get(chunk_id) if chunk_id else None
        if chunk:
            definition = {"symbol": symbol_id, "path": chunk.path, "kind": chunk.kind,
                          "lines": [chunk.start_line, chunk.end_line]}
            definitions.append(definition)
            short_name = symbol_id.rsplit("::", 1)[-1].split(".")[-1]
            if chunk.kind in {"class", "interface", "type", "enum"} and not short_name.startswith("_"):
                public_candidates.append({**definition, "evidence": "public-looking name; language visibility rules may differ"})

    term_values = list(extra_terms or []) + symbols + paths
    config_candidates, doc_candidates = _text_candidates(index, term_values)
    return {
        "resolved_version": commit,
        "changed_paths": sorted(set(paths))[:100],
        "changed_path_count": len(set(paths)),
        "requested_symbols": symbols,
        "unresolved_symbols": unresolved,
        "definitions": definitions[:limit],
        "public_surface_candidates": public_candidates[:limit],
        "direct_dependents": dependents[:limit],
        "related_tests": tests[:limit],
        "related_configs": config_candidates,
        "related_docs": doc_candidates,
        "coverage_note": "Relationships are static links resolved by Seraph's supported-language graph; absence of a link does not prove no impact.",
    }


def change_impact(index: VersionedIndex, symbols: list[str] | None = None,
                  paths: list[str] | None = None, scope: str = "worktree", limit: int = 30,
                  task: str | None = None):
    symbols, paths = symbols or [], paths or []
    view = snapshot(index) if scope == "worktree" else None
    if view is not None and not paths:
        paths = view.changed_paths
    return _impact(view, index, symbols, paths, limit, [task] if task else None)


def _repo_map(chunks, path_prefix: str | None, max_files: int, max_chars: int):
    files: dict[str, set[str]] = {}
    directories: set[str] = set()
    for chunk in chunks:
        if path_prefix and not (chunk.path == path_prefix or chunk.path.startswith(path_prefix + "/")):
            continue
        files.setdefault(chunk.path, set())
        if chunk.symbol:
            files[chunk.path].add(f"{chunk.kind} {chunk.symbol}")
        relative = chunk.path[len(path_prefix) + 1:] if path_prefix else chunk.path
        parts = relative.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            value = "/".join(parts[:i])
            directories.add(f"{path_prefix}/{value}" if path_prefix else value)
    lines = [f"{p}: {', '.join(sorted(s)) if s else '(file-level code)'}" for p, s in sorted(files.items())]
    output: list[str] = []
    used = 0
    for line in lines[:max_files]:
        extra = len(line) + (1 if output else 0)
        if used + extra > max_chars:
            break
        output.append(line)
        used += extra
    return {"map": "\n".join(output), "subdirectories": sorted(directories)[:100],
            "subdirectories_truncated": len(directories) > 100,
            "total_files": len(lines), "shown_files": len(output), "truncated": len(output) < len(lines)}


def prepare_change_context(index: VersionedIndex, task: str, scope: str = "worktree",
                           max_chars: int = 12000) -> dict[str, Any]:
    """Build a compact evidence pack; max_chars bounds output while the agent manages its token budget."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    view = snapshot(index) if scope == "worktree" else None
    if view is None:
        base = index.index_commit("HEAD").commit
        chunks = list(index.iter_chunks(base))
        paths = []
        search = search_index(index, task, base, limit=8)
        results = [{"path": h["path"], "symbol": h["symbol"], "kind": h["kind"],
                    "lines": [h["start_line"], h["end_line"]], "score": h["score"],
                    "text": h["text"][:700], "retrieval_scores": h["retrieval_scores"]}
                   for h in search["results"]]
    else:
        base, chunks, paths = view.base_commit, view.chunks, view.changed_paths
        results = [{"path": item["path"], "symbol": item["symbol"], "kind": item["kind"],
                    "lines": [item["start_line"], item["end_line"]], "score": item["score"],
                    "text": item["text"][:700], "retrieval_scores": item["retrieval_scores"]}
                   for item in _search_snapshot(index, view, task, 8)]
    symbols = [r["symbol"] for r in results if r["symbol"]]
    impact = _impact(view, index, symbols, paths, 20, [task])
    repo_map = _repo_map(chunks, None, 20, min(max_chars // 3, 4000))
    history: list[dict[str, str]] = []
    history_paths = sorted(set(paths + [r["path"] for r in results]))[:20]
    if history_paths:
        raw = index._git("log", "-5", "--format=%H%x00%s", "--", *history_paths).decode("utf-8", "replace")
        history = [{"commit": row.split("\0", 1)[0], "subject": row.split("\0", 1)[-1]}
                   for row in raw.splitlines() if "\0" in row]
    changed_path_count = len(paths)
    pack: dict[str, Any] = {"task": task[:1000], "task_truncated": len(task) > 1000,
                            "scope": scope, "base_commit": base,
                            "working_tree_snapshot": view.version if view else None,
                            "changed_paths": paths[:100], "changed_path_count": changed_path_count,
                            "changed_paths_truncated": changed_path_count > 100, "search_evidence": results,
                            "retrieval_mode": _retrieval_mode(),
                            "repo_map": repo_map, "change_impact": impact,
                            "recent_history": history,
                            "evidence_note": "Graph links are directly resolved static evidence; search hits are relevance-ranked candidates."}
    pack["truncated"] = False
    pack["expand_hint"] = "Use search_worktree, change_impact, or get_repo_map with a narrower path_prefix to expand evidence."

    def size() -> int:
        return len(json.dumps(pack, ensure_ascii=False))

    target_chars = max_chars - 24  # leave room for the final character-count field
    if size() > target_chars:
        pack["truncated"] = True
        pack["search_evidence"] = pack["search_evidence"][:5]
        for result in pack["search_evidence"]:
            result["text"] = result["text"][:240]
        pack["repo_map"]["map"] = pack["repo_map"]["map"][:max_chars // 6]
        pack["change_impact"]["direct_dependents"] = pack["change_impact"]["direct_dependents"][:8]
        pack["change_impact"]["related_tests"] = pack["change_impact"]["related_tests"][:8]
        pack["change_impact"]["related_configs"] = pack["change_impact"]["related_configs"][:5]
        pack["change_impact"]["related_docs"] = pack["change_impact"]["related_docs"][:5]
        pack["recent_history"] = pack["recent_history"][:3]
        pack["changed_paths"] = pack["changed_paths"][:30]
        while size() > target_chars:
            if pack["change_impact"]["direct_dependents"]:
                pack["change_impact"]["direct_dependents"].pop()
            elif pack["change_impact"]["related_tests"]:
                pack["change_impact"]["related_tests"].pop()
            elif pack["change_impact"]["related_configs"]:
                pack["change_impact"]["related_configs"].pop()
            elif pack["change_impact"]["related_docs"]:
                pack["change_impact"]["related_docs"].pop()
            elif pack["change_impact"]["definitions"]:
                pack["change_impact"]["definitions"].pop()
            elif pack["change_impact"]["public_surface_candidates"]:
                pack["change_impact"]["public_surface_candidates"].pop()
            elif pack["change_impact"]["unresolved_symbols"]:
                pack["change_impact"]["unresolved_symbols"].pop()
            elif pack["change_impact"]["requested_symbols"]:
                pack["change_impact"]["requested_symbols"].pop()
            elif pack["change_impact"]["changed_paths"]:
                pack["change_impact"]["changed_paths"].pop()
            elif pack["changed_paths"]:
                pack["changed_paths"].pop()
            elif pack["recent_history"]:
                pack["recent_history"].pop()
            elif pack["repo_map"]["subdirectories"]:
                pack["repo_map"]["subdirectories"].pop()
            elif pack["repo_map"]["map"]:
                text = pack["repo_map"]["map"]
                pack["repo_map"]["map"] = text[:len(text) // 2]
            elif len(pack["search_evidence"]) > 1:
                pack["search_evidence"].pop()
            elif pack["search_evidence"] and len(pack["search_evidence"][0]["text"]) > 80:
                result = pack["search_evidence"][0]
                result["text"] = result["text"][:max(80, len(result["text"]) // 2)]
            elif pack["search_evidence"] and pack["search_evidence"][0].get("retrieval_scores"):
                pack["search_evidence"][0]["retrieval_scores"] = {}
            elif len(pack["task"]) > 100:
                pack["task"] = pack["task"][:max(100, len(pack["task"]) - 100)]
                pack["task_truncated"] = True
            elif pack["change_impact"]["coverage_note"] == "Relationships are static links resolved by Seraph's supported-language graph; absence of a link does not prove no impact.":
                pack["change_impact"]["coverage_note"] = "Static graph coverage may be incomplete."
            elif pack["change_impact"]["coverage_note"]:
                pack["change_impact"]["coverage_note"] = ""
            elif pack["evidence_note"] == "Graph links are directly resolved static evidence; search hits are relevance-ranked candidates.":
                pack["evidence_note"] = "Evidence is ranked or statically resolved."
            elif pack["evidence_note"]:
                pack["evidence_note"] = ""
            elif pack["expand_hint"] == "Use search_worktree, change_impact, or get_repo_map with a narrower path_prefix to expand evidence.":
                pack["expand_hint"] = "Expand with targeted Seraph tools."
            elif pack["expand_hint"]:
                pack["expand_hint"] = ""
            else:
                break
    if size() > target_chars:
        pack["change_impact"] = {
            key: value
            for key, value in pack["change_impact"].items()
            if value or key == "resolved_version"
        }
        pack["repo_map"] = {
            "map": pack["repo_map"]["map"],
            "truncated": pack["repo_map"]["truncated"],
        }
        pack["recent_history"] = []
        pack["expand_hint"] = ""
    while size() > target_chars:
        if len(pack["search_evidence"]) > 1:
            pack["search_evidence"].pop()
        elif pack["search_evidence"] and pack["search_evidence"][0]["text"]:
            result = pack["search_evidence"][0]
            result["text"] = result["text"][:max(0, len(result["text"]) // 2)]
        elif pack["search_evidence"] and pack["search_evidence"][0].get("retrieval_scores"):
            pack["search_evidence"][0]["retrieval_scores"] = {}
        elif pack["repo_map"]["map"]:
            pack["repo_map"]["map"] = pack["repo_map"]["map"][:len(pack["repo_map"]["map"]) // 2]
        elif len(pack["task"]) > 20:
            pack["task"] = pack["task"][:max(20, len(pack["task"]) // 2)]
            pack["task_truncated"] = True
        else:
            break

    def exact_size() -> int:
        pack["char_count"] = 0
        for _ in range(4):
            count = len(json.dumps(pack, ensure_ascii=False))
            if pack["char_count"] == count:
                return count
            pack["char_count"] = count
        return len(json.dumps(pack, ensure_ascii=False))

    if exact_size() > max_chars:
        first = pack["search_evidence"][:1]
        compact_evidence = []
        for result in first:
            compact_evidence.append({
                "path": result["path"], "symbol": result["symbol"], "kind": result["kind"],
                "lines": result["lines"], "score": result["score"],
                "retrieval_scores": result["retrieval_scores"], "text": result["text"][:120],
            })
        pack = {
            "task": pack["task"][:240], "scope": scope, "base_commit": base,
            "changed_path_count": changed_path_count, "retrieval_mode": _retrieval_mode(),
            "search_evidence": compact_evidence, "truncated": True,
            "expand_hint": "Use targeted Seraph tools to expand this evidence.", "char_count": 0,
        }
        exact_size()
        while exact_size() > max_chars:
            evidence = pack["search_evidence"]
            if evidence and evidence[0]["text"]:
                text = evidence[0]["text"]
                evidence[0]["text"] = text[:len(text) // 2]
            elif evidence and evidence[0]["retrieval_scores"]:
                evidence[0]["retrieval_scores"] = {}
            elif pack["expand_hint"]:
                pack["expand_hint"] = ""
            elif len(pack["task"]) > 20:
                pack["task"] = pack["task"][:max(20, len(pack["task"]) // 2)]
            elif evidence and "kind" in evidence[0]:
                evidence[0].pop("kind")
            elif evidence and "score" in evidence[0]:
                evidence[0].pop("score")
            elif evidence and "lines" in evidence[0]:
                evidence[0].pop("lines")
            elif evidence and "symbol" in evidence[0]:
                evidence[0].pop("symbol")
            elif pack["changed_path_count"]:
                pack["changed_path_count"] = 0
            else:
                minimum = exact_size()
                raise ValueError(f"max_chars is too small for the minimum evidence pack ({minimum} characters)")
        exact_size()
    return pack
