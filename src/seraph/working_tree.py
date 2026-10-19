"""Ephemeral index overlay for the user's current working tree."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from seraph.graph.builder import Ref, extract_refs
from seraph.index import (
    MAX_FILE_BYTES,
    SUPPORTED_SUFFIXES,
    Chunk,
    VersionedIndex,
    _chunks,
    _digest,
    _language,
)


@dataclass
class WorktreeSnapshot:
    base_commit: str
    version: str
    changed_paths: list[str]
    status: list[str]
    chunks: list[Chunk]
    refs: list[tuple[str, Ref]]


def snapshot(index: VersionedIndex) -> WorktreeSnapshot:
    """Overlay staged, unstaged and untracked source files on indexed HEAD without persisting them."""
    base_commit = index.index_commit("HEAD").commit
    changed = set(index._git("diff", "--name-only", "--no-renames", "HEAD", "--").decode().splitlines())
    untracked = set(index._git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    changed |= untracked
    status = [line for line in index._git("status", "--short", "--untracked-files=all").decode().splitlines() if line]

    contents: dict[str, str] = {}
    for rel in sorted(changed):
        path = (index.repo / rel).resolve()
        try:
            path.relative_to(index.repo)
        except ValueError:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        raw = path.read_bytes()
        if len(raw) > MAX_FILE_BYTES or b"\0" in raw:
            continue
        contents[rel] = raw.decode("utf-8", "replace")

    digest = hashlib.sha256(base_commit.encode())
    for rel, text in sorted(contents.items()):
        digest.update(rel.encode())
        digest.update(text.encode("utf-8"))
    version = f"WORKTREE:{digest.hexdigest()}"

    base_chunks = [c for c in index.iter_chunks(base_commit) if c.path not in changed]
    chunks: list[Chunk] = []
    for c in base_chunks:
        oid = _digest("\0".join((version, c.path, c.symbol or "", str(c.start_line), c.content_hash)))
        chunks.append(replace(c, occurrence_id=oid, commit=version))

    refs = [(path, ref) for path, ref in index.iter_refs(base_commit) if path not in changed]
    for rel, text in contents.items():
        language = _language(rel)
        spans = _chunks(rel, text)
        for symbol, kind, start, end, part in spans:
            content_hash = _digest(language + "\0" + part)
            oid = _digest("\0".join((version, rel, symbol or "", str(start), content_hash)))
            chunks.append(Chunk(oid, version, rel, symbol, kind, start, end, content_hash, part, language))
        refs.extend((rel, ref) for ref in extract_refs(rel, text, spans))

    return WorktreeSnapshot(base_commit, version, sorted(changed), status, chunks, refs)
