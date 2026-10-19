import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from seraph.config import SeraphConfig
from seraph.context_tools import prepare_change_context
from seraph.index import VersionedIndex
from seraph.service import IndexChunkStore, _apply_intent_ranking, _retrieval_mode, search_index


def test_explicit_search_intent_ranks_tests_and_implementations():
    test_hit = SimpleNamespace(file="tests/test_storage.py", symbol="test_storage", score=10.0)
    impl_hit = SimpleNamespace(file="src/storage.cpp", symbol="Storage::open", score=1.0)
    results = [test_hit, impl_hit]

    assert _apply_intent_ranking("find the implementation", results) == "implementation"
    assert results == [impl_hit, test_hit]
    assert _apply_intent_ranking("find the tests", results) == "tests"
    assert results == [test_hit, impl_hit]
    assert _apply_intent_ranking("how does storage behave", results) is None


def test_retrieval_mode_reports_active_graph_and_reranker():
    cfg = SeraphConfig()
    cfg.retrieval.use_graph_expansion = True
    cfg.retrieval.use_reranker = True
    cfg.retrieval.reranker_model = "test-ranker"

    assert _retrieval_mode(cfg) == "BM25+graph+rerank:test-ranker+fusion:rrf"


def test_change_context_keeps_search_evidence_when_compacting(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    source = tmp_path / "src"
    source.mkdir()
    (source / "storage.py").write_text(
        'def persist_session_state(session):\n'
        '    \"\"\"Persist session state to storage.\"\"\"\n'
        '    return write_record(session)\n'
    )
    for i in range(30):
        (source / f"module_{i}.py").write_text(
            f"def helper_{i}(value):\n    return normalize_value(value)\n"
        )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "add session storage")

    with VersionedIndex(tmp_path, tmp_path / "idx.sqlite") as index:
        pack = prepare_change_context(
            index, "find the implementation that persists session state to storage", scope="head", max_chars=1000
        )
        compact = prepare_change_context(
            index, "find the implementation that persists session state", scope="head", max_chars=400
        )
        with pytest.raises(ValueError, match="minimum evidence pack"):
            prepare_change_context(index, "find storage", scope="head", max_chars=256)

    assert pack["search_evidence"]
    assert pack["retrieval_mode"] == "BM25"
    assert pack["char_count"] <= 1000

    assert compact["search_evidence"]
    assert compact["truncated"]
    assert compact["char_count"] == len(json.dumps(compact, ensure_ascii=False)) <= 400


def test_explicit_search_intent_prioritizes_matching_results(repo):
    path, _, _ = repo
    (path / "tests").mkdir()
    (path / "tests" / "test_config.py").write_text("def test_parse_config():\n    pass\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "add config test")
    with VersionedIndex(path, path / "idx.sqlite") as index:
        impl = search_index(index, "find the implementation of parse config", limit=2, cfg=SeraphConfig())
        tests = search_index(index, "find tests for parse config", limit=2, cfg=SeraphConfig())
    assert impl["ranking_intent"] == "implementation"
    assert impl["results"][0]["path"] == "config.py"
    assert tests["ranking_intent"] == "tests"
    assert tests["results"][0]["path"] == "tests/test_config.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    (tmp_path / "config.py").write_text(
        "def parse_config(path):\n    return json.load(open(path))\n\n"
        "def normalize_input(text):\n    return text.strip().lower()\n"
    )
    (tmp_path / "net.py").write_text("def open_socket(host, port):\n    return socket.create_connection((host, port))\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    old = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "config.py").write_text(
        "def parse_config(path):\n    data = yaml.safe_load(open(path))\n    validate_schema(data)\n    return data\n\n"
        "def normalize_input(text):\n    return text.strip().lower()\n"
    )
    git(tmp_path, "commit", "-qam", "switch config to yaml with schema validation")
    return tmp_path, old, git(tmp_path, "rev-parse", "HEAD")


def test_store_maps_index_chunks(repo):
    path, old, new = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(old)
        index.index_commit(new)
        store = IndexChunkStore.from_index(index, "HEAD", include_history=True)
        head = list(store.iter_chunks("HEAD"))
        assert {c.symbol for c in head} == {"parse_config", "normalize_input", "open_socket"}
        assert all(c.version_id == new for c in head)
        assert len(list(store.iter_chunks(old[:10]))) == 3 and len(list(store.iter_chunks("*"))) == 6
        parse = next(c for c in head if c.symbol == "parse_config")
        old_parse = next(c for c in store.iter_chunks(old[:10]) if c.symbol == "parse_config")
        assert parse.lineage_id == old_parse.lineage_id is not None and store.get(parse.chunk_hash) == parse
        changes = store.changed_since(old)
        assert [store.get(o).symbol for o in changes.added] == ["parse_config"]
        assert len(changes.unchanged) == 2


def test_search_index_uses_pipeline_and_versions(repo):
    path, old, new = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        out = search_index(index, "parseConfig yaml schema", "HEAD", 3, cfg=SeraphConfig())
        assert out["resolved_commit"] == new
        top = out["results"][0]
        assert top["symbol"] == "parse_config" and top["commit"] == new and "lexical" in top["retrieval_scores"]
        assert {"occurrence_id", "path", "start_line", "end_line", "text", "content_hash"} <= set(top)
        old_out = search_index(index, "json load config", old, 3, cfg=SeraphConfig())
        assert old_out["results"][0]["commit"] == old and "json.load" in old_out["results"][0]["text"]
        hist = search_index(index, "json load config", "HEAD", 5, include_history=True, cfg=SeraphConfig())
        assert {r["commit"] for r in hist["results"]} == {old, new}
        assert search_index(index, "   ", "HEAD", 3, cfg=SeraphConfig())["results"] == []


def test_history_search_collapses_identical_code_across_versions(repo):
    path, old, new = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(old)
        index.index_commit(new)
        out = search_index(index, "normalize input text", "HEAD", 5, include_history=True, cfg=SeraphConfig())
        changed = search_index(index, "parse config yaml json", "HEAD", 5, include_history=True, cfg=SeraphConfig())
    normalize = [r for r in out["results"] if r["symbol"] == "normalize_input"]
    assert len(normalize) == 1
    assert sorted(normalize[0]["versions"]) == sorted([old, new])
    parse = [r for r in changed["results"] if r["symbol"] == "parse_config"]
    assert {r["commit"] for r in parse} == {old, new}
