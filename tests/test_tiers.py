import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seraph import provision, tiers
from seraph.config import EMBEDDERS
from seraph.retrieval import semantic
from seraph.service import service_config, tier_notice


@pytest.fixture(autouse=True)
def fresh_model_cache():
    semantic._LOADED.clear()


def hide_modules(monkeypatch, *names):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a, **k: None if n in names else real(n, *a, **k)
    )


def test_every_tier_embedder_is_registered():
    for spec in tiers.TIERS.values():
        assert spec.embedder is None or spec.embedder in EMBEDDERS


def test_lite_is_keyword_only_and_others_enable_the_dense_view():
    assert not tiers.tier_config("lite").retrieval.use_dense
    fast = tiers.tier_config("fast")
    assert fast.retrieval.use_dense and fast.retrieval.dense_model == "potion-code-16m"
    assert fast.allow_local_models
    quality = tiers.tier_config("quality")
    assert quality.retrieval.dense_model == "bge-code-v1"


def test_tier_cache_lives_under_the_seraph_home(tmp_path):
    assert Path(tiers.tier_config("fast").cache_dir) == tmp_path / "seraph-home" / "cache"


def test_choice_precedence_is_env_then_settings_then_default(monkeypatch):
    assert tiers.configured_tier() == ("lite", "default")
    tiers.write_settings(tier="fast")
    assert tiers.configured_tier() == ("fast", "settings")
    monkeypatch.setenv("SERAPH_TIER", "quality")
    assert tiers.configured_tier() == ("quality", "env")


def test_unknown_tier_is_rejected(monkeypatch):
    monkeypatch.setenv("SERAPH_TIER", "turbo")
    with pytest.raises(ValueError, match="unknown tier"):
        tiers.configured_tier()


def test_missing_packages_fall_back_to_lite_with_a_reason(monkeypatch):
    tiers.write_settings(tier="fast")
    hide_modules(monkeypatch, "model2vec")
    state = tiers.effective_tier()
    assert (state.requested, state.effective) == ("fast", "lite")
    assert "model2vec" in state.problems[0]
    assert not service_config().retrieval.use_dense
    assert "fast" in tier_notice()["notice"] and "unavailable" in tier_notice()["notice"]


def test_incompatible_transformers_major_is_reported(monkeypatch):
    monkeypatch.setattr(tiers, "_installed_major", lambda dist: 5 if dist == "transformers" else 4)
    monkeypatch.setattr(tiers, "model_cached", lambda name: True)
    hide_modules(monkeypatch)
    found = tiers.problems("quality")
    assert any("transformers<5" in p for p in found)


def test_heavy_tier_needs_its_model_downloaded(monkeypatch):
    monkeypatch.setattr(tiers, "model_cached", lambda name: False)
    monkeypatch.setattr(tiers, "_installed_major", lambda dist: 4)
    hide_modules(monkeypatch)
    assert any("not downloaded" in p for p in tiers.problems("quality"))
    assert not any("not downloaded" in p for p in tiers.problems("fast"))


def test_default_choice_prompts_the_agent_and_a_saved_choice_does_not():
    assert "ask the user" in tier_notice()["notice"]
    tiers.write_settings(tier="lite")
    assert "notice" not in tier_notice()


def test_status_tells_the_agent_whether_the_user_has_chosen():
    assert not provision.status()["user_has_chosen"]
    provision.provision("lite")
    info = provision.status()
    assert info["user_has_chosen"] and info["active_tier"] == "lite"


def test_provision_lite_saves_the_choice():
    result = provision.provision("lite")
    assert result["ok"] and tiers.read_settings()["tier"] == "lite"


def test_failed_provision_does_not_save_the_choice(monkeypatch):
    hide_modules(monkeypatch, "model2vec")
    result = provision.provision("fast", install=False)
    assert not result["ok"] and "model2vec" in result["next"]
    assert "tier" not in tiers.read_settings()


def test_provision_downloads_and_verifies_the_fast_tier(monkeypatch):
    monkeypatch.setattr(provision, "_download", lambda name: "minishlab/potion-code-16M")
    monkeypatch.setattr(provision, "_verify", lambda name: "ok")
    monkeypatch.setattr(tiers, "problems", lambda name, require_model=True: [])
    result = provision.provision("fast")
    assert result["ok"] and [s["step"] for s in result["steps"]] == ["dependencies", "download", "verify", "save"]
    assert tiers.read_settings()["tier"] == "fast"


def test_explicit_yaml_paths_resolve_against_the_project_root_not_the_cwd(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    cfg = root / "configs" / "x.yaml"
    cfg.write_text("retrieval:\n  structural_model_path: models/s.pkl\n")
    monkeypatch.setenv("SERAPH_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    loaded = service_config()
    assert loaded.retrieval.structural_model_path == str(root / "models" / "s.pkl")
    assert loaded.cache_dir == str(root / ".seraph_cache")


def test_static_embedder_normalizes_and_is_loaded_once(monkeypatch):
    loads = []

    class FakeStaticModel:
        embedding = np.zeros((10, 4))

        @classmethod
        def from_pretrained(cls, name, **kw):
            loads.append(name)
            return cls()

        def encode(self, texts, max_length=None):
            return np.array([[3.0, 4.0, 0.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "model2vec", SimpleNamespace(StaticModel=FakeStaticModel))
    spec = EMBEDDERS["potion-code-16m"]
    first = semantic._load_local(spec, "cpu")
    assert semantic._load_local(spec, "cpu") is first and loads == [spec.hf_id]
    vecs = first.encode_documents(["a", "b"])
    assert vecs.shape == (2, 4) and np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert first.dim == 4


def test_pyproject_extras_match_the_tier_pins():
    data = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert data["project"]["name"] == "seraph-mcp"
    assert data["project"]["scripts"]["seraph-mcp"] == "seraph.server:main"
    for name in ("fast", "quality"):
        assert sorted(extras[name]) == sorted(tiers.TIERS[name].pip)
