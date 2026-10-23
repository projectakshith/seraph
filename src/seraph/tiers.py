"""Search tiers: lite (BM25), fast (static embeddings) and quality (BGE-Code-v1), chosen per user."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from seraph.config import EMBEDDERS, ModelSpec, SeraphConfig

DEFAULT_TIER = "lite"
HEAVY_DOWNLOAD_MB = 500


@dataclass(frozen=True)
class TierSpec:
    name: str
    summary: str
    embedder: str | None
    modules: tuple[str, ...]
    pip: tuple[str, ...]
    download_mb: int
    first_search: str
    max_versions: dict[str, int]


TIERS: dict[str, TierSpec] = {
    t.name: t
    for t in [
        TierSpec(
            name="lite",
            summary="Keyword search (BM25). No models, no downloads, instant.",
            embedder=None,
            modules=(),
            pip=(),
            download_mb=0,
            first_search="instant",
            max_versions={},
        ),
        TierSpec(
            name="fast",
            summary="BM25 + potion-code-16M static embeddings. 62 MB, CPU only, no torch.",
            embedder="potion-code-16m",
            modules=("model2vec", "huggingface_hub"),
            pip=("model2vec>=0.9,<0.10", "huggingface-hub>=0.34"),
            download_mb=62,
            first_search="about a second",
            max_versions={},
        ),
        TierSpec(
            name="quality",
            summary="BM25 + BGE-Code-v1 (1.5B). Best accuracy, about 6 GB, wants a GPU or Apple silicon.",
            embedder="bge-code-v1",
            modules=("torch", "transformers", "sentence_transformers", "einops"),
            pip=(
                "torch>=2.4",
                "transformers>=4.51,<5",
                "sentence-transformers>=3.4,<6",
                "einops>=0.8",
            ),
            download_mb=5800,
            first_search="tens of seconds while the repository is embedded",
            max_versions={"transformers": 4, "sentence-transformers": 5},
        ),
    ]
}


@dataclass(frozen=True)
class TierState:
    requested: str
    effective: str
    source: str
    problems: tuple[str, ...]


def home() -> Path:
    return Path(os.environ.get("SERAPH_HOME") or Path.home() / ".seraph")


def settings_path() -> Path:
    return home() / "settings.json"


def read_settings() -> dict:
    try:
        data = json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(**values) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**read_settings(), **values}, indent=2) + "\n")
    return path


def check_name(name: str) -> str:
    if name not in TIERS:
        raise ValueError(f"unknown tier {name!r}; choose one of {', '.join(TIERS)}")
    return name


def configured_tier() -> tuple[str, str]:
    env = os.environ.get("SERAPH_TIER")
    if env:
        return check_name(env.strip().lower()), "env"
    saved = read_settings().get("tier")
    if saved:
        return check_name(saved), "settings"
    return DEFAULT_TIER, "default"


def tier_config(name: str) -> SeraphConfig:
    spec = TIERS[check_name(name)]
    cfg = SeraphConfig(cache_dir=str(home() / "cache"))
    if spec.embedder:
        cfg.allow_local_models = True
        cfg.retrieval.use_dense = True
        cfg.retrieval.dense_model = spec.embedder
        cfg.retrieval.fusion = "weighted"
    return cfg


def model_spec(name: str) -> ModelSpec | None:
    embedder = TIERS[check_name(name)].embedder
    return EMBEDDERS[embedder] if embedder else None


def model_cached(name: str) -> bool:
    spec = model_spec(name)
    if spec is None:
        return True
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (root / f"models--{spec.hf_id.replace('/', '--')}" / "refs" / spec.revision).exists()


def _installed_major(dist: str) -> int | None:
    try:
        return int(metadata.version(dist).split(".", 1)[0])
    except (metadata.PackageNotFoundError, ValueError):
        return None


def problems(name: str, require_model: bool = True) -> list[str]:
    spec = TIERS[check_name(name)]
    found: list[str] = []
    missing = [m for m in spec.modules if importlib.util.find_spec(m) is None]
    if missing:
        found.append(f"missing Python packages: {', '.join(missing)}")
    for dist, ceiling in spec.max_versions.items():
        major = _installed_major(dist)
        if major is not None and major > ceiling:
            found.append(
                f"{dist} {major}.x is installed but {spec.embedder} needs {dist}<{ceiling + 1}; "
                "use a dedicated environment for this tier"
            )
    if require_model and spec.download_mb >= HEAVY_DOWNLOAD_MB and not model_cached(name):
        found.append(f"{spec.embedder} is not downloaded ({spec.download_mb} MB)")
    return found


def effective_tier() -> TierState:
    requested, source = configured_tier()
    found = problems(requested)
    return TierState(requested, DEFAULT_TIER if found else requested, source, tuple(found))


def install_command(name: str) -> str:
    spec = TIERS[check_name(name)]
    if not spec.pip:
        return ""
    prefix = Path(sys.prefix).as_posix()
    quoted = " ".join(f'"{p}"' for p in spec.pip)
    if "/uv/tools/" in prefix:
        return f'uv tool install --reinstall "seraph-mcp[{name}]"'
    if "/pipx/venvs/" in prefix:
        return f"pipx inject seraph-mcp {quoted}"
    return f"{Path(sys.executable).as_posix()} -m pip install {quoted}"


def setup_command(name: str) -> str:
    flags = " --install" if TIERS[check_name(name)].pip else ""
    return f"seraph setup --tier {name}{flags}"
