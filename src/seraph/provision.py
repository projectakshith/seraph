"""Install, download and verify a search tier; shared by `seraph setup` and the MCP tools."""

from __future__ import annotations

import importlib
import subprocess
import sys
import time

from seraph import tiers


def tier_status(name: str) -> dict:
    spec = tiers.TIERS[name]
    found = tiers.problems(name)
    return {
        "name": name,
        "summary": spec.summary,
        "ready": not found,
        "problems": found,
        "download_mb": spec.download_mb,
        "model_downloaded": tiers.model_cached(name),
        "first_search": spec.first_search,
        "setup_command": tiers.setup_command(name),
        "install_command": tiers.install_command(name),
    }


def status() -> dict:
    state = tiers.effective_tier()
    return {
        "active_tier": state.effective,
        "requested_tier": state.requested,
        "chosen_by": state.source,
        "problems": list(state.problems),
        "user_has_chosen": state.source != "default",
        "tiers": [tier_status(name) for name in tiers.TIERS],
        "settings_file": str(tiers.settings_path()),
        "guidance": (
            "Search is keyword-only until the user picks a tier. Ask them which they want: "
            "lite (no models), fast (62 MB, instant) or quality (about 6 GB, slow first search), "
            "then call set_retrieval_tier."
            if state.source == "default"
            else "The user has already chosen a tier; do not ask again unless they want to change it."
        ),
    }


def _step(name: str, ok: bool, detail: str = "") -> dict:
    return {"step": name, "ok": ok, "detail": detail}


def _pip_install(packages: tuple[str, ...]) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages], capture_output=True, text=True
    )
    importlib.invalidate_caches()
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return proc.returncode == 0, "\n".join(tail)


def _download(name: str) -> str:
    spec = tiers.model_spec(name)
    from huggingface_hub import snapshot_download

    snapshot_download(spec.hf_id, revision=spec.revision)
    return spec.hf_id


def _verify(name: str) -> str:
    from seraph.retrieval.semantic import _load_local

    spec = tiers.model_spec(name)
    t0 = time.perf_counter()
    model = _load_local(spec, "auto")
    vecs = model.encode_documents(["def add(a, b): return a + b", "class Cache: pass"])
    if vecs.shape != (2, model.dim):
        raise RuntimeError(f"unexpected embedding shape {vecs.shape}")
    return f"{spec.name} dim {model.dim}, loaded and encoded in {time.perf_counter() - t0:.1f}s"


def provision(name: str, install: bool = False, download: bool = True) -> dict:
    spec = tiers.TIERS[tiers.check_name(name)]
    steps: list[dict] = []

    def finish(ok: bool, nxt: str = "") -> dict:
        if ok:
            steps.append(_step("save", True, str(tiers.write_settings(tier=name))))
        return {"tier": name, "ok": ok, "steps": steps, "next": nxt, "status": status()}

    if spec.embedder is None:
        return finish(True)

    found = tiers.problems(name, require_model=False)
    if found and install:
        ok, detail = _pip_install(spec.pip)
        steps.append(_step("install", ok, detail))
        found = tiers.problems(name, require_model=False)
    if found:
        hint = tiers.install_command(name)
        return finish(False, f"{'; '.join(found)}. Fix with: {hint}")
    steps.append(_step("dependencies", True))

    if download:
        try:
            steps.append(_step("download", True, _download(name)))
        except Exception as exc:
            steps.append(_step("download", False, f"{type(exc).__name__}: {exc}"))
            return finish(False, "Model download failed; check the network and retry.")
    try:
        steps.append(_step("verify", True, _verify(name)))
    except Exception as exc:
        steps.append(_step("verify", False, f"{type(exc).__name__}: {exc}"))
        return finish(False, "The model downloaded but did not load; see the detail above.")
    return finish(True)
