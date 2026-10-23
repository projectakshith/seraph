import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def isolated_seraph_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SERAPH_HOME", str(tmp_path / "seraph-home"))
    monkeypatch.delenv("SERAPH_TIER", raising=False)
    monkeypatch.delenv("SERAPH_CONFIG", raising=False)
