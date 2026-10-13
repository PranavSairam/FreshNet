from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_file = tmp_path / "freshnet.sqlite"
    checkpoint = tmp_path / "missing.pt"
    monkeypatch.setenv("FRESHNET_SQLITE_PATH", str(db_file))
    monkeypatch.setenv("FRESHNET_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("FRESHNET_BUNDLE_DIR", str(tmp_path / "missing-bundle"))
    monkeypatch.setenv("FRESHNET_DEVICE", "cpu")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
