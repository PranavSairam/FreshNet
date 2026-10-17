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


BUNDLE = ROOT / "artifacts/deployment/v3/fused_epoch29_b962c650ed80"


@pytest.fixture
def ready_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App with the verified v3 deployment bundle loaded."""
    monkeypatch.setenv("FRESHNET_SQLITE_PATH", str(tmp_path / "freshnet.sqlite"))
    monkeypatch.setenv("FRESHNET_BUNDLE_DIR", str(BUNDLE))
    # Existing tests exercise manual selection; the produce model is opt-in per test.
    monkeypatch.setenv("FRESHNET_PRODUCE_BUNDLE_DIR", str(tmp_path / "no-produce-bundle"))
    monkeypatch.setenv("FRESHNET_DEVICE", "cpu")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_file = tmp_path / "freshnet.sqlite"
    checkpoint = tmp_path / "missing.pt"
    monkeypatch.setenv("FRESHNET_SQLITE_PATH", str(db_file))
    monkeypatch.setenv("FRESHNET_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("FRESHNET_BUNDLE_DIR", str(tmp_path / "missing-bundle"))
    monkeypatch.setenv("FRESHNET_PRODUCE_BUNDLE_DIR", str(tmp_path / "no-produce-bundle"))
    monkeypatch.setenv("FRESHNET_DEVICE", "cpu")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
