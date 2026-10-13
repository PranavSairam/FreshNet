import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.main import app
from app.services.inference import classify_image

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/deployment/v3/fused_epoch29_b962c650ed80"
SAMPLE = ROOT / "data/Dataset/Fresh/FreshBanana/b_f856.png"


@pytest.fixture
def ready_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FRESHNET_SQLITE_PATH", str(tmp_path / "freshnet.sqlite"))
    monkeypatch.setenv("FRESHNET_BUNDLE_DIR", str(BUNDLE))
    monkeypatch.setenv("FRESHNET_DEVICE", "cpu")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_missing_bundle_predicts_unavailable(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"
    assert "traceback" not in response.text.lower()
    assert "/" not in response.json()["message"]


def test_corrupt_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("notes.png", b"this is not an image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_image"


def test_oversized_upload_is_rejected(client: TestClient) -> None:
    payload = b"\x89PNG\r\n" + (b"x" * (8 * 1024 * 1024 + 1))
    response = client.post(
        "/api/predict",
        files={"file": ("big.png", payload, "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["code"] == "upload_too_large"


def test_transparency_and_orientation_and_animation_are_rejected(client: TestClient) -> None:
    rgba = Image.new("RGBA", (16, 16), (10, 20, 30, 128))
    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG")
    response = client.post("/api/predict", files={"file": ("alpha.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 415
    assert response.json()["code"] == "transparency_unsupported"

    oriented = Image.new("RGB", (16, 16), (1, 2, 3))
    exif = oriented.getexif()
    exif[274] = 6
    jpeg = io.BytesIO()
    oriented.save(jpeg, format="JPEG", exif=exif)
    response = client.post("/api/predict", files={"file": ("turned.jpg", jpeg.getvalue(), "image/jpeg")})
    assert response.status_code == 415
    assert response.json()["code"] == "exif_orientation"

    gif = io.BytesIO()
    Image.new("RGB", (16, 16), (4, 5, 6)).save(gif, format="GIF", save_all=True, append_images=[Image.new("RGB", (16, 16))])
    response = client.post("/api/predict", files={"file": ("anim.gif", gif.getvalue(), "image/gif")})
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_format"


def test_real_bundle_matches_direct_inference(ready_client: TestClient) -> None:
    data = SAMPLE.read_bytes()
    response = ready_client.post("/api/predict", files={"file": ("b_f856.png", data, "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    from app.services.runtime import _bundle

    assert _bundle is not None
    direct = classify_image(_bundle, SAMPLE)
    assert body["label"] == direct["label"] == "Fresh"
    assert body["label_index"] == direct["label_index"] == 0
    assert body["model_version"] == "freshnet-v3-fused-epoch29-b962c650ed80"
    assert body["timing_scope"] == "decode_validation_preprocessing_and_forward"
    assert body["elapsed_ms"] >= 0
    for name in ("Fresh", "Spoiled"):
        assert abs(body["probabilities"][name] - direct["probabilities"][name]) < 1e-5
    health = ready_client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["model"]["available"] is True
    assert health["model"]["model_version"] == body["model_version"]


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), (20, 40, 10)).save(buffer, format="PNG")
    return buffer.getvalue()
