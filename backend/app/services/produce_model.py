"""Load the separate produce-type bundle and identify produce in one image.

This is independent of the freshness bundle: its own weights, class mapping,
preprocessing and version. Softmax probabilities here describe only how the
12 trained produce classes compare. The model has no "not produce" class and
cannot detect unsupported items, so low confidence is handled as a request for
the user to confirm, never as proof that the item is unsupported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from app.core.config import Settings
from app.services.produce import ProduceIdentity, SOURCE_MODEL
from produce.model import build_produce_model
from produce.preprocess import image_to_tensor

logger = logging.getLogger("freshnet.produce_model")

SCHEMA = "freshnet.produce.deployment.v1"
REQUIRED_FILES = (
    "weights.pt",
    "architecture.json",
    "class_mapping.json",
    "preprocessing.json",
    "metadata.json",
    "checksums.json",
)


class ProduceBundleError(RuntimeError):
    """The produce bundle is missing, corrupt, a smoke artifact, or incompatible."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProduceBundleError(f"Unreadable produce bundle file: {path.name}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_produce_bundle(directory: Path) -> dict[str, Any]:
    bundle = Path(directory)
    if not bundle.is_dir():
        raise ProduceBundleError(f"Produce bundle directory not found: {directory}")
    checksums = _read_json(bundle / "checksums.json")
    files = checksums.get("files")
    if not isinstance(files, dict) or not files:
        raise ProduceBundleError("checksums.json has no file entries.")
    for name in REQUIRED_FILES:
        if name != "checksums.json" and name not in files:
            raise ProduceBundleError(f"Checksum manifest is missing {name}.")
    for name, expected in files.items():
        if name == "checksums.json" or Path(name).name != name:
            raise ProduceBundleError(f"Refusing checksum entry {name!r}.")
        asset = bundle / name
        if not asset.is_file():
            raise ProduceBundleError(f"Missing produce bundle file: {name}")
        if _sha256(asset) != expected:
            raise ProduceBundleError(f"Checksum mismatch for {name}.")
    metadata = _read_json(bundle / "metadata.json")
    architecture = _read_json(bundle / "architecture.json")
    mapping = _read_json(bundle / "class_mapping.json")
    preprocessing = _read_json(bundle / "preprocessing.json")
    if metadata.get("bundle_schema_version") != SCHEMA:
        raise ProduceBundleError(f"Unsupported produce bundle schema: {metadata.get('bundle_schema_version')!r}")
    if metadata.get("smoke_test") is not False:
        raise ProduceBundleError("Refusing a smoke artifact. There is no fallback produce model.")
    if architecture.get("architecture") != "efficientnet_b0":
        raise ProduceBundleError("Produce bundle architecture is not efficientnet_b0.")
    if architecture.get("reconstruct_with_pretrained_download") is not False:
        raise ProduceBundleError("Produce bundle requested an ImageNet download. Refusing.")
    order = mapping.get("class_order")
    index = mapping.get("class_to_index")
    if (
        not isinstance(order, list)
        or not order
        or index != {name: i for i, name in enumerate(order)}
        or mapping.get("num_classes") != len(order)
        or architecture.get("num_classes") != len(order)
    ):
        raise ProduceBundleError("Incompatible produce class mapping.")
    if preprocessing.get("eval_deterministic") is not True or preprocessing.get("augmentation_at_inference") is not False:
        raise ProduceBundleError("Produce bundle preprocessing is not the deterministic eval path.")
    confirmation = metadata.get("confirmation")
    if not isinstance(confirmation, dict) or "enabled" not in confirmation:
        raise ProduceBundleError("Produce bundle metadata has no confirmation rule.")
    return {
        "bundle": bundle,
        "metadata": metadata,
        "architecture": architecture,
        "class_mapping": mapping,
        "preprocessing": preprocessing,
    }


@dataclass
class LoadedProduceBundle:
    directory: Path
    model: torch.nn.Module
    device: torch.device
    model_version: str
    class_order: list[str]
    preprocessing: dict[str, Any]
    metadata: dict[str, Any]
    confirmation_enabled: bool
    confirmation_threshold: float | None


def load_produce_bundle(directory: Path, device: str = "cpu") -> LoadedProduceBundle:
    checked = verify_produce_bundle(directory)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_device = torch.device(device)
    architecture = checked["architecture"]
    order = list(checked["class_mapping"]["class_order"])
    model = build_produce_model(
        len(order), {"dropout": float(architecture.get("dropout", 0.3))}, pretrained=False
    )
    weights = torch.load(checked["bundle"] / "weights.pt", map_location="cpu", weights_only=False)
    if "state_dict" not in weights:
        raise ProduceBundleError("weights.pt has no state_dict.")
    incompatible = model.load_state_dict(weights["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ProduceBundleError(f"Strict weight load failed: {incompatible}")
    model.to(torch_device)
    model.eval()
    confirmation = checked["metadata"]["confirmation"]
    threshold = confirmation.get("threshold")
    enabled = bool(confirmation["enabled"])
    if enabled and not isinstance(threshold, (int, float)):
        raise ProduceBundleError("Confirmation is enabled but the threshold is missing.")
    return LoadedProduceBundle(
        directory=checked["bundle"],
        model=model,
        device=torch_device,
        model_version=str(checked["metadata"]["model_version"]),
        class_order=order,
        preprocessing=checked["preprocessing"],
        metadata=checked["metadata"],
        confirmation_enabled=enabled,
        confirmation_threshold=float(threshold) if enabled else None,
    )


def predict_probabilities(bundle: LoadedProduceBundle, image: Any) -> dict[str, Any]:
    """Run the produce model on one decoded RGB image with the bundle's own preprocessing."""
    tensor = image_to_tensor(image, bundle.preprocessing).unsqueeze(0).to(bundle.device)
    with torch.inference_mode():
        logits = bundle.model(tensor)
    logits_cpu = logits.detach().float().cpu()
    if logits_cpu.shape != (1, len(bundle.class_order)) or not torch.isfinite(logits_cpu).all():
        raise ProduceBundleError(f"Unexpected produce logits: {tuple(logits_cpu.shape)}")
    probabilities = F.softmax(logits_cpu, dim=1)[0]
    index = int(logits_cpu.argmax(dim=1).item())
    return {
        "index": index,
        "produce_id": bundle.class_order[index],
        "confidence": float(probabilities[index]),
        "logits": [float(v) for v in logits_cpu[0].tolist()],
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(bundle.class_order)},
    }


class BundleProduceIdentifier:
    """`ProduceIdentifier` backed by a verified produce bundle."""

    def __init__(self, bundle: LoadedProduceBundle) -> None:
        self.bundle = bundle
        self._lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return self.bundle.model_version

    @property
    def class_order(self) -> list[str]:
        return self.bundle.class_order

    @property
    def confirmation_threshold(self) -> float | None:
        return self.bundle.confirmation_threshold

    def identify(self, image: Any) -> ProduceIdentity:
        with self._lock:
            result = predict_probabilities(self.bundle, image)
        confidence = result["confidence"]
        threshold = self.bundle.confirmation_threshold
        needs_confirmation = self.bundle.confirmation_enabled and threshold is not None and confidence < threshold
        return ProduceIdentity(
            produce_id=result["produce_id"],
            label=result["produce_id"],
            source=SOURCE_MODEL,
            model_version=self.bundle.model_version,
            probabilities=result["probabilities"],
            confidence=confidence,
            confirmation_required=needs_confirmation,
            confirmation_threshold=threshold,
        )


def load_produce_identifier(settings: Settings) -> BundleProduceIdentifier | None:
    """Load the identifier, or return None (manual selection only) if it cannot be loaded."""
    if not settings.produce_bundle_dir.strip():
        logger.warning("Automatic produce identification disabled: FRESHNET_PRODUCE_BUNDLE_DIR is empty.")
        return None
    try:
        bundle = load_produce_bundle(settings.produce_bundle_directory, device=settings.device)
    except ProduceBundleError as exc:
        logger.error("Produce bundle not loaded; manual selection only: %s", exc)
        return None
    except Exception:
        logger.exception("Produce bundle failed to load; manual selection only.")
        return None
    logger.info("Produce bundle loaded", extra={"model_version": bundle.model_version, "device": str(bundle.device)})
    return BundleProduceIdentifier(bundle)
