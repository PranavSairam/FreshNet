"""Process-wide model handle. The bundle is loaded once at startup."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.health import ModelStatus
from app.services.image_policy import validate_image_bytes
from app.services.inference import BundleError, LoadedBundle, classify_pil, load_bundle
from app.services.produce import ProduceIdentity, describe_result, identify_produce, unknown_identity

logger = logging.getLogger("freshnet.runtime")

_bundle: LoadedBundle | None = None
_unavailable_message = "The deployment bundle is not available."
_lock = threading.Lock()


def load_model(settings: Settings) -> None:
    global _bundle, _unavailable_message
    _bundle = None
    if not settings.bundle_dir.strip():
        _unavailable_message = "No deployment bundle is configured."
        logger.error("Prediction disabled: FRESHNET_BUNDLE_DIR is empty.")
        return
    try:
        loaded = load_bundle(settings.bundle_directory, device=settings.device)
    except BundleError:
        logger.exception("Deployment bundle failed to load.")
        _unavailable_message = "The deployment bundle could not be loaded."
        return
    _bundle = loaded
    _unavailable_message = ""
    logger.info(
        "Deployment bundle loaded",
        extra={"model_version": loaded.model_version, "device": str(loaded.device)},
    )


def current_model_status() -> ModelStatus:
    if _bundle is None:
        return ModelStatus(
            available=False,
            status="unavailable",
            message=_unavailable_message,
            architecture="fused",
            model_version="",
        )
    return ModelStatus(
        available=True,
        status="ready",
        message="Model loaded",
        architecture="fused",
        model_version=_bundle.model_version,
    )


def predict_bytes(data: bytes, produce: ProduceIdentity | None = None) -> dict[str, Any]:
    """Classify one image.

    `produce` is the already validated, user-selected identity (or None). It never
    influences the freshness model. The legacy top-level fields are kept unchanged;
    `produce`, `freshness`, `severity`, `guidance` and `combined_label` are additive.
    """
    started = time.perf_counter()
    image = validate_image_bytes(data)
    if _bundle is None:
        raise AppError("model_unavailable", _unavailable_message, 503)
    with _lock:
        try:
            result = classify_pil(_bundle, image)
        except Exception as exc:
            logger.exception("Inference failed.")
            raise AppError(
                "inference_failed",
                "The image could not be classified.",
                500,
            ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities = result["probabilities"]
    if not all(_finite(value) for value in probabilities.values()):
        raise AppError("inference_failed", "The image could not be classified.", 500)
    # Timing above covers the freshness path only. Identity and guidance are
    # deterministic lookups done afterwards.
    # The produce model gets the same validated image but its own preprocessing, and a
    # failure here can never change or block the freshness result computed above.
    identity, identification = identify_produce(produce or unknown_identity(), image)
    described = describe_result(result["label"], identity)
    return {
        "label": result["label"],
        "label_index": result["label_index"],
        "probabilities": probabilities,
        "model_version": result["model_version"],
        "elapsed_ms": round(elapsed_ms, 3),
        "timing_scope": "decode_validation_preprocessing_and_forward",
        "contract_version": "freshnet.predict.v2",
        "combined_label": described["combined_label"],
        "produce": described["produce"],
        "identification": identification,
        "freshness": {
            "label": result["label"],
            "class_index": result["label_index"],
            "probabilities": probabilities,
            "model_version": result["model_version"],
        },
        "severity": described["severity"],
        "guidance": described["guidance"],
    }


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
