"""Load a verified FreshNet deployment bundle and classify one image.

This does not start the HTTP API. Pass the bundle directory explicitly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models import build_model
from training.data import eval_branch_tensors, open_rgb_image

REQUIRED_FILES = (
    "weights.pt",
    "architecture.json",
    "class_mapping.json",
    "preprocessing.json",
    "metadata.json",
    "checksums.json",
)


class BundleError(RuntimeError):
    """The bundle is missing, corrupt, or incompatible."""


@dataclass
class LoadedBundle:
    directory: Path
    model: torch.nn.Module
    device: torch.device
    model_version: str
    class_mapping: dict[str, int]
    preprocessing: dict[str, Any]
    metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Unreadable bundle file: {path.name}") from exc


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle(directory: Path) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        repo = Path(__file__).resolve().parents[3]
        path = repo / path
    if not path.is_dir():
        raise BundleError(f"Bundle directory not found: {directory}")
    return path


def verify_bundle_files(directory: Path) -> dict[str, Any]:
    bundle = _resolve_bundle(directory)
    checksums = _read_json(bundle / "checksums.json")
    files = checksums.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError("checksums.json has no file entries.")
    for name in REQUIRED_FILES:
        if name == "checksums.json":
            continue
        if name not in files:
            raise BundleError(f"Checksum manifest is missing {name}.")
    for name, expected in files.items():
        if name == "checksums.json" or Path(name).name != name:
            raise BundleError(f"Refusing checksum entry {name!r}.")
        asset = bundle / name
        if not asset.is_file():
            raise BundleError(f"Missing bundle file: {name}")
        actual = _sha256(asset)
        if actual != expected:
            raise BundleError(f"Checksum mismatch for {name}.")
    metadata = _read_json(bundle / "metadata.json")
    architecture = _read_json(bundle / "architecture.json")
    mapping = _read_json(bundle / "class_mapping.json")
    preprocessing = _read_json(bundle / "preprocessing.json")
    if metadata.get("bundle_schema_version") != "freshnet.deployment.v1":
        raise BundleError(f"Unsupported bundle schema: {metadata.get('bundle_schema_version')!r}")
    if metadata.get("architecture") != "fused" or architecture.get("architecture") != "fused":
        raise BundleError("Bundle architecture is not fused.")
    if metadata.get("smoke_test") is not False:
        raise BundleError("Refusing a smoke artifact. There is no fallback model.")
    if mapping.get("Fresh") != 0 or mapping.get("Spoiled") != 1 or mapping.get("num_classes") != 2:
        raise BundleError(f"Incompatible class mapping: {mapping}")
    if architecture.get("reconstruct_with_pretrained_download") is not False:
        raise BundleError("Bundle requested an ImageNet download. Refusing.")
    if preprocessing.get("eval_deterministic") is not True or preprocessing.get("augmentation_at_inference") is not False:
        raise BundleError("Bundle preprocessing is not the frozen deterministic eval path.")
    return {
        "bundle": bundle,
        "metadata": metadata,
        "architecture": architecture,
        "class_mapping": mapping,
        "preprocessing": preprocessing,
    }


def load_bundle(directory: Path, device: str = "cpu") -> LoadedBundle:
    checked = verify_bundle_files(directory)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_device = torch.device(device)
    architecture = checked["architecture"]
    model_cfg = {
        "embedding_dim": int(architecture["embedding_dim"]),
        "dropout": float(architecture["dropout"]),
        "context": {"backbone": "efficientnet_b0", "pretrained": False},
        "detail": {
            "channels": architecture["detail"]["channels"],
            "patch_layout": architecture["detail"].get("patch_layout", "corners_center"),
        },
    }
    model = build_model("fused", model_cfg, num_classes=2)
    weights = torch.load(checked["bundle"] / "weights.pt", map_location="cpu", weights_only=False)
    if "state_dict" not in weights:
        raise BundleError("weights.pt has no state_dict.")
    incompatible = model.load_state_dict(weights["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BundleError(f"Strict weight load failed: {incompatible}")
    model.to(torch_device)
    model.eval()
    mapping = checked["class_mapping"]
    return LoadedBundle(
        directory=checked["bundle"],
        model=model,
        device=torch_device,
        model_version=str(checked["metadata"]["model_version"]),
        class_mapping={"Fresh": int(mapping["Fresh"]), "Spoiled": int(mapping["Spoiled"])},
        preprocessing=checked["preprocessing"],
        metadata=checked["metadata"],
    )


def _preprocess_config(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "preprocessing": {
            "context_size": spec["context_size"],
            "detail_size": spec["detail_size"],
            "patch_size": spec["patch_size"],
            "imagenet_mean": spec["imagenet_mean"],
            "imagenet_std": spec["imagenet_std"],
            "eval_deterministic": True,
            "train_augmentation": {},
        }
    }


def classify_pil(bundle: LoadedBundle, image: Any) -> dict[str, Any]:
    """Classify one already-decoded RGB image with the frozen bundle preprocessing."""
    context, patches = eval_branch_tensors(image, _preprocess_config(bundle.preprocessing))
    with torch.inference_mode():
        logits = bundle.model(
            context=context.unsqueeze(0).to(bundle.device),
            patches=patches.unsqueeze(0).to(bundle.device),
        )
    logits_cpu = logits.detach().float().cpu()
    if logits_cpu.shape != (1, 2) or not torch.isfinite(logits_cpu).all():
        raise BundleError(f"Unexpected logits: {logits_cpu}")
    probabilities = F.softmax(logits_cpu, dim=1)[0]
    index = int(logits_cpu.argmax(dim=1).item())
    names = {index: name for name, index in bundle.class_mapping.items()}
    return {
        "model_version": bundle.model_version,
        "label": names[index],
        "label_index": index,
        "logits": [float(v) for v in logits_cpu[0].tolist()],
        "probabilities": {
            "Fresh": float(probabilities[bundle.class_mapping["Fresh"]]),
            "Spoiled": float(probabilities[bundle.class_mapping["Spoiled"]]),
        },
        "decision_rule": "argmax",
        "device": str(bundle.device),
    }


def classify_image(bundle: LoadedBundle, image_path: Path) -> dict[str, Any]:
    if not Path(image_path).is_file():
        raise BundleError(f"Image not found: {image_path}")
    return classify_pil(bundle, open_rgb_image(Path(image_path)))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify one image with a FreshNet deployment bundle.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    args = parser.parse_args(argv)
    try:
        bundle = load_bundle(Path(args.bundle), device=args.device)
        print(json.dumps(classify_image(bundle, Path(args.image)), indent=2))
    except BundleError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
