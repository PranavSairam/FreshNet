"""Export the selected v3 fused checkpoint as a weights-only deployment bundle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models import build_model
from training.config import downstream_hold_reason, resolve_path
from training.data import eval_branch_tensors, open_rgb_image

ORIGINAL_SPLIT_FINGERPRINT = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"
V3_FINGERPRINT = "4f2534505b3ca656e72b786046d0d7f7da1088a221fe857a3b16639bd87c8f57"
SELECTED_SHA256 = "b962c650ed807f362885e1f6c240d8abc693f824980b3afbfe8f597cafd60200"
SELECTED_CHECKPOINT = "artifacts/checkpoints/v3/fused/best.pt"
SCHEMA_VERSION = "freshnet.deployment.v1"
MODEL_VERSION = "freshnet-v3-fused-epoch29-b962c650ed80"
CPU_ATOL = 1e-5
BUNDLE_FILES = (
    "weights.pt",
    "architecture.json",
    "class_mapping.json",
    "preprocessing.json",
    "metadata.json",
    "runtime.json",
    "verification.json",
    "README.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hold_block_reason(fingerprint: str | None) -> str | None:
    """Block the provisional original split. Leave the hold file in place."""
    reason = downstream_hold_reason()
    if not reason:
        return None
    if not fingerprint or fingerprint == ORIGINAL_SPLIT_FINGERPRINT:
        return reason
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_selected_checkpoint(checkpoint: Path) -> dict[str, Any]:
    digest = sha256_file(checkpoint)
    if digest != SELECTED_SHA256:
        raise SystemExit(
            f"Checkpoint SHA-256 mismatch: computed {digest}, expected {SELECTED_SHA256}. Export stopped."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    problems: list[str] = []
    if payload.get("architecture") != "fused":
        problems.append(f"architecture={payload.get('architecture')!r}")
    if int(payload.get("epoch", -1)) != 29:
        problems.append(f"epoch={payload.get('epoch')!r}")
    if payload.get("smoke_test") is not False:
        problems.append(f"smoke_test={payload.get('smoke_test')!r}")
    if payload.get("split_fingerprint") != V3_FINGERPRINT:
        problems.append(f"fingerprint={payload.get('split_fingerprint')!r}")
    mapping = payload.get("class_mapping") or {}
    if mapping.get("Fresh") != 0 or mapping.get("Spoiled") != 1:
        problems.append(f"class_mapping={mapping!r}")
    if "state_dict" not in payload:
        problems.append("missing state_dict")
    selection_path = resolve_path("artifacts/metrics/v3/eval/selection.json")
    metrics_path = resolve_path("artifacts/metrics/v3/eval/fused/test/metrics.json")
    if not selection_path.is_file():
        problems.append("selection record missing")
    else:
        selection = _read_json(selection_path)
        if selection.get("checkpoint_sha256") != SELECTED_SHA256:
            problems.append("selection record hash does not match the checkpoint")
        if selection.get("checkpoint") != SELECTED_CHECKPOINT:
            problems.append(f"selection checkpoint path {selection.get('checkpoint')!r}")
        if selection.get("architecture") != "fused" or int(selection.get("epoch", -1)) != 29:
            problems.append("selection record architecture/epoch mismatch")
        if selection.get("dataset_fingerprint") != V3_FINGERPRINT:
            problems.append("selection fingerprint mismatch")
    if not metrics_path.is_file():
        problems.append("saved test metrics missing")
    else:
        metrics = _read_json(metrics_path)
        if metrics.get("checkpoint") != SELECTED_CHECKPOINT:
            problems.append(f"test metrics checkpoint {metrics.get('checkpoint')!r}")
        if metrics.get("split_fingerprint") != V3_FINGERPRINT or metrics.get("split") != "test":
            problems.append("test metrics fingerprint or split mismatch")
        if metrics.get("architecture") != "fused" or metrics.get("smoke_test") is not False:
            problems.append("test metrics architecture or smoke flag mismatch")
        if metrics.get("n") != 8035:
            problems.append(f"test metrics n={metrics.get('n')}")
    if problems:
        raise SystemExit("Export stopped. Identity check failed: " + "; ".join(problems))
    model_cfg = json.loads(json.dumps(payload["model_config"]))
    model_cfg.setdefault("context", {})["pretrained"] = False
    model = build_model("fused", model_cfg, num_classes=2)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SystemExit(f"Strict weight load failed: {incompatible}")
    return payload


def _runtime_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "platform": platform.platform(),
    }


def _preprocessing_spec(payload: dict[str, Any]) -> dict[str, Any]:
    stored = payload.get("preprocessing") or {}
    return {
        "accepted_input": "A still image that PIL.Image.open can decode. Deployment opens the file and calls convert('RGB').",
        "rgb_conversion": "PIL Image.convert('RGB') after Image.open. No EXIF transpose.",
        "exif_orientation": "Not applied. Photos whose pixels are stored rotated, with orientation only in EXIF, are an integration constraint. Parity was not claimed for that case.",
        "transparency": "Alpha is not a separate input. PIL convert('RGB') flattens it. Parity was not claimed for transparent images.",
        "animation": "Image.open uses the default first frame. Later frames are not read.",
        "resize_sequence": [
            "Resize the RGB image to detail_size x detail_size with torchvision.transforms.Resize. Training calls Resize((384, 384)) and does not pass interpolation or antialias, so the installed defaults apply: bilinear and antialias=True. On PIL images, bilinear resize is antialiased.",
            "From that 384 image, resize a context copy to 224 x 224 with PIL.Image.resize and Image.Resampling.BILINEAR.",
            "From the same 384 image, crop five 160 x 160 patches. There is no center-crop of the original and no second patch extraction.",
        ],
        "context_size": int(stored.get("context_size", 224)),
        "detail_size": int(stored.get("detail_size", 384)),
        "patch_size": int(stored.get("patch_size", 160)),
        "patch_order": ["top_left", "top_right", "bottom_left", "bottom_right", "center"],
        "patch_positions_on_384": [[0, 0], [224, 0], [0, 224], [224, 224], [112, 112]],
        "interpolation": "bilinear",
        "torchvision_resize_antialias_default": True,
        "tensor_scaling": "torchvision.transforms.ToTensor scales uint8 RGB to float32 in [0, 1] and layouts channels as CHW.",
        "normalization": {
            "mean": stored.get("imagenet_mean", [0.485, 0.456, 0.406]),
            "std": stored.get("imagenet_std", [0.229, 0.224, 0.225]),
            "applied": "once, after ToTensor, separately to the context image and to each patch",
        },
        "imagenet_mean": stored.get("imagenet_mean", [0.485, 0.456, 0.406]),
        "imagenet_std": stored.get("imagenet_std", [0.229, 0.224, 0.225]),
        "branch_transforms_inside_model": False,
        "augmentation_at_inference": False,
        "eval_deterministic": True,
        "train_augmentation_recorded_but_not_used": stored.get("train_augmentation"),
        "input_dtype": "float32",
        "channel_order": "RGB",
        "context_shape": [3, 224, 224],
        "patches_shape": [5, 3, 160, 160],
    }


def _architecture_spec(payload: dict[str, Any]) -> dict[str, Any]:
    model_cfg = payload["model_config"]
    return {
        "architecture": "fused",
        "reconstruct_with_pretrained_download": False,
        "num_classes": 2,
        "embedding_dim": int(model_cfg.get("embedding_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.3)),
        "context": {
            "backbone": "efficientnet_b0",
            "classifier_replaced_with": "Identity",
            "projection": "Linear(1280, 256) then ReLU",
            "load_imagenet_weights": False,
        },
        "detail": {
            "patch_layout": model_cfg.get("detail", {}).get("patch_layout", "corners_center"),
            "channels": model_cfg.get("detail", {}).get("channels", [32, 64, 128, 256]),
            "shared_cnn": "Conv-BN-ReLU stride 2 per channel stage, adaptive average pool, Linear to 256",
            "aggregation": "mean over the five patch embeddings, then Linear(256, 256) and ReLU",
        },
        "fusion": "concatenate context and detail embeddings on dim 1 (512), then LayerNorm, Dropout(0.3), Linear(512, 2)",
        "classifier_in_features": 512,
        "classifier_out_features": 2,
        "decision_rule": "argmax over the two logits",
        "probabilities": "softmax over dim 1 (the class dimension). These are not a calibrated freshness percentage.",
        "label_order": ["Fresh", "Spoiled"],
    }


def _limitations() -> list[str]:
    return [
        "The v3 benchmark includes 53,052 of 71,318 original images. 18,266 images were quarantined.",
        "Confirmed retained same-capture links have zero cross-split crossings.",
        "Duplicate candidate search was capped at 64 comparisons per image in an average-hash band and is not exhaustive. Specimen and capture-session metadata are incomplete.",
        "Reported scores apply to the included v3 benchmark. The dataset is not conclusively leakage-free.",
        "Image predictions are visual estimates. They do not certify food safety, remaining shelf life, or a calibrated freshness percentage.",
        "There is no Medium class and no tuned decision threshold.",
    ]


def _select_val_images(limit_groups: int = 4) -> list[dict[str, str]]:
    """First fresh and spoiled validation rows from distinct groups, in file order."""
    import csv

    path = resolve_path("data/processed/manifests_v3_source/val.csv")
    chosen: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    need = {"Fresh": 2, "Spoiled": 2}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"]
            group = row.get("group_id") or ""
            if need.get(label, 0) <= 0 or group in seen_groups:
                continue
            chosen.append(
                {
                    "image_path": row["image_path"],
                    "label": label,
                    "group_id": group,
                    "split": "val",
                }
            )
            seen_groups.add(group)
            need[label] -= 1
            if sum(need.values()) == 0 or len(chosen) >= limit_groups:
                break
    if {row["label"] for row in chosen} != {"Fresh", "Spoiled"} or len(seen_groups) < 2:
        raise SystemExit("Could not build a two-label, multi-group validation parity set.")
    return chosen


def _config_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
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


def _forward(model: torch.nn.Module, context: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(context=context, patches=patches)


def _compare(reference: torch.Tensor, exported: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    exp = exported.detach().float().cpu()
    if ref.shape != exp.shape:
        raise SystemExit(f"Logit shape mismatch: {tuple(ref.shape)} vs {tuple(exp.shape)}")
    if not torch.isfinite(exp).all():
        raise SystemExit("Exported logits are not finite.")
    probs_ref = F.softmax(ref, dim=1)
    probs_exp = F.softmax(exp, dim=1)
    prob_sums = probs_exp.sum(dim=1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        raise SystemExit(f"Softmax rows do not sum to one: {prob_sums.tolist()}")
    pred_ref = ref.argmax(dim=1)
    pred_exp = exp.argmax(dim=1)
    mismatches = int((pred_ref != pred_exp).sum().item())
    logit_diff = float((ref - exp).abs().max().item())
    prob_diff = float((probs_ref - probs_exp).abs().max().item())
    if logit_diff > CPU_ATOL or mismatches:
        raise SystemExit(
            f"CPU parity failed. max_abs_logit={logit_diff} mismatches={mismatches} atol={CPU_ATOL}."
        )
    return {
        "max_abs_logit_difference": logit_diff,
        "max_abs_probability_difference": prob_diff,
        "class_mismatches": mismatches,
        "output_shape": list(exp.shape),
        "tolerance_atol": CPU_ATOL,
        "dtype": "float32",
    }


def _load_exported(bundle_dir: Path, device: torch.device) -> torch.nn.Module:
    arch = _read_json(bundle_dir / "architecture.json")
    weights = torch.load(bundle_dir / "weights.pt", map_location="cpu", weights_only=False)
    model_cfg = {
        "embedding_dim": arch["embedding_dim"],
        "dropout": arch["dropout"],
        "context": {"backbone": "efficientnet_b0", "pretrained": False},
        "detail": {"channels": arch["detail"]["channels"], "patch_layout": "corners_center"},
    }
    model = build_model("fused", model_cfg, num_classes=2)
    model.load_state_dict(weights["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _verify_parity(payload: dict[str, Any], bundle_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    model_cfg = json.loads(json.dumps(payload["model_config"]))
    model_cfg.setdefault("context", {})["pretrained"] = False
    source = build_model("fused", model_cfg, num_classes=2)
    source.load_state_dict(payload["state_dict"], strict=True)
    source.to("cpu")
    source.eval()
    exported = _load_exported(bundle_dir, torch.device("cpu"))
    config = _config_from_spec(spec)
    samples = _select_val_images()
    contexts = []
    patches = []
    reference_contexts = []
    reference_patches = []
    for sample in samples:
        image = open_rgb_image(Path(sample["image_path"]))
        context_t, patches_t = eval_branch_tensors(image, config)
        contexts.append(context_t)
        patches.append(patches_t)
        reference_contexts.append(context_t.clone())
        reference_patches.append(patches_t.clone())
    context_batch = torch.stack(contexts, dim=0)
    patches_batch = torch.stack(patches, dim=0)
    tensor_pair = _compare(
        _forward(source, context_batch, patches_batch),
        _forward(exported, context_batch, patches_batch),
    )
    single = _compare(
        _forward(source, context_batch[:1], patches_batch[:1]),
        _forward(exported, context_batch[:1], patches_batch[:1]),
    )
    reloaded_context, reloaded_patches = [], []
    for sample in samples:
        image = open_rgb_image(Path(sample["image_path"]))
        context_t, patches_t = eval_branch_tensors(image, config)
        reloaded_context.append(context_t)
        reloaded_patches.append(patches_t)
    image_path = _compare(
        _forward(source, torch.stack(reference_contexts), torch.stack(reference_patches)),
        _forward(exported, torch.stack(reloaded_context), torch.stack(reloaded_patches)),
    )
    torch.manual_seed(0)
    synthetic_context = torch.randn(2, 3, 224, 224)
    synthetic_patches = torch.randn(2, 5, 3, 160, 160)
    synthetic = _compare(
        _forward(source, synthetic_context, synthetic_patches),
        _forward(exported, synthetic_context, synthetic_patches),
    )
    mps_report: dict[str, Any] = {"checked": False}
    if torch.backends.mps.is_available():
        mps_model = _load_exported(bundle_dir, torch.device("mps"))
        cpu_logits = _forward(exported, context_batch[:1], patches_batch[:1])
        mps_logits = _forward(
            mps_model,
            context_batch[:1].to("mps"),
            patches_batch[:1].to("mps"),
        )
        diff = float((cpu_logits - mps_logits.float().cpu()).abs().max().item())
        mps_report = {
            "checked": True,
            "images": 1,
            "max_abs_logit_difference_vs_cpu": diff,
            "class_mismatch_vs_cpu": int(
                (cpu_logits.argmax(dim=1) != mps_logits.argmax(dim=1).cpu()).sum().item()
            ),
            "note": "CPU and MPS are not claimed to be bitwise identical. This difference is not the export tolerance.",
        }
    return {
        "passed": True,
        "sample_split": "val",
        "sample_count": len(samples),
        "sample_selection": (
            "Validation manifest order. The first Fresh and Spoiled rows whose group_id "
            "values are all distinct, two of each label. The test split was not read."
        ),
        "samples": [
            {
                "label": row["label"],
                "group_sha256_12": hashlib.sha256(row["group_id"].encode()).hexdigest()[:12],
                "split": "val",
            }
            for row in samples
        ],
        "devices_for_export_parity": ["cpu"],
        "cpu_atol": CPU_ATOL,
        "identical_preprocessed_tensors": tensor_pair,
        "single_image": single,
        "bundle_image_path_vs_reference_tensors": image_path,
        "synthetic_batch": synthetic,
        "mps_smoke": mps_report,
        "class_mismatch_count": (
            tensor_pair["class_mismatches"]
            + single["class_mismatches"]
            + image_path["class_mismatches"]
            + synthetic["class_mismatches"]
        ),
        "fresh_process": _fresh_process_check(bundle_dir, Path(samples[0]["image_path"]), spec),
    }


def _fresh_process_check(bundle_dir: Path, image_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Load the staged bundle in a new interpreter and score one copied image."""
    with tempfile.TemporaryDirectory() as tmp:
        copied = Path(tmp) / "sample.png"
        shutil.copyfile(image_path, copied)
        script = r"""
import json, sys
from pathlib import Path
import torch
from deployment.export_bundle import _config_from_spec, _load_exported, _read_json
from training.data import eval_branch_tensors, open_rgb_image

bundle = Path(sys.argv[1])
image = Path(sys.argv[2])
spec = _read_json(bundle / "preprocessing.json")
model = _load_exported(bundle, torch.device("cpu"))
context, patches = eval_branch_tensors(open_rgb_image(image), _config_from_spec(spec))
logits = model(context=context.unsqueeze(0), patches=patches.unsqueeze(0))
print(json.dumps({"shape": list(logits.shape), "logits": [round(float(v), 8) for v in logits[0].tolist()]}))
"""
        env = os.environ.copy()
        ml_path = str(resolve_path("ml"))
        env["PYTHONPATH"] = ml_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["TORCH_HOME"] = tmp
        completed = subprocess.run(
            [sys.executable, "-c", script, str(bundle_dir), str(copied)],
            cwd=str(resolve_path(".")),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                "Fresh-process bundle load failed.\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        reported = json.loads(completed.stdout.strip().splitlines()[-1])
    reference = _load_exported(bundle_dir, torch.device("cpu"))
    context, patches = eval_branch_tensors(open_rgb_image(image_path), _config_from_spec(spec))
    logits = _forward(reference, context.unsqueeze(0), patches.unsqueeze(0))
    fresh = torch.tensor(reported["logits"], dtype=torch.float32)
    diff = float((logits[0].cpu() - fresh).abs().max().item())
    if reported["shape"] != [1, 2] or diff > 1e-5:
        raise SystemExit(f"Fresh-process logits diverged: diff={diff} shape={reported['shape']}")
    return {
        "passed": True,
        "checkpoint_path_not_passed": True,
        "max_abs_logit_difference_vs_parent": diff,
        "output_shape": reported["shape"],
    }


def _checksums(bundle_dir: Path) -> dict[str, str]:
    digests = {}
    for name in BUNDLE_FILES:
        path = bundle_dir / name
        if not path.is_file():
            raise SystemExit(f"Bundle file missing before checksum: {name}")
        digests[name] = sha256_file(path)
    return digests


def _bundle_is_reusable(bundle_dir: Path) -> bool:
    checksum_path = bundle_dir / "checksums.json"
    if not checksum_path.is_file():
        return False
    manifest = _read_json(checksum_path)
    if manifest.get("model_version") != MODEL_VERSION:
        return False
    try:
        for name, digest in manifest["files"].items():
            if sha256_file(bundle_dir / name) != digest:
                return False
    except OSError:
        return False
    metadata = _read_json(bundle_dir / "metadata.json")
    verification = _read_json(bundle_dir / "verification.json")
    return (
        metadata.get("source_checkpoint_sha256") == SELECTED_SHA256
        and metadata.get("smoke_test") is False
        and verification.get("passed") is True
    )


def export_selected_bundle(checkpoint: Path, bundle_dir: Path) -> dict[str, Any]:
    block = hold_block_reason(V3_FINGERPRINT)
    if block:
        raise SystemExit(block)
    if bundle_dir.exists() and _bundle_is_reusable(bundle_dir):
        return {
            "reused": True,
            "bundle_dir": "artifacts/deployment/v3/fused_epoch29_b962c650ed80",
            "model_version": MODEL_VERSION,
        }
    if bundle_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing bundle: {bundle_dir}")
    payload = verify_selected_checkpoint(checkpoint)
    selection = _read_json(resolve_path("artifacts/metrics/v3/eval/selection.json"))
    test_metrics = _read_json(resolve_path("artifacts/metrics/v3/eval/fused/test/metrics.json"))
    staging = bundle_dir.parent / f".{bundle_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        spec = _preprocessing_spec(payload)
        architecture = _architecture_spec(payload)
        _write_json(staging / "architecture.json", architecture)
        _write_json(
            staging / "class_mapping.json",
            {"Fresh": 0, "Spoiled": 1, "num_classes": 2, "label_order": ["Fresh", "Spoiled"]},
        )
        _write_json(staging / "preprocessing.json", spec)
        weights_path = staging / "weights.pt"
        torch.save({"state_dict": payload["state_dict"]}, weights_path)
        weights_sha = sha256_file(weights_path)
        _write_json(
            staging / "metadata.json",
            {
                "bundle_schema_version": SCHEMA_VERSION,
                "model_version": MODEL_VERSION,
                "source_checkpoint": SELECTED_CHECKPOINT,
                "source_checkpoint_sha256": SELECTED_SHA256,
                "exported_weights_sha256": weights_sha,
                "architecture": "fused",
                "epoch": 29,
                "smoke_test": False,
                "dataset_fingerprint": V3_FINGERPRINT,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "initialization": payload.get("initialization"),
                "decision_rule": "argmax over the two output logits",
                "probabilities": "softmax over the class dimension for display only",
                "validation": {
                    "label": "validation results reused from training and the selection record; not recomputed",
                    "candidates": selection["validation_results"]["candidates"],
                    "selected_checkpoint": selection["validation_results"]["selected_checkpoint_val_metrics"],
                },
                "test": {
                    "label": "completed test evaluation reused; the test set was not rescored",
                    "metrics": test_metrics,
                },
                "benchmark_limitations": _limitations(),
                "runtime": _runtime_info(),
            },
        )
        _write_json(staging / "runtime.json", _runtime_info())
        (staging / "README.md").write_text(
            "\n".join(
                [
                    "# FreshNet v3 fused deployment bundle",
                    "",
                    f"Model version: `{MODEL_VERSION}`",
                    "",
                    "Load `weights.pt` with `pretrained=False` and `load_state_dict(..., strict=True)`.",
                    "Do not download ImageNet weights. The trained tensors are in this bundle.",
                    "Preprocess with `training.data.eval_branch_tensors` after `open_rgb_image`.",
                    "Predict with argmax on the two logits. Softmax is for display only.",
                    "",
                    "```bash",
                    "PYTHONPATH=backend:ml backend/.venv/bin/python -m app.services.inference \\",
                    "  --bundle artifacts/deployment/v3/fused_epoch29_b962c650ed80 \\",
                    "  --image path/to/image.jpg",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        verification = _verify_parity(payload, staging, spec)
        _write_json(staging / "verification.json", verification)
        _write_json(
            staging / "checksums.json",
            {
                "algorithm": "sha256",
                "model_version": MODEL_VERSION,
                "files": _checksums(staging),
            },
        )
        staging.rename(bundle_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    metadata = _read_json(bundle_dir / "metadata.json")
    return {
        "reused": False,
        "bundle_dir": "artifacts/deployment/v3/fused_epoch29_b962c650ed80",
        "model_version": MODEL_VERSION,
        "source_checkpoint_sha256": SELECTED_SHA256,
        "exported_weights_sha256": metadata["exported_weights_sha256"],
        "verification": _read_json(bundle_dir / "verification.json"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export the selected FreshNet v3 fused bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--bundle-dir",
        default="artifacts/deployment/v3/fused_epoch29_b962c650ed80",
    )
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    payload_fp = None
    try:
        header = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload_fp = header.get("split_fingerprint")
    except Exception as exc:  # noqa: BLE001
        print(f"Unreadable checkpoint: {exc}", file=sys.stderr)
        return 2
    blocked = hold_block_reason(str(payload_fp) if payload_fp else None)
    if blocked:
        print(
            json.dumps(
                {
                    "event": "downstream_hold",
                    "stage": "export",
                    "reason": blocked,
                    "fingerprint": payload_fp,
                    "note": "The hold file remains. It blocks the provisional original split.",
                }
            )
        )
        return 3
    if payload_fp != V3_FINGERPRINT:
        print(f"Refusing export for fingerprint {payload_fp}", file=sys.stderr)
        return 3
    result = export_selected_bundle(checkpoint, resolve_path(args.bundle_dir))
    print(json.dumps(result, indent=2))
    return 0
