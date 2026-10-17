#!/usr/bin/env python3
"""Export the selected produce checkpoint as a separate, versioned deployment bundle.

    backend/.venv/bin/python ml/produce/export_bundle.py

Requires the frozen selection and the one-time test evaluation. The freshness bundle is
never read or modified. Bundle layout (schema freshnet.produce.deployment.v1):

    weights.pt  architecture.json  class_mapping.json  preprocessing.json
    metadata.json  checksums.json  README.md

After writing, a parity check loads the exported bundle through the backend loader and
compares its logits with the source checkpoint on a small validation sample.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

from produce.build_manifests import CANONICAL  # noqa: E402
from produce.data import ProduceDataset, read_manifest  # noqa: E402
from produce.evaluate import display_path, load_selected_model, sha256_file  # noqa: E402
from produce.train import load_config, pick_device, resolve  # noqa: E402

SCHEMA = "freshnet.produce.deployment.v1"
PARITY_SAMPLE = 48
PARITY_TOLERANCE = 1e-4


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_checksums(bundle: Path) -> None:
    files = {p.name: sha256_file(p) for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "checksums.json"}
    write_json(bundle / "checksums.json", {"algorithm": "sha256", "files": files})


def parity_check(bundle_dir: Path, source_model, cfg, class_to_index, device: torch.device) -> dict[str, Any]:
    from app.services.produce_model import load_produce_bundle, predict_probabilities

    exported = load_produce_bundle(bundle_dir, device="cpu")
    dataset = ProduceDataset(resolve(cfg["data"]["val_csv"]), cfg, class_to_index, train=False)
    rows = dataset.rows
    step = max(len(rows) // PARITY_SAMPLE, 1)
    picked = list(range(0, len(rows), step))[:PARITY_SAMPLE]
    source_model_cpu = source_model.to("cpu").eval()
    max_diff, agree = 0.0, 0
    for index in picked:
        item = dataset[index]
        with torch.inference_mode():
            reference = source_model_cpu(item["image"].unsqueeze(0)).float()[0].numpy()
        from produce.data import open_rgb

        image = open_rgb(Path(rows[index]["image_path"]))
        result = predict_probabilities(exported, image)
        diff = float(np.max(np.abs(np.array(result["logits"]) - reference)))
        max_diff = max(max_diff, diff)
        agree += int(result["index"] == int(reference.argmax()))
    return {
        "images": len(picked),
        "max_abs_logit_difference": max_diff,
        "argmax_agreement": f"{agree}/{len(picked)}",
        "tolerance": PARITY_TOLERANCE,
        "passed": max_diff <= PARITY_TOLERANCE and agree == len(picked),
        "source_device": "cpu",
        "exported_device": "cpu",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the selected produce checkpoint.")
    parser.add_argument("--config", default="configs/produce_v1.yaml")
    parser.add_argument("--force", action="store_true", help="Replace an existing bundle directory of the same name.")
    args = parser.parse_args(argv)
    cfg = load_config(resolve(args.config))
    eval_dir = resolve(cfg["evaluation"]["output_dir"])
    selection_path, results_path = eval_dir / "selection.json", eval_dir / "test_results.json"
    if not selection_path.is_file() or not results_path.is_file():
        print("Run `evaluate.py select` and `evaluate.py test` first.", file=sys.stderr)
        return 2
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoint = resolve(selection["checkpoint"])
    if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after selection. Refusing to export.")
    if results["checkpoint_sha256"] != selection["checkpoint_sha256"]:
        raise RuntimeError("Test results belong to a different checkpoint.")

    class_order = selection["class_order"]
    class_to_index = selection["class_to_index"]
    short = selection["checkpoint_sha256"][:12]
    model_version = f"freshnet-produce-v1-effb0-{short}"
    out_root = resolve(cfg["evaluation"]["export_dir"])
    bundle = out_root / f"effb0_epoch{selection['best_epoch']}_{short}"
    if bundle.exists():
        if not args.force:
            print(f"{bundle} exists. Use --force to replace it.", file=sys.stderr)
            return 2
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    device = pick_device("cpu")
    model, payload = load_selected_model(checkpoint, class_order, cfg, device)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, bundle / "weights.pt")

    write_json(
        bundle / "architecture.json",
        {
            "architecture": "efficientnet_b0",
            "family": "torchvision.models.efficientnet_b0 with a new classifier head",
            "classifier": f"Sequential(Dropout({cfg['model']['dropout']}), Linear(1280, {len(class_order)}))",
            "num_classes": len(class_order),
            "dropout": cfg["model"]["dropout"],
            "initialization_for_training": "ImageNet (torchvision EfficientNet_B0_Weights.DEFAULT)",
            "reconstruct_with_pretrained_download": False,
            "decision_rule": "argmax over softmax of the 12 logits; softmax is display-only",
        },
    )
    write_json(
        bundle / "class_mapping.json",
        {
            "class_order": class_order,
            "class_to_index": class_to_index,
            "num_classes": len(class_order),
            "labels": {cid: CANONICAL[cid]["label"] for cid in class_order},
            "source_folder_aliases": {cid: CANONICAL[cid]["aliases"] for cid in class_order},
            "supported_only": "These 12 produce types only. There is no 'other' or 'not produce' class.",
        },
    )
    pre = cfg["preprocessing"]
    write_json(
        bundle / "preprocessing.json",
        {
            "input_size": pre["input_size"],
            "resize": "Resize((224, 224)) bilinear with antialiasing, no crop, aspect ratio not preserved",
            "imagenet_mean": pre["imagenet_mean"],
            "imagenet_std": pre["imagenet_std"],
            "color_mode": "RGB",
            "eval_deterministic": True,
            "augmentation_at_inference": False,
            "separate_from_freshness_preprocessing": True,
        },
    )
    chosen = selection["confirmation"]["chosen"]
    write_json(
        bundle / "metadata.json",
        {
            "bundle_schema_version": SCHEMA,
            "model_version": model_version,
            "task": "produce-type classification",
            "architecture": "efficientnet_b0",
            "smoke_test": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": selection["checkpoint"],
            "source_checkpoint_sha256": selection["checkpoint_sha256"],
            "best_epoch": selection["best_epoch"],
            "selection_rule": selection["selection_rule"],
            "manifest_fingerprint": selection["manifest_fingerprint"],
            "source_split_fingerprint": selection["source_split_fingerprint"],
            "training": selection["training"],
            "seed": cfg["training"]["seed"],
            "confirmation": {
                "enabled": True,
                "method": selection["confirmation"]["method"],
                "threshold": chosen["threshold"],
                "chosen_on": "validation only",
                "rule": cfg["confirmation"],
                "validation_coverage": chosen["coverage"],
                "validation_selective_accuracy": chosen["selective_accuracy"],
                "note": "Asks the user to confirm low-confidence guesses. Not an out-of-distribution detector.",
            },
            "validation_metrics": {k: selection["validation_metrics"][k] for k in ("n", "accuracy", "macro_f1", "balanced_accuracy")},
            "test_metrics": {
                "scope": results["split"],
                **{k: results["test_metrics"][k] for k in ("n", "accuracy", "macro_f1", "balanced_accuracy")},
            },
            "evaluation_files": {
                "selection": display_path(selection_path),
                "test_results": display_path(results_path),
            },
            "freshness_model_untouched": "artifacts/deployment/v3/fused_epoch29_b962c650ed80",
            "limitations": [
                "Recognises only the 12 trained produce types and has no non-produce class.",
                "Softmax confidence is not a calibrated probability that the item is supported.",
                "Test images are the existing holdout images, also used for the freshness evaluation (separate-task evaluation).",
            ],
        },
    )
    write_checksums(bundle)  # core files only, so the backend loader can verify the bundle for the parity check
    parity = parity_check(bundle, model, cfg, class_to_index, device)
    write_json(bundle / "parity_check.json", parity)
    if not parity["passed"]:
        print(json.dumps(parity, indent=2), file=sys.stderr)
        shutil.rmtree(bundle)
        print("Parity check failed; bundle removed.", file=sys.stderr)
        return 3
    (bundle / "README.md").write_text(
        f"# FreshNet produce bundle `{model_version}`\n\n"
        "Separate 12-class produce-type classifier (EfficientNet-B0). It is independent of the freshness bundle.\n\n"
        f"- Validation accuracy / macro F1: {selection['validation_metrics']['accuracy']} / {selection['validation_metrics']['macro_f1']}\n"
        f"- Test accuracy / macro F1 (existing holdout, separate-task evaluation): {results['test_metrics']['accuracy']} / {results['test_metrics']['macro_f1']}\n"
        f"- Confirmation threshold (validation-chosen): {chosen['threshold']}\n\n"
        "Serve with `FRESHNET_PRODUCE_BUNDLE_DIR=<this directory>`. If it cannot be loaded, the API falls back to manual produce selection.\n",
        encoding="utf-8",
    )
    write_checksums(bundle)
    print(json.dumps({"bundle": str(bundle), "model_version": model_version, "parity": parity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
