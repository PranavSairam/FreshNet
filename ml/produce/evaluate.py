#!/usr/bin/env python3
"""Select the produce checkpoint on validation, freeze the choice, then evaluate once on test.

    backend/.venv/bin/python ml/produce/evaluate.py select   # needs a completed training run
    backend/.venv/bin/python ml/produce/evaluate.py test     # needs selection.json; runs once

`select` never opens the test CSV. It records the checkpoint SHA-256, class mapping,
preprocessing and manifest fingerprint in selection.json, and chooses the optional
"please confirm produce type" threshold from validation data only.

`test` verifies the checkpoint still matches selection.json, evaluates the produce test
subset once, and refuses to run again. This is a separate-task evaluation on the existing
holdout images (they were also the freshness model's test images). It is not a new
independent external test set, and it must never be quoted as the freshness accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from produce.data import ProduceDataset, collate  # noqa: E402
from produce.metrics import classification_metrics  # noqa: E402
from produce.model import build_produce_model  # noqa: E402
from produce.train import load_config, manifest_fingerprint, pick_device, resolve, verify_data  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_model(checkpoint: Path, class_order: list[str], cfg: dict[str, Any], device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("smoke_test"):
        raise RuntimeError("Refusing a smoke checkpoint.")
    model = build_produce_model(len(class_order), cfg["model"], pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def collect_logits(model, dataset: ProduceDataset, device: torch.device, batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    logits, labels = [], []
    for batch in loader:
        logits.append(model(batch["image"].to(device)).float().cpu())
        labels.append(batch["label"])
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def coverage_table(probs: np.ndarray, labels: np.ndarray, grid: list[float]) -> list[dict[str, Any]]:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    rows = []
    for threshold in grid:
        accepted = confidence >= threshold
        n_acc = int(accepted.sum())
        rows.append(
            {
                "threshold": threshold,
                "coverage": round(n_acc / len(labels), 6),
                "accepted": n_acc,
                "selective_accuracy": round(float(correct[accepted].mean()), 6) if n_acc else None,
                "needs_confirmation": int(len(labels) - n_acc),
                "errors_accepted": int((~correct[accepted]).sum()),
                "errors_sent_to_confirmation": int((~correct[~accepted]).sum()),
            }
        )
    return rows


def choose_threshold(table: list[dict[str, Any]], target: float) -> dict[str, Any]:
    for row in table:  # grid is ascending
        if row["selective_accuracy"] is not None and row["selective_accuracy"] >= target:
            return {**row, "rule_outcome": f"smallest grid value with selective accuracy >= {target}"}
    return {**table[-1], "rule_outcome": f"no grid value reached {target}; used the largest grid value"}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _eval_dir(cfg: dict[str, Any]) -> Path:
    return resolve(cfg["evaluation"]["output_dir"])


def select(cfg: dict[str, Any], device_spec: str | None) -> int:
    tr = cfg["training"]
    ckpt_dir = resolve(tr["checkpoint_dir"])
    complete = ckpt_dir / "complete.json"
    if not complete.is_file():
        print("Training is not complete (no complete.json). Refusing to select a checkpoint.", file=sys.stderr)
        return 2
    out = _eval_dir(cfg)
    selection_path = out / "selection.json"
    if selection_path.is_file():
        print(f"{selection_path} already exists. The selection is frozen; not overwriting.", file=sys.stderr)
        return 2
    class_to_index, class_order, fingerprint = verify_data(cfg)
    best = ckpt_dir / "best.pt"
    device = pick_device(device_spec or str(tr.get("device", "auto")))
    model, payload = load_selected_model(best, class_order, cfg, device)
    if payload["manifest_fingerprint"] != fingerprint or payload["class_order"] != class_order:
        raise RuntimeError("Checkpoint was trained on different data or class order.")

    val_ds = ProduceDataset(resolve(cfg["data"]["val_csv"]), cfg, class_to_index, train=False)
    logits, labels = collect_logits(model, val_ds, device, 64, int(tr["num_workers"]))
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    val_metrics = classification_metrics(labels.tolist(), preds.tolist(), class_order)

    rule = cfg["confirmation"]
    grid = sorted(float(v) for v in rule["grid"])
    table = coverage_table(probs, labels, grid)
    chosen = choose_threshold(table, float(rule["target_selective_accuracy"]))

    complete_info = json.loads(complete.read_text(encoding="utf-8"))
    selection = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "highest validation macro F1 across epochs (computed during training), best.pt",
        "checkpoint": display_path(best),
        "checkpoint_sha256": sha256_file(best),
        "best_epoch": int(payload["best_epoch"]),
        "training": {k: complete_info.get(k) for k in ("stop_reason", "epochs_completed", "best_epoch", "best_val_macro_f1")},
        "manifest_fingerprint": fingerprint,
        "source_split_fingerprint": cfg["experiment"]["expected_source_split_fingerprint"],
        "class_order": class_order,
        "class_to_index": class_to_index,
        "preprocessing": cfg["preprocessing"] | {"train_augmentation": None},
        "validation_metrics": val_metrics,
        "validation_recomputed_matches_training_log": abs(val_metrics["macro_f1"] - float(payload["best_metric"])) < 1e-4,
        "confirmation": {
            "method": rule["method"],
            "target_selective_accuracy": float(rule["target_selective_accuracy"]),
            "chosen_on": "validation only",
            "chosen": chosen,
            "validation_trade_off": table,
            "note": "A confirmation/abstention rule. It is not an out-of-distribution detector: the model has no non-produce class.",
        },
        "test_split_opened": False,
        "device": str(device),
        "torch": torch.__version__,
    }
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "validation_logits.npy", logits)
    np.save(out / "validation_labels.npy", labels)
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({k: selection[k] for k in ("checkpoint_sha256", "best_epoch", "manifest_fingerprint")}, indent=2))
    print("validation macro_f1", val_metrics["macro_f1"], "accuracy", val_metrics["accuracy"])
    print("confirmation threshold", chosen["threshold"], chosen["rule_outcome"])
    return 0


def subgroup_breakdown(rows: list[dict[str, str]], labels: np.ndarray, preds: np.ndarray, class_order: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    freshness = np.array([r["freshness_label"] for r in rows])
    for group in ("Fresh", "Spoiled"):
        mask = freshness == group
        correct = preds[mask] == labels[mask]
        per_class = {}
        for index, name in enumerate(class_order):
            m = mask & (labels == index)
            if m.sum():
                per_class[name] = {"n": int(m.sum()), "accuracy": round(float((preds[m] == labels[m]).mean()), 6)}
        result[group] = {
            "n": int(mask.sum()),
            "accuracy": round(float(correct.mean()), 6) if mask.sum() else None,
            "per_class_accuracy": per_class,
        }
    return result


def test(cfg: dict[str, Any], device_spec: str | None) -> int:
    tr = cfg["training"]
    out = _eval_dir(cfg)
    selection_path = out / "selection.json"
    results_path = out / "test_results.json"
    if not selection_path.is_file():
        print("No selection.json. Run `select` first.", file=sys.stderr)
        return 2
    if results_path.is_file():
        print(f"{results_path} exists. The test set is evaluated once; not re-running.", file=sys.stderr)
        return 2
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    class_to_index, class_order, fingerprint = verify_data(cfg)
    best = resolve(selection["checkpoint"])
    if sha256_file(best) != selection["checkpoint_sha256"]:
        raise RuntimeError("best.pt changed after selection. Refusing to evaluate.")
    if selection["manifest_fingerprint"] != fingerprint or selection["class_order"] != class_order:
        raise RuntimeError("Data or class order differs from the frozen selection.")
    device = pick_device(device_spec or str(tr.get("device", "auto")))
    model, _ = load_selected_model(best, class_order, cfg, device)

    test_ds = ProduceDataset(resolve(cfg["data"]["test_csv"]), cfg, class_to_index, train=False)
    logits, labels = collect_logits(model, test_ds, device, 64, int(tr["num_workers"]))
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    metrics = classification_metrics(labels.tolist(), preds.tolist(), class_order)
    threshold = float(selection["confirmation"]["chosen"]["threshold"])
    test_table = coverage_table(probs, labels, [threshold])[0]
    results = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "task": "produce-type classification (12 classes), NOT freshness",
        "split": "produce test subset of the existing v3 holdout; a separate-task evaluation, not a new independent external test",
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "manifest_fingerprint": fingerprint,
        "test_metrics": metrics,
        "fresh_spoiled_subgroups": subgroup_breakdown(test_ds.rows, labels, preds, class_order),
        "confirmation_at_validation_threshold": {
            **test_table,
            "note": "Reported at the threshold chosen on validation. The threshold was not tuned on test.",
        },
        "test_images": int(len(labels)),
    }
    out.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.save(out / "test_logits.npy", logits)
    np.save(out / "test_labels.npy", labels)
    write_report(selection, results, out / "EVALUATION.md")
    print("test accuracy", metrics["accuracy"], "macro_f1", metrics["macro_f1"], "balanced_accuracy", metrics["balanced_accuracy"])
    return 0


def write_report(selection: dict[str, Any], results: dict[str, Any], path: Path) -> None:
    val = selection["validation_metrics"]
    test_m = results["test_metrics"]
    chosen = selection["confirmation"]["chosen"]
    lines = [
        "# Produce classifier evaluation",
        "",
        "Task: 12-class produce type. This is **not** the freshness model and its numbers must not be mixed with the freshness results.",
        "",
        f"- Checkpoint SHA-256: `{selection['checkpoint_sha256']}` (best epoch {selection['best_epoch']}, chosen by validation macro F1)",
        f"- Produce manifest fingerprint: `{selection['manifest_fingerprint']}`",
        f"- Test subset: {results['test_images']} images. {results['split']}.",
        "",
        "| Split | Accuracy | Macro F1 | Balanced accuracy |",
        "|---|---|---|---|",
        f"| Validation (used for selection) | {val['accuracy']:.4f} | {val['macro_f1']:.4f} | {val['balanced_accuracy']:.4f} |",
        f"| Test (evaluated once) | {test_m['accuracy']:.4f} | {test_m['macro_f1']:.4f} | {test_m['balanced_accuracy']:.4f} |",
        "",
        "## Per-class test results",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name, row in test_m["per_class"].items():
        lines.append(f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |")
    lines += ["", "## Confusion matrix (rows = true, columns = predicted)", "", "| true \\ pred | " + " | ".join(test_m["class_order"]) + " |", "|---|" + "---|" * len(test_m["class_order"])]
    for name, row in zip(test_m["class_order"], test_m["confusion_matrix"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Fresh / Spoiled subgroup accuracy (produce-type correctness)", ""]
    for group, info in results["fresh_spoiled_subgroups"].items():
        lines.append(f"- {group}: n={info['n']}, produce accuracy={info['accuracy']}")
    conf = results["confirmation_at_validation_threshold"]
    lines += [
        "",
        "## Confirmation rule (chosen on validation only)",
        "",
        f"Threshold {chosen['threshold']} ({chosen['rule_outcome']}).",
        f"- Validation: coverage {chosen['coverage']}, selective accuracy {chosen['selective_accuracy']}",
        f"- Test: coverage {conf['coverage']}, selective accuracy {conf['selective_accuracy']}, {conf['needs_confirmation']} images asked for confirmation",
        "",
        "Limits: 12 supported classes only, no non-produce class, no out-of-distribution detection. Source images may be near-related across the original dataset in ways the group split cannot fully rule out.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and evaluate the produce classifier.")
    parser.add_argument("stage", choices=["select", "test"])
    parser.add_argument("--config", default="configs/produce_v1.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(resolve(args.config))
    return select(cfg, args.device) if args.stage == "select" else test(cfg, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
