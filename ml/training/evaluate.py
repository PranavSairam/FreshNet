from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch and matplotlib are required for evaluation.") from exc

from models import build_model
from training.config import (
    downstream_hold_reason,
    load_class_mapping,
    load_train_config,
    resolve_path,
    validate_num_classes,
)

# Provisional original-split fingerprint. The hold file still blocks this split.
ORIGINAL_SPLIT_FINGERPRINT = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"
from training.data import ManifestFreshnessDataset
from training.metrics import compute_classification_metrics
from training.train import pick_device


def plot_confusion(matrix: list[list[float]], labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    split: str,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    device = pick_device(str(config["training"].get("device", "auto")))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = payload["architecture"]
    if split == "test" and bool(payload.get("smoke_test")):
        raise ValueError(f"Refusing to evaluate smoke checkpoint {checkpoint_path} on test.")
    if split == "test" and payload.get("split_fingerprint") == ORIGINAL_SPLIT_FINGERPRINT:
        raise ValueError(
            "Refusing test evaluation of a provisional original-split checkpoint. "
            "artifacts/reviews/hold_downstream_training.json remains in place."
        )
    if config["preprocessing"].get("eval_deterministic") is not True:
        raise ValueError("Evaluation preprocessing must stay deterministic.")
    mapping = payload.get("class_mapping") or load_class_mapping(resolve_path(config["data"]["class_mapping_json"]))
    num_classes = validate_num_classes(config, mapping)
    model = build_model(architecture, payload.get("model_config") or config["model"], num_classes=num_classes)
    model.load_state_dict(payload["state_dict"])
    model.to(device)

    csv_key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
    dataset = ManifestFreshnessDataset(resolve_path(config["data"][csv_key]), config, mapping, train=False)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    metrics, predictions = _score_loader(model, loader, device, architecture)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "architecture": architecture,
            "split": split,
            "split_fingerprint": payload.get("split_fingerprint"),
            "smoke_test": bool(payload.get("smoke_test")),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "decision_rule": "argmax class; no threshold tuning",
            "preprocessing_eval": "deterministic resize and ImageNet normalization; train augmentation disabled",
            "note": (
                "Test split metrics are for final reporting only and must not be used for tuning."
                if split == "test"
                else "Validation metrics."
            ),
        }
    )

    out_dir = output_dir / architecture / split
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_path = out_dir / "predictions.csv"
    _write_predictions(predictions, predictions_path)
    metrics["predictions_path"] = str(predictions_path)
    plot_confusion(
        metrics["confusion_matrix"],
        ["Fresh", "Spoiled"],
        out_dir / "confusion_matrix.png",
        title=f"FreshNet {architecture} {split} counts — rows true, columns predicted",
    )
    plot_confusion(
        [[int(round(v * 1000)) for v in row] for row in metrics["confusion_matrix_normalized"]],
        ["Fresh", "Spoiled"],
        out_dir / "confusion_matrix_normalized_x1000.png",
        title=f"FreshNet {architecture} {split} row-normalized ×1000",
    )
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def _score_loader(model, loader, device, architecture: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One argmax pass. No probability threshold is chosen."""
    model.eval()
    names = {0: "Fresh", 1: "Spoiled"}
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            patches = batch["patches"].to(device)
            labels = [int(v) for v in batch["label"].tolist()]
            if architecture == "context":
                logits = model(context=context)
            elif architecture == "detail":
                logits = model(patches=patches)
            else:
                logits = model(context=context, patches=patches)
            preds = [int(v) for v in logits.argmax(dim=1).tolist()]
            y_true.extend(labels)
            y_pred.extend(preds)
            for path, true_label, pred in zip(batch["image_path"], labels, preds):
                rows.append(
                    {
                        "image_path": path,
                        "true_label": true_label,
                        "true_name": names[true_label],
                        "prediction": pred,
                        "prediction_name": names[pred],
                    }
                )
    return compute_classification_metrics(y_true, y_pred), rows


def _write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_label", "true_name", "prediction", "prediction_name"],
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_architectures(results: list[dict[str, Any]], output_dir: Path) -> Path:
    summary = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "architecture": r["architecture"],
                "split": r["split"],
                "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "macro_f1": r["macro_f1"],
                "spoiled_to_fresh_rate": r["spoiled_to_fresh_rate"],
                "checkpoint": r["checkpoint"],
                "smoke_test": r.get("smoke_test"),
            }
            for r in results
        ],
        "note": (
            "Do not claim fused is better unless measured macro_f1/balanced_accuracy "
            "on the same split and budget support that conclusion."
        ),
    }
    path = output_dir / "architecture_comparison.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FreshNet checkpoints.")
    parser.add_argument("--config", default=str(resolve_path("configs/train.yaml")))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    config = load_train_config(Path(args.config))
    if args.split == "test":
        reason = downstream_hold_reason()
        configured = str(config.get("experiment", {}).get("expected_split_fingerprint") or "")
        if reason and configured == ORIGINAL_SPLIT_FINGERPRINT:
            print(
                json.dumps(
                    {
                        "event": "downstream_hold",
                        "stage": "test_evaluation",
                        "reason": reason,
                        "fingerprint": configured,
                        "note": (
                            "The hold file remains. It blocks test evaluation of the provisional "
                            "original split only. A different fingerprint is not cleared by deleting the hold."
                        ),
                    }
                )
            )
            return 3
        if reason and not configured:
            print(
                json.dumps(
                    {
                        "event": "downstream_hold",
                        "stage": "test_evaluation",
                        "reason": reason,
                        "note": "Config has no expected_split_fingerprint, so the hold still blocks test evaluation.",
                    }
                )
            )
            return 3
    output_dir = resolve_path(args.output_dir or config["evaluation"]["output_dir"])
    metrics = evaluate_checkpoint(
        Path(args.checkpoint),
        split=args.split,
        config=config,
        output_dir=output_dir,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
