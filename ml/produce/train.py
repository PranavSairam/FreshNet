#!/usr/bin/env python3
"""Train the separate produce-type classifier (EfficientNet-B0, ImageNet init).

Reads only the produce train and validation manifests. The test split is never
opened here. Checkpoints are chosen by validation macro F1. Everything is written
under artifacts/produce/, apart from the freshness artifacts.

    backend/.venv/bin/python ml/produce/train.py --smoke
    backend/.venv/bin/python ml/produce/train.py --resume auto      # real run / continuation
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from produce.data import ProduceDataset, collate  # noqa: E402
from produce.metrics import class_weights_sqrt_inverse, classification_metrics  # noqa: E402
from produce.model import build_produce_model  # noqa: E402


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(spec: str) -> torch.device:
    if spec in ("cpu", "mps", "cuda"):
        if spec == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        if spec == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception as exc:  # pragma: no cover
            state["mps_error"] = str(exc)
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def lr_factor(epoch: int, tr: dict[str, Any]) -> float:
    """Multiplier on the base learning rates for 1-indexed `epoch` (same for both groups)."""
    freeze = int(tr["freeze_backbone_epochs"])
    warmup = int(tr["warmup_epochs"])
    total = int(tr["epochs"])
    final = float(tr["final_lr_fraction"])
    if epoch <= freeze:
        return 1.0
    if epoch <= freeze + warmup:
        return 0.2
    start = freeze + warmup + 1
    t = (epoch - start) / max(total - start, 1)
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


def manifest_fingerprint(all_csv: Path, class_order: list[str], source_fp: str, version: str) -> str:
    import csv

    with all_csv.open(newline="", encoding="utf-8") as handle:
        rows = sorted(
            (r["relative_path"], r["split"], r["produce"], r["group_id"], r["content_sha256"])
            for r in csv.DictReader(handle)
        )
    payload = json.dumps(
        {"manifest_version": version, "source_split_fingerprint": source_fp, "class_order": class_order, "rows": rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_data(cfg: dict[str, Any]) -> tuple[dict[str, int], list[str], str]:
    mapping = json.loads(resolve(cfg["data"]["class_mapping_json"]).read_text(encoding="utf-8"))
    order = list(mapping["label_order"])
    report = json.loads(resolve(cfg["data"]["fingerprint_json"]).read_text(encoding="utf-8"))
    expected = cfg["experiment"]["expected_produce_fingerprint"]
    if report["produce_fingerprint"] != expected:
        raise RuntimeError(f"Fingerprint file {report['produce_fingerprint']} != config {expected}.")
    if report["source_split_fingerprint"] != cfg["experiment"]["expected_source_split_fingerprint"]:
        raise RuntimeError("Source (v3) split fingerprint differs from the config.")
    all_csv = resolve(cfg["data"]["manifests_dir"]) / "all.csv"
    actual = manifest_fingerprint(all_csv, order, report["source_split_fingerprint"], report["manifest_version"])
    if actual != expected:
        raise RuntimeError(f"Manifests do not match the recorded fingerprint ({actual} != {expected}). Refusing to train.")
    if len(order) != int(mapping["num_classes"]):
        raise RuntimeError("Class mapping is inconsistent.")
    return {name: i for i, name in enumerate(order)}, order, expected


class Lock:
    """Exclusive non-blocking file lock so two trainers can never run together."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(f"Another produce trainer holds {path}. Not starting a second one.") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}))
        self.handle.flush()


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, class_order: list[str]) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    return classification_metrics(y_true, y_pred, class_order)


def make_loaders(cfg, class_to_index, smoke: bool, generator: torch.Generator | None):
    tr = cfg["training"]
    smoke_cfg = tr["smoke"]
    seed = int(tr["seed"])
    train_ds = ProduceDataset(
        resolve(cfg["data"]["train_csv"]), cfg, class_to_index, train=True,
        max_samples=int(smoke_cfg["max_train_samples"]) if smoke else None, seed=seed,
    )
    val_ds = ProduceDataset(
        resolve(cfg["data"]["val_csv"]), cfg, class_to_index, train=False,
        max_samples=int(smoke_cfg["max_val_samples"]) if smoke else None, seed=seed,
    )
    batch_size = int(smoke_cfg["batch_size"]) if smoke else int(tr["batch_size"])
    workers = 0 if smoke else int(tr["num_workers"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate,
        generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=workers, collate_fn=collate,
        persistent_workers=workers > 0,
    )
    return train_ds, val_ds, train_loader, val_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the FreshNet produce-type classifier.")
    parser.add_argument("--config", default="configs/produce_v1.yaml")
    parser.add_argument("--smoke", action="store_true", help="Tiny isolated run. Artifacts are never deployable.")
    parser.add_argument("--resume", default=None, help="'auto' (last.pt in the checkpoint dir) or a checkpoint path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch budget (resume testing only).")
    args = parser.parse_args(argv)

    cfg = load_config(resolve(args.config))
    tr = cfg["training"]
    smoke = bool(args.smoke)
    if args.device:
        tr["device"] = args.device
    if smoke:
        tr["epochs"] = int(tr["smoke"]["epochs"])
        tr["freeze_backbone_epochs"] = int(tr["smoke"]["freeze_backbone_epochs"])
        ckpt_dir = resolve(tr["smoke_dir"]) / "checkpoints"
        hist_dir = resolve(tr["smoke_dir"]) / "metrics"
        run_root = resolve(tr["smoke_dir"]) / "runs"
    else:
        ckpt_dir = resolve(tr["checkpoint_dir"])
        hist_dir = resolve(tr["history_dir"])
        run_root = resolve(tr["run_dir"])
    if args.epochs is not None:
        tr["epochs"] = int(args.epochs)
    lock = Lock(ckpt_dir.parent / ("smoke.lock" if smoke else "train.lock"))  # noqa: F841 (held for process lifetime)

    seed = int(tr["seed"])
    set_seed(seed)
    class_to_index, class_order, fingerprint = verify_data(cfg)
    num_classes = len(class_order)
    device = pick_device(str(tr.get("device", "auto")))
    use_amp = device.type == "cuda"  # float32 on MPS and CPU; autocast only on CUDA

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, class_to_index, smoke, generator)
    weights = class_weights_sqrt_inverse(train_ds.labels, num_classes)

    model = build_produce_model(num_classes, cfg["model"], pretrained=(cfg["model"]["pretrained"] == "imagenet")).to(device)
    # The classifier output dimension must equal the number of supported classes.
    assert model.net.classifier[1].out_features == num_classes

    head_lr = float(tr["head_learning_rate"])
    backbone_lr = float(tr["backbone_learning_rate"])
    wd = float(tr["weight_decay"])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head_parameters, "lr": head_lr, "weight_decay": wd, "name": "head"},
            {"params": model.backbone_parameters, "lr": backbone_lr, "weight_decay": wd, "name": "backbone"},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda idx: lr_factor(idx + 1, tr))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(tr["epochs"])
    freeze = int(tr["freeze_backbone_epochs"])
    patience = int(tr["early_stopping_patience"])
    best_metric, best_epoch, bad_epochs, start_epoch = -1.0, 0, 0, 1
    history: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / (("smoke_" if smoke else "") + stamp)
    resume_path: Path | None = None
    if args.resume:
        resume_path = ckpt_dir / "last.pt" if args.resume == "auto" else resolve(args.resume)
        if args.resume == "auto" and not resume_path.is_file():
            resume_path = None
    resume_notes: list[str] = []
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if bool(payload.get("smoke_test")) != smoke:
            raise RuntimeError("Refusing to resume across smoke and real runs.")
        if payload["manifest_fingerprint"] != fingerprint or payload["class_order"] != class_order:
            raise RuntimeError("Checkpoint fingerprint or class mapping differs from the current data. Refusing to resume.")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        best_epoch = int(payload["best_epoch"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload.get("history", []))
        generator.set_state(payload["loader_generator_state"])
        restore_rng(payload["rng_state"])
        run_dir = Path(payload["run_dir"])
        resume_notes.append(f"resumed from {resume_path.name} after epoch {payload['epoch']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(record: dict[str, Any]) -> None:
        line = json.dumps(record)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    env_meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "dtype": "float32 (autocast only on CUDA; none on MPS/CPU)",
        "initialization": model.initialization,
        "smoke_test": smoke,
        "seed": seed,
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "manifest_fingerprint": fingerprint,
        "num_classes": num_classes,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "class_weights": dict(zip(class_order, [round(w, 4) for w in weights])),
        "pid": os.getpid(),
        "run_dir": str(run_dir),
    }
    (run_dir / "environment.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")
    log({"event": "train_start", **env_meta, "resume": resume_notes})

    stop_reason = "max_epochs"
    for epoch in range(start_epoch, epochs + 1):
        frozen = epoch <= freeze
        model.set_backbone_trainable(not frozen)
        model.train()
        epoch_start = time.perf_counter()
        running, seen, steps = 0.0, 0, len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
            if step == 1 or step % 100 == 0 or step == steps:
                log({"event": "batch", "epoch": epoch, "step": step, "steps": steps, "loss": round(float(loss.item()), 5)})
        train_seconds = time.perf_counter() - epoch_start
        lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()
        val_start = time.perf_counter()
        val = evaluate_loader(model, val_loader, device, class_order)
        record = {
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 6),
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "lr_head": lrs[0],
            "lr_backbone": lrs[1],
            "backbone_frozen": frozen,
            "train_seconds": round(train_seconds, 1),
            "val_seconds": round(time.perf_counter() - val_start, 1),
            "epoch_seconds": round(time.perf_counter() - epoch_start, 1),
            "smoke_test": smoke,
        }
        history.append(record)
        log({"event": "epoch", **record})

        improved = record["val_macro_f1"] > best_metric
        if improved:
            best_metric, best_epoch, bad_epochs = record["val_macro_f1"], epoch, 0
        elif not frozen:
            bad_epochs += 1
        payload = {
            "task": "produce_type_identification",
            "architecture": "efficientnet_b0",
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "early_stopping": {"patience": patience, "metric": "val_macro_f1", "counted_after_epoch": freeze},
            "rng_state": capture_rng(),
            "loader_generator_state": generator.get_state(),
            "class_mapping": class_to_index,
            "class_order": class_order,
            "manifest_fingerprint": fingerprint,
            "config": cfg,
            "val_metrics": val,
            "train_class_weights": weights,
            "history": history,
            "environment": env_meta,
            "run_dir": str(run_dir),
            "smoke_test": smoke,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save(payload, ckpt_dir / "last.pt")
        if improved:
            atomic_save(payload, ckpt_dir / "best.pt")
        (hist_dir / ("history_smoke.json" if smoke else "history.json")).write_text(
            json.dumps(
                {
                    "note": "SMOKE ONLY, not performance." if smoke else "Validation metrics only. The test split is not used in training.",
                    "manifest_fingerprint": fingerprint,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_metric,
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not smoke and not frozen and bad_epochs >= patience:
            stop_reason = "early_stopping"
            log({"event": "early_stopping", "epoch": epoch, "patience": patience, "best_epoch": best_epoch})
            break

    if smoke:
        # Checkpoint save/load round trip, output dimensions, mapping and gradient flow.
        reloaded = build_produce_model(num_classes, cfg["model"], pretrained=False)
        saved = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        reloaded.load_state_dict(saved["state_dict"], strict=True)
        reloaded.eval()
        model.eval()
        probe = val_ds[0]["image"].unsqueeze(0)
        with torch.no_grad():
            live = model(probe.to(device)).cpu()
            again = reloaded(probe)
        checks = {
            "output_shape": list(again.shape),
            "output_dim_equals_num_classes": again.shape == (1, num_classes),
            "checkpoint_roundtrip_max_abs_logit_diff": float((live - again).abs().max()),
            "class_mapping_roundtrip": saved["class_mapping"] == class_to_index,
            "class_order": saved["class_order"] == class_order,
            "labels_in_range": all(0 <= y < num_classes for y in train_ds.labels + val_ds.labels),
            "all_classes_seen_in_smoke_train": sorted(set(train_ds.labels)) == list(range(num_classes)),
            "backbone_frozen_in_epoch_1": history[0]["backbone_frozen"] is True,
            "backbone_unfrozen_after": history[-1]["backbone_frozen"] is False,
            "finite_losses": all(math.isfinite(h["train_loss"]) for h in history),
            "val_metrics_present": all("val_macro_f1" in h for h in history),
        }
        reloaded.train()
        reloaded.set_backbone_trainable(True)
        reloaded.zero_grad()
        nn.CrossEntropyLoss()(reloaded(probe.repeat(2, 1, 1, 1)), torch.tensor([0, 1])).backward()
        grads = [p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in reloaded.parameters() if p.requires_grad]
        checks["gradients_finite_on_all_trainable"] = all(grads)
        (hist_dir / "smoke_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        log({"event": "smoke_checks", **checks})

    summary = {
        "event": "train_end",
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "smoke_test": smoke,
        "manifest_fingerprint": fingerprint,
        "run_dir": str(run_dir),
    }
    log(summary)
    (ckpt_dir / "complete.json").write_text(json.dumps(summary | {"completed_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
