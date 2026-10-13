from __future__ import annotations

import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required to train FreshNet.") from exc

try:
    from torch.amp import GradScaler, autocast

    def _make_scaler(enabled: bool) -> Any:
        return GradScaler("cuda", enabled=enabled)

    def _autocast(enabled: bool):
        return autocast("cuda", enabled=enabled)

except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

    def _make_scaler(enabled: bool) -> Any:
        return GradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        return autocast(enabled=enabled)

from models import build_model
from training.config import (
    downstream_hold_reason,
    load_class_mapping,
    load_train_config,
    resolve_path,
    validate_num_classes,
)
from training.data import ManifestFreshnessDataset, preprocessing_dict
from training.metrics import class_weights_from_labels, compute_classification_metrics

EXPECTED_DEFAULT_FP = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_fingerprint(config: dict[str, Any], fingerprint: str | None) -> None:
    if not config["data"].get("require_fingerprint_match", True):
        return
    expected = (
        config.get("experiment", {}).get("expected_split_fingerprint")
        or EXPECTED_DEFAULT_FP
    )
    if fingerprint != expected:
        raise RuntimeError(
            f"Split fingerprint mismatch. Expected {expected}, got {fingerprint}. "
            "Refusing to train on regenerated/unknown manifests."
        )


def build_param_groups(
    model: nn.Module,
    architecture: str,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    if architecture == "detail" or not hasattr(model, "context"):
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": head_lr, "weight_decay": weight_decay}]

    backbone_params = [p for p in model.context.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in model.context.backbone.parameters()}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups: list[dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "lr": head_lr, "weight_decay": weight_decay, "name": "head_new"})
    if backbone_params:
        groups.append(
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"}
        )
    if not groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return groups


def make_optimizer(model: nn.Module, architecture: str, config: dict[str, Any]) -> torch.optim.Optimizer:
    head_lr = float(config["training"]["learning_rate"])
    backbone_lr = float(config["training"].get("backbone_learning_rate", head_lr * 0.1))
    weight_decay = float(config["training"]["weight_decay"])
    groups = build_param_groups(
        model,
        architecture,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )
    return torch.optim.AdamW(groups)


def make_scheduler(optimizer: torch.optim.Optimizer, remaining_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1))


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception as exc:  # pragma: no cover
            state["mps_error"] = str(exc)
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, architecture: str) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        context = batch["context"].to(device)
        patches = batch["patches"].to(device)
        labels = batch["label"].to(device)
        if architecture == "context":
            logits = model(context=context)
        elif architecture == "detail":
            logits = model(patches=patches)
        else:
            logits = model(context=context, patches=patches)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return compute_classification_metrics(y_true, y_pred)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def forward_batch(model: nn.Module, architecture: str, context, patches):
    if architecture == "context":
        return model(context=context)
    if architecture == "detail":
        return model(patches=patches)
    return model(context=context, patches=patches)


def train_one_model(
    *,
    architecture: str,
    config: dict[str, Any],
    smoke: bool = False,
    benchmark_batches: int | None = None,
) -> dict[str, Any]:
    set_seed(int(config["training"]["seed"]))
    mapping_path = resolve_path(config["data"]["class_mapping_json"])
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing class mapping: {mapping_path}")
    mapping = load_class_mapping(mapping_path)
    num_classes = validate_num_classes(config, mapping)

    fingerprint_path = resolve_path(config["data"]["fingerprint_json"])
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")).get("split_fingerprint")
    validate_fingerprint(config, fingerprint)

    train_csv = resolve_path(config["data"]["train_csv"])
    val_csv = resolve_path(config["data"]["val_csv"])
    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError("Train/val manifests missing.")

    smoke_cfg = config["training"].get("smoke", {})
    max_train = int(smoke_cfg["max_train_samples"]) if smoke else None
    max_val = int(smoke_cfg["max_val_samples"]) if smoke else None
    epochs = int(smoke_cfg["epochs"]) if smoke else int(config["training"]["epochs"])
    batch_size = int(smoke_cfg["batch_size"]) if smoke else int(config["training"]["batch_size"])

    train_ds = ManifestFreshnessDataset(
        train_csv, config, mapping, train=True, max_samples=max_train, seed=int(config["training"]["seed"])
    )
    val_ds = ManifestFreshnessDataset(
        val_csv, config, mapping, train=False, max_samples=max_val, seed=int(config["training"]["seed"])
    )
    train_labels = [int(mapping[r["label"]]) for r in train_ds.rows]
    weights = class_weights_from_labels(train_labels, num_classes=num_classes)

    device = pick_device(str(config["training"].get("device", "auto")))
    model = build_model(architecture, config["model"], num_classes=num_classes).to(device)
    initialization = getattr(model, "initialization", "unknown")

    freeze_epochs = 0 if smoke or architecture == "detail" else int(config["training"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0 and hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(False)

    optimizer = make_optimizer(model, architecture, config)
    remaining = epochs
    scheduler = make_scheduler(optimizer, remaining)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = _make_scaler(use_amp)

    if smoke:
        ckpt_root = resolve_path(config["training"].get("smoke_checkpoint_dir", "artifacts/checkpoints/smoke"))
        hist_root = resolve_path(config["training"].get("smoke_history_dir", "artifacts/metrics/smoke"))
    else:
        ckpt_root = resolve_path(config["training"]["checkpoint_dir"])
        hist_root = resolve_path(config["training"]["history_dir"])
    ckpt_dir = ckpt_root / architecture
    hist_dir = hist_root / architecture
    run_dir = resolve_path(config["training"].get("run_dir", "artifacts/runs")) / (
        f"{'smoke_' if smoke else ''}{architecture}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    best_metric = -1.0
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    history: list[dict[str, Any]] = []
    patience = int(config["training"].get("early_stopping_patience", 7))
    bad_epochs = 0
    start_epoch = 1
    backbone_frozen = freeze_epochs > 0

    resume = config["training"].get("resume")
    resume_notes: list[str] = []
    if resume:
        resume_path = resolve_path(resume)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("smoke_test") and not smoke:
            raise RuntimeError("Refusing to resume a smoke checkpoint into a full training run.")
        model.load_state_dict(payload["state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_metric = float(payload.get("best_metric", -1.0))
        backbone_frozen = bool(payload.get("backbone_frozen", False))
        if hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(not backbone_frozen)
        optimizer = make_optimizer(model, architecture, config)
        if "optimizer_state" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
            except Exception as exc:
                resume_notes.append(f"optimizer_state_not_loaded: {exc}")
        else:
            resume_notes.append("optimizer_state_missing: new AdamW created from config learning rates")
        remaining = max(epochs - start_epoch + 1, 1)
        scheduler = make_scheduler(optimizer, remaining)
        if payload.get("scheduler_state"):
            try:
                scheduler.load_state_dict(payload["scheduler_state"])
            except Exception as exc:
                resume_notes.append(f"scheduler_state_not_loaded: {exc}")
        else:
            resume_notes.append(
                "scheduler_state_missing: CosineAnnealingLR restarted at last_epoch=-1 "
                f"with T_max={remaining} from the learning rates currently in the optimizer. "
                "This is not the original cosine cycle."
            )
        sibling_best = resume_path.parent / "best.pt"
        if sibling_best.is_file() and sibling_best.resolve() != resume_path.resolve():
            best_payload = torch.load(sibling_best, map_location="cpu", weights_only=False)
            sibling_score = float(best_payload.get("best_metric", -1.0))
            if sibling_score > best_metric:
                resume_notes.append(
                    "best_metric_corrected_from_best_pt: "
                    f"continuation checkpoint stored {best_metric}, using {sibling_score}. "
                    "Weights still come from the continuation checkpoint."
                )
                best_metric = sibling_score
        if "bad_epochs" in payload:
            bad_epochs = int(payload["bad_epochs"])
        else:
            bad_epochs = 0
            resume_notes.append("bad_epochs_missing: early-stopping counter reset to 0")
        if payload.get("rng_state"):
            restore_rng_state(payload["rng_state"])
            resume_notes.append("rng_state_restored")
        else:
            resume_notes.append(
                "rng_state_missing: seed was reapplied at process start. "
                "Data-order and augmentation are not an exact continuation."
            )

    env_meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device": str(device),
        "initialization": initialization,
        "smoke_test": smoke,
        "batch_size": batch_size,
        "freeze_backbone_epochs": freeze_epochs,
        "split_fingerprint": fingerprint,
    }
    (run_dir / "environment.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")

    def log(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(json.dumps({"event": "train_start", **env_meta}))
    if resume:
        log(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(resolve_path(resume)),
                    "start_epoch": start_epoch,
                    "best_metric": best_metric,
                    "bad_epochs": bad_epochs,
                    "backbone_frozen": backbone_frozen,
                    "optimizer_groups": len(optimizer.param_groups),
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "notes": resume_notes,
                    "exact_continuation": not any(
                        note.startswith(
                            (
                                "scheduler_state_missing",
                                "rng_state_missing",
                                "bad_epochs_missing",
                                "optimizer_state_missing",
                                "optimizer_state_not_loaded",
                                "scheduler_state_not_loaded",
                            )
                        )
                        for note in resume_notes
                    ),
                }
            )
        )

    # Optional short benchmark path: time N train batches then exit without claiming full training.
    if benchmark_batches is not None:
        model.train()
        timed = 0
        t0 = time.perf_counter()
        for batch in train_loader:
            context = batch["context"].to(device)
            patches = batch["patches"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                logits = forward_batch(model, architecture, context, patches)
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            timed += 1
            if timed >= benchmark_batches:
                break
        elapsed = time.perf_counter() - t0
        steps_per_epoch = max(len(train_loader), 1)
        sec_per_batch = elapsed / max(timed, 1)
        est_epoch = sec_per_batch * steps_per_epoch
        result = {
            "benchmark": True,
            "architecture": architecture,
            "device": str(device),
            "batch_size": batch_size,
            "batches_timed": timed,
            "seconds_timed": round(elapsed, 3),
            "seconds_per_batch": round(sec_per_batch, 3),
            "steps_per_epoch": steps_per_epoch,
            "estimated_seconds_per_epoch": round(est_epoch, 1),
            "estimated_hours_per_epoch": round(est_epoch / 3600, 3),
            "estimated_hours_for_30_epochs": round(est_epoch * 30 / 3600, 2),
            "initialization": initialization,
            "split_fingerprint": fingerprint,
            "smoke_test": False,
            "note": "Measured from timed train batches only; not a full-epoch wall clock.",
        }
        (run_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(json.dumps(result))
        return result

    for epoch in range(start_epoch, epochs + 1):
        # Unfreeze after freeze stage; rebuild optimizer/scheduler so new params are included.
        if backbone_frozen and epoch == freeze_epochs + 1 and hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(True)
            backbone_frozen = False
            optimizer = make_optimizer(model, architecture, config)
            scheduler = make_scheduler(optimizer, max(epochs - epoch + 1, 1))
            log(json.dumps({"event": "unfreeze_backbone", "epoch": epoch, "optimizer_groups": len(optimizer.param_groups)}))

        model.train()
        running_loss = 0.0
        seen = 0
        epoch_t0 = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            context = batch["context"].to(device)
            patches = batch["patches"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                logits = forward_batch(model, architecture, context, patches)
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
            if step == 1 or step % 100 == 0:
                log(
                    json.dumps(
                        {
                            "event": "batch_progress",
                            "epoch": epoch,
                            "step": step,
                            "steps_total": len(train_loader),
                            "loss": round(float(loss.item()), 6),
                        }
                    )
                )
        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_loader(model, val_loader, device, architecture)
        epoch_seconds = time.perf_counter() - epoch_t0
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_spoiled_recall": val_metrics["per_class"]["Spoiled"]["recall"],
            "val_spoiled_to_fresh_errors": val_metrics["spoiled_to_fresh_errors"],
            "lr_groups": [g["lr"] for g in optimizer.param_groups],
            "backbone_frozen": backbone_frozen,
            "epoch_seconds": round(epoch_seconds, 1),
            "smoke_test": smoke,
            "initialization": initialization,
            "split_fingerprint": fingerprint,
        }
        history.append(record)
        log(json.dumps(record))

        metric = float(val_metrics["macro_f1"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
        payload = {
            "architecture": architecture,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "rng_state": capture_rng_state(),
            "num_classes": num_classes,
            "class_mapping": mapping,
            "preprocessing": preprocessing_dict(config),
            "model_config": config["model"],
            "split_fingerprint": fingerprint,
            "val_metrics": val_metrics,
            "train_class_weights": weights,
            "environment": env_meta,
            "initialization": initialization,
            "backbone_frozen": backbone_frozen,
            "smoke_test": smoke,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        if not smoke and bad_epochs >= patience:
            log(json.dumps({"early_stopping": True, "epoch": epoch, "patience": patience}))
            break

    history_path = hist_dir / ("history_smoke.json" if smoke else "history.json")
    history_blob = {
        "architecture": architecture,
        "smoke_test": smoke,
        "split_fingerprint": fingerprint,
        "initialization": initialization,
        "history": history,
        "environment": env_meta,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "run_dir": str(run_dir),
        "note": (
            "SMOKE-TEST RESULTS ONLY — not final performance."
            if smoke
            else "Validation metrics only; test set was not used for tuning."
        ),
    }
    history_path.write_text(json.dumps(history_blob, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history_blob, indent=2), encoding="utf-8")
    result = {
        "architecture": architecture,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(history_path),
        "run_dir": str(run_dir),
        "best_val_macro_f1": best_metric,
        "smoke_test": smoke,
        "epochs_ran": len(history),
        "initialization": initialization,
        "device": str(device),
        "split_fingerprint": fingerprint,
    }
    if not smoke:
        (ckpt_dir / "complete.json").write_text(
            json.dumps(
                {
                    "architecture": architecture,
                    "best_val_macro_f1": best_metric,
                    "epochs_ran_this_process": len(history),
                    "split_fingerprint": fingerprint,
                    "initialization": initialization,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    log(json.dumps({"event": "train_end", **result}))
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train FreshNet binary freshness models.")
    parser.add_argument("--config", default=str(resolve_path("configs/train.yaml")))
    parser.add_argument("--architecture", choices=["context", "detail", "fused"], required=True)
    parser.add_argument("--smoke", action="store_true", help="Tiny subset run; metrics are not final.")
    parser.add_argument("--benchmark-batches", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)

    config = load_train_config(Path(args.config))
    active_fingerprint = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"
    configured_fingerprint = str(config.get("experiment", {}).get("expected_split_fingerprint") or "")
    if args.architecture in {"detail", "fused"} and not args.smoke and args.benchmark_batches is None:
        reason = downstream_hold_reason()
        if reason and configured_fingerprint == active_fingerprint:
            print(
                json.dumps(
                    {
                        "event": "downstream_hold",
                        "architecture": args.architecture,
                        "reason": reason,
                        "fingerprint": configured_fingerprint,
                        "note": "The provisional split is held. Context checkpoints were not modified.",
                    }
                )
            )
            return 3
    if args.resume and configured_fingerprint and configured_fingerprint != active_fingerprint:
        resume_path = str(args.resume).replace("\\", "/")
        if "artifacts/checkpoints/context/" in resume_path or "artifacts/checkpoints/detail/" in resume_path or "artifacts/checkpoints/fused/" in resume_path:
            print(
                json.dumps(
                    {
                        "event": "refuse_old_checkpoint",
                        "resume": args.resume,
                        "reason": "Old provisional checkpoints cannot be resumed onto a corrected split.",
                    }
                )
            )
            return 3
    if args.resume:
        config["training"]["resume"] = args.resume
    if args.device:
        config["training"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.pretrained:
        config["model"].setdefault("context", {})["pretrained"] = True
    if args.no_pretrained:
        config["model"].setdefault("context", {})["pretrained"] = False

    result = train_one_model(
        architecture=args.architecture,
        config=config,
        smoke=args.smoke,
        benchmark_batches=args.benchmark_batches,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    from pathlib import Path

    raise SystemExit(main())
