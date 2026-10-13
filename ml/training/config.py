from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from models import build_model  # noqa: F401 — re-export helper used by train scripts

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_train_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (REPO_ROOT / "configs" / "train.yaml")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


DOWNSTREAM_HOLD = "artifacts/reviews/hold_downstream_training.json"


def downstream_hold_reason() -> str | None:
    """Return the hold reason when later experiments must not start."""
    path = resolve_path(DOWNSTREAM_HOLD)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "hold file is present"
    return str(payload.get("reason") or "hold file is present")


def load_class_mapping(path: Path) -> dict[str, Any]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("Fresh") != 0 or mapping.get("Spoiled") != 1:
        raise ValueError(f"Unexpected class mapping in {path}: {mapping}")
    if int(mapping.get("num_classes", 2)) != 2:
        raise ValueError(f"num_classes must be 2 for binary FreshNet, got {mapping}")
    return mapping


def validate_num_classes(config: dict[str, Any], mapping: dict[str, Any]) -> int:
    configured = int(config["task"]["num_classes"])
    mapped = int(mapping["num_classes"])
    if configured != mapped:
        raise ValueError(
            f"Config num_classes={configured} does not match class_mapping.json num_classes={mapped}."
        )
    if configured != 2:
        raise ValueError("Active task is binary classification (Fresh=0, Spoiled=1).")
    labels = config["task"]["labels"]
    if labels.get("Fresh") != 0 or labels.get("Spoiled") != 1:
        raise ValueError(f"Config labels must be Fresh=0, Spoiled=1; got {labels}")
    return configured
