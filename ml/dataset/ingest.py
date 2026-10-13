from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import REPO_ROOT, DatasetConfig, deep_get
from .labels import map_label
from .records import ImageRecord

OFFICIAL_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def _is_excluded(path: Path, source: Path, exclude_parts: set[str]) -> bool:
    """Exclude nested fixture trees when scanning a broader data root."""
    try:
        rel_parts = path.resolve().relative_to(source.resolve()).parts
    except ValueError:
        rel_parts = path.parts
    return any(part.lower() in exclude_parts for part in rel_parts)


def _infer_food_type(path: Path, class_dir_name: str) -> str | None:
    parts = path.parts
    try:
        idx = parts.index(class_dir_name)
    except ValueError:
        return None
    if idx + 1 < len(parts) - 1:
        return parts[idx + 1]
    return None


def _infer_official_split(path: Path, official_names: set[str]) -> str | None:
    for part in path.parts:
        key = part.lower()
        if key in official_names:
            return OFFICIAL_SPLIT_ALIASES.get(key, key)
    return None


def _strip_aug_prefix(stem: str, prefixes: list[str]) -> str:
    value = stem
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if value.lower().startswith(prefix.lower()):
                value = value[len(prefix) :]
                changed = True
    return value


def derive_group_id(record: ImageRecord, prefixes: list[str]) -> str:
    if record.specimen_id and record.capture_session:
        return f"specimen:{record.specimen_id}|session:{record.capture_session}"
    if record.specimen_id:
        return f"specimen:{record.specimen_id}"
    if record.capture_session:
        return f"session:{record.capture_session}"

    path = Path(record.image_path)
    stem = _strip_aug_prefix(path.stem, prefixes)
    stem = re.sub(r"\s+", " ", stem).strip().lower()
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    food = record.food_type or "unknown_food"
    return f"heuristic:{food}:{stem}"


def ingest_class_folders(config: DatasetConfig) -> list[ImageRecord]:
    root = config.source_path
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset source not found: {root}. Place class folders there or update "
            "configs/dataset.yaml dataset.source_path."
        )

    official_names = {
        name.lower()
        for name in deep_get(config, "splits", "official_split_dir_names", default=[])
    }
    prefixes = list(deep_get(config, "grouping", "augmentation_prefixes", default=[]))
    records: list[ImageRecord] = []

    top_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]

    def walk_class_tree(class_dir: Path, label_raw: str, split_official: str | None) -> None:
        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in config.image_extensions:
                continue
            if _is_excluded(path, root, config.exclude_path_parts):
                continue
            food_type = _infer_food_type(path, class_dir.name)
            record = ImageRecord(
                image_path=str(path.resolve()),
                label_raw=label_raw,
                label=map_label(label_raw, config.label_mapping),
                food_type=food_type,
                source=str(config.source_path.resolve()),
                split_official=split_official or _infer_official_split(path, official_names),
            )
            record.group_id = derive_group_id(record, prefixes)
            records.append(record)

    split_like = [d for d in top_dirs if d.name.lower() in official_names]
    class_like = [d for d in top_dirs if d.name.lower() not in official_names]

    if split_like and not class_like:
        for split_dir in split_like:
            split_name = OFFICIAL_SPLIT_ALIASES[split_dir.name.lower()]
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                walk_class_tree(class_dir, class_dir.name, split_name)
    else:
        for class_dir in class_like:
            walk_class_tree(class_dir, class_dir.name, None)

    return records


def ingest_csv_manifest(config: DatasetConfig) -> list[ImageRecord]:
    manifest = config.csv_manifest_path
    if manifest is None or not manifest.is_file():
        raise FileNotFoundError(
            "CSV manifest mode requires dataset.csv_manifest_path pointing to an existing file "
            "with columns image_path,label and optional food_type,specimen_id,"
            "capture_session,source,split."
        )

    prefixes = list(deep_get(config, "grouping", "augmentation_prefixes", default=[]))
    records: list[ImageRecord] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV manifest must include image_path and label columns.")
        for row in reader:
            image_path = Path(row["image_path"].strip())
            if not image_path.is_absolute():
                image_path = (REPO_ROOT / image_path).resolve()
            label_raw = row["label"].strip()
            split_raw = (row.get("split") or "").strip().lower() or None
            split_official = OFFICIAL_SPLIT_ALIASES.get(split_raw, split_raw) if split_raw else None
            record = ImageRecord(
                image_path=str(image_path),
                label_raw=label_raw,
                label=map_label(label_raw, config.label_mapping),
                food_type=(row.get("food_type") or None) or None,
                specimen_id=(row.get("specimen_id") or None) or None,
                capture_session=(row.get("capture_session") or None) or None,
                source=(row.get("source") or None) or str(manifest),
                split_official=split_official,
                fixture=str(row.get("fixture", "")).lower() in {"1", "true", "yes"},
            )
            record.group_id = derive_group_id(record, prefixes)
            records.append(record)
    return records


def ingest(config: DatasetConfig) -> list[ImageRecord]:
    mode = config.input_mode
    if mode == "class_folders":
        return ingest_class_folders(config)
    if mode == "csv_manifest":
        return ingest_csv_manifest(config)
    raise ValueError(f"Unsupported input_mode: {mode}")
