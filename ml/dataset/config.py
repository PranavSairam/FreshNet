from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset.yaml"


@dataclass
class DatasetConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def source_path(self) -> Path:
        value = Path(self.raw["dataset"]["source_path"])
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def input_mode(self) -> str:
        return str(self.raw["dataset"]["input_mode"])

    @property
    def csv_manifest_path(self) -> Path | None:
        value = self.raw["dataset"].get("csv_manifest_path")
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def image_extensions(self) -> set[str]:
        return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in self.raw["dataset"]["image_extensions"]}

    @property
    def exclude_path_parts(self) -> set[str]:
        return {part.lower() for part in self.raw["dataset"].get("exclude_path_parts", [])}

    @property
    def label_mapping(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.raw["labels"]["mapping"].items()}

    @property
    def intended_classes(self) -> list[str]:
        return list(self.raw["labels"]["intended_classes"])

    @property
    def require_for_three_class(self) -> list[str]:
        return list(self.raw["labels"]["require_for_three_class"])

    @property
    def allow_two_class_training(self) -> bool:
        return bool(self.raw["labels"].get("allow_two_class_training", True))

    @property
    def processed_dir(self) -> Path:
        value = Path(self.raw["output"]["processed_dir"])
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def manifests_dir(self) -> Path:
        value = Path(self.raw["output"]["manifests_dir"])
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def reports_dir(self) -> Path:
        value = Path(self.raw["output"]["reports_dir"])
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def fixtures_dir(self) -> Path:
        value = Path(self.raw["output"]["fixtures_dir"])
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def split_ratios(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["splits"]["ratios"].items()}

    @property
    def split_seed(self) -> int:
        return int(self.raw["splits"]["seed"])


def load_config(path: Path | None = None) -> DatasetConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return DatasetConfig(raw=raw, path=config_path)


def deep_get(config: DatasetConfig, *keys: str, default: Any = None) -> Any:
    node: Any = config.raw
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
