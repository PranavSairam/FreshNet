"""Manifest dataset for the produce classifier."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from produce.preprocess import eval_transform


def open_rgb(path: Path) -> Image.Image:
    """Decode one still frame to RGB. EXIF orientation is not applied (same as the freshness path)."""
    with Image.open(path) as img:
        return img.convert("RGB")


def train_transform(cfg: dict[str, Any]):
    pre = cfg["preprocessing"]
    aug = pre["train_augmentation"]
    size = int(pre["input_size"])
    jitter = aug["color_jitter"]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=tuple(aug["random_resized_crop"]["scale"]),
                ratio=tuple(aug["random_resized_crop"]["ratio"]),
            ),
            transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip_p"])),
            transforms.RandomRotation(degrees=float(aug["rotation_degrees"])),
            transforms.ColorJitter(
                brightness=float(jitter["brightness"]),
                contrast=float(jitter["contrast"]),
                saturation=float(jitter["saturation"]),
                hue=float(jitter["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(pre["imagenet_mean"]), std=list(pre["imagenet_std"])),
        ]
    )


def read_manifest(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ProduceDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        cfg: dict[str, Any],
        class_to_index: dict[str, int],
        *,
        train: bool,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        rows = read_manifest(csv_path)
        for row in rows:
            if row["produce"] not in class_to_index:
                raise ValueError(f"Produce class {row['produce']!r} is not in the class mapping.")
            if int(row["produce_index"]) != class_to_index[row["produce"]]:
                raise ValueError(f"Manifest index disagrees with the class mapping for {row['produce']!r}.")
        if max_samples is not None and len(rows) > max_samples:
            # Class-stratified subset (round robin over classes) so a small smoke run still
            # exercises every class. Random within each class, deterministic from the seed.
            rng = random.Random(seed)
            by_class: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                by_class.setdefault(row["produce"], []).append(row)
            for members in by_class.values():
                rng.shuffle(members)
            picked: list[dict[str, str]] = []
            while len(picked) < max_samples and any(by_class.values()):
                for name in sorted(by_class):
                    if by_class[name] and len(picked) < max_samples:
                        picked.append(by_class[name].pop())
            rows = picked
        self.rows = rows
        self.class_to_index = class_to_index
        self.transform = train_transform(cfg) if train else eval_transform(cfg["preprocessing"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = open_rgb(Path(row["image_path"]))
        return {
            "image": self.transform(image),
            "label": int(self.class_to_index[row["produce"]]),
            "index": index,
        }

    @property
    def labels(self) -> list[int]:
        return [int(self.class_to_index[r["produce"]]) for r in self.rows]


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
