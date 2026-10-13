from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Callable

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for FreshNet datasets.") from exc


def extract_corner_center_patches(image: Image.Image, patch_size: int = 160) -> list[Image.Image]:
    width, height = image.size
    if width < patch_size or height < patch_size:
        image = image.resize((max(width, patch_size), max(height, patch_size)), Image.Resampling.BILINEAR)
        width, height = image.size
    coords = [
        (0, 0),
        (width - patch_size, 0),
        (0, height - patch_size),
        (width - patch_size, height - patch_size),
        ((width - patch_size) // 2, (height - patch_size) // 2),
    ]
    return [image.crop((x, y, x + patch_size, y + patch_size)) for x, y in coords]


def build_source_transform(cfg: dict[str, Any], train: bool) -> Callable[[Image.Image], Image.Image]:
    pre = cfg["preprocessing"]
    detail_size = int(pre["detail_size"])
    ops: list[Any] = [transforms.Resize((detail_size, detail_size))]
    if train:
        aug = pre.get("train_augmentation", {})
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=float(aug.get("horizontal_flip_p", 0.5))),
                transforms.RandomRotation(degrees=float(aug.get("rotation_degrees", 12))),
                transforms.ColorJitter(
                    brightness=float(aug.get("color_jitter", {}).get("brightness", 0.08)),
                    contrast=float(aug.get("color_jitter", {}).get("contrast", 0.08)),
                    saturation=float(aug.get("color_jitter", {}).get("saturation", 0.05)),
                    hue=float(aug.get("color_jitter", {}).get("hue", 0.01)),
                ),
            ]
        )
    return transforms.Compose(ops)


def build_tensor_normalize(cfg: dict[str, Any]) -> Callable[[Any], Any]:
    pre = cfg["preprocessing"]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=pre["imagenet_mean"], std=pre["imagenet_std"]),
        ]
    )


class ManifestFreshnessDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        config: dict[str, Any],
        class_mapping: dict[str, int],
        *,
        train: bool,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.class_mapping = class_mapping
        self.train = train
        self.source_tf = build_source_transform(config, train=train)
        self.to_tensor = build_tensor_normalize(config)
        self.context_size = int(config["preprocessing"]["context_size"])
        self.patch_size = int(config["preprocessing"]["patch_size"])

        rows: list[dict[str, str]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("fixture", "").lower() in {"1", "true", "yes"}:
                    continue
                label = row["label"]
                if label not in class_mapping:
                    raise ValueError(f"Label {label!r} not in class mapping {class_mapping}")
                rows.append(row)
        if max_samples is not None and len(rows) > max_samples:
            rng = random.Random(seed)
            rows = rng.sample(rows, max_samples)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["image_path"])
        image = open_rgb_image(path)
        if self.train:
            source = self.source_tf(image)
            context = source.resize(
                (self.context_size, self.context_size),
                Image.Resampling.BILINEAR,
            )
            patches = extract_corner_center_patches(source, patch_size=self.patch_size)
            context_t = self.to_tensor(context)
            patches_t = torch.stack([self.to_tensor(p) for p in patches], dim=0)
        else:
            context_t, patches_t = eval_branch_tensors(image, self.config)
        label = int(self.class_mapping[row["label"]])
        return {
            "context": context_t,
            "patches": patches_t,
            "label": label,
            "image_path": str(path),
        }


def eval_branch_tensors(image: Image.Image, config: dict[str, Any]) -> tuple[Any, Any]:
    """Deterministic eval tensors from an already-decoded RGB image.

    This is the training evaluation path: one 384 resize, then a 224 context
    resize and five 160 patches from that same 384 image. No augmentation.
    """
    if config["preprocessing"].get("eval_deterministic") is not True:
        raise ValueError("Evaluation preprocessing must stay deterministic.")
    source = build_source_transform(config, train=False)(image)
    context_size = int(config["preprocessing"]["context_size"])
    patch_size = int(config["preprocessing"]["patch_size"])
    context = source.resize((context_size, context_size), Image.Resampling.BILINEAR)
    patches = extract_corner_center_patches(source, patch_size=patch_size)
    to_tensor = build_tensor_normalize(config)
    context_t = to_tensor(context)
    patches_t = torch.stack([to_tensor(patch) for patch in patches], dim=0)
    return context_t, patches_t


def open_rgb_image(path: Path) -> Image.Image:
    """Open one still frame and convert to RGB.

    EXIF orientation is not applied. Alpha is flattened by PIL ``convert('RGB')``.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def preprocessing_dict(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_size": config["preprocessing"]["context_size"],
        "detail_size": config["preprocessing"]["detail_size"],
        "patch_size": config["preprocessing"]["patch_size"],
        "imagenet_mean": config["preprocessing"]["imagenet_mean"],
        "imagenet_std": config["preprocessing"]["imagenet_std"],
        "train_augmentation": config["preprocessing"]["train_augmentation"],
        "eval_deterministic": config["preprocessing"]["eval_deterministic"],
        "note": "Source augmentation is applied once before context/detail branch inputs.",
    }
