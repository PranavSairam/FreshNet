"""Deterministic produce preprocessing, shared by evaluation, export and the API.

Resize((224, 224)) bilinear with antialiasing (the torchvision default for PIL),
no crop, ToTensor (float32 in [0, 1], CHW), ImageNet normalisation. It has no
randomness. Training augmentation lives in data.py and is never used here.
"""

from __future__ import annotations

from typing import Any

from PIL import Image
from torchvision import transforms

DEFAULT_SPEC = {
    "input_size": 224,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
}


def eval_transform(spec: dict[str, Any]):
    size = int(spec["input_size"])
    return transforms.Compose(
        [
            transforms.Resize((size, size)),  # bilinear + antialias defaults, as in the freshness path
            transforms.ToTensor(),
            transforms.Normalize(mean=list(spec["imagenet_mean"]), std=list(spec["imagenet_std"])),
        ]
    )


def image_to_tensor(image: Image.Image, spec: dict[str, Any]):
    """RGB PIL image -> normalised float tensor (3, size, size)."""
    return eval_transform(spec)(image.convert("RGB"))
