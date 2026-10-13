from __future__ import annotations

from typing import Any


def build_train_transform_spec(config_raw: dict[str, Any]) -> dict[str, Any]:
    train = config_raw.get("transforms", {}).get("train", {})
    size = int(config_raw.get("transforms", {}).get("image_size", 224))
    return {
        "split": "train",
        "image_size": size,
        "operations": [
            {"name": "resize", "size": [size, size]},
            {
                "name": "random_horizontal_flip",
                "p": float(train.get("horizontal_flip_p", 0.5)),
            },
            {
                "name": "random_rotation",
                "degrees": float(train.get("rotation_degrees", 12)),
            },
            {
                "name": "color_jitter",
                "params": train.get(
                    "color_jitter",
                    {"brightness": 0.08, "contrast": 0.08, "saturation": 0.05, "hue": 0.01},
                ),
                "note": "Mild on purpose; discoloration is a freshness cue.",
            },
            {"name": "to_tensor"},
        ],
        "augmentation_scope": "training_only",
    }


def build_eval_transform_spec(config_raw: dict[str, Any]) -> dict[str, Any]:
    size = int(config_raw.get("transforms", {}).get("image_size", 224))
    return {
        "split": "eval",
        "image_size": size,
        "deterministic": True,
        "operations": [
            {"name": "resize", "size": [size, size]},
            {"name": "to_tensor"},
        ],
        "augmentation_scope": "none",
    }


def try_build_torchvision_transforms(config_raw: dict[str, Any]) -> dict[str, Any]:
    """Return live torchvision transforms when torch is installed; otherwise specs only."""
    specs = {
        "train_spec": build_train_transform_spec(config_raw),
        "eval_spec": build_eval_transform_spec(config_raw),
    }
    try:
        from torchvision import transforms
    except ImportError:
        specs["torchvision_available"] = False
        return specs

    size = int(config_raw.get("transforms", {}).get("image_size", 224))
    train_cfg = config_raw.get("transforms", {}).get("train", {})
    jitter = train_cfg.get("color_jitter", {})
    train = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=float(train_cfg.get("horizontal_flip_p", 0.5))),
            transforms.RandomRotation(degrees=float(train_cfg.get("rotation_degrees", 12))),
            transforms.ColorJitter(
                brightness=float(jitter.get("brightness", 0.08)),
                contrast=float(jitter.get("contrast", 0.08)),
                saturation=float(jitter.get("saturation", 0.05)),
                hue=float(jitter.get("hue", 0.01)),
            ),
            transforms.ToTensor(),
        ]
    )
    eval_t = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    specs["torchvision_available"] = True
    specs["train_transform"] = train
    specs["eval_transform"] = eval_t
    return specs
