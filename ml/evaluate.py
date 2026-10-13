"""Evaluation CLI. Requires a real checkpoint and a labeled folder.

Does not convert probabilities into a freshness score or remaining shelf life.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DualPathFreshNet without fabricating metrics.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    if not data_dir.is_dir():
        print(f"Dataset directory not found: {data_dir}", file=sys.stderr)
        return 2
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        print("Model not trained or checkpoint missing. Evaluation aborted.", file=sys.stderr)
        return 2

    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError:
        print("PyTorch/torchvision are not installed. See ml/requirements.txt.", file=sys.stderr)
        return 2

    from dual_path import DualPathFreshNet

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = payload["class_names"]
    model = DualPathFreshNet(num_classes=int(payload["num_classes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predicted = logits.argmax(dim=1)
            correct += int((predicted == labels).sum().item())
            total += int(labels.size(0))

    report = {
        "checkpoint": str(checkpoint),
        "class_names": class_names,
        "image_count": total,
        "accuracy": correct / max(total, 1),
        "output_kind": "visual_freshness_estimate",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
