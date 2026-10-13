"""Training CLI. Refuses to run without a real on-disk dataset.

This script does not invent accuracy, dataset size, or learning curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DualPathFreshNet on a local image folder.")
    parser.add_argument("--data-dir", required=True, help="ImageFolder root with one subdirectory per class.")
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Must equal the number of class folders. Do not invent extra classes.",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--metrics-path", default=str(REPO_ROOT / "artifacts" / "metrics" / "history.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args(argv)


def count_images(data_dir: Path) -> tuple[list[str], int]:
    class_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    class_names = [path.name for path in class_dirs]
    counts: Counter[str] = Counter()
    for class_dir in class_dirs:
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            counts[class_dir.name] += len(list(class_dir.glob(suffix)))
    total = sum(counts.values())
    return class_names, total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Dataset directory not found: {data_dir}", file=sys.stderr)
        print("Training aborted. No checkpoint or metrics were written.", file=sys.stderr)
        return 2

    class_names, total = count_images(data_dir)
    if len(class_names) != args.num_classes:
        print(
            f"num-classes={args.num_classes} does not match {len(class_names)} class folders: {class_names}",
            file=sys.stderr,
        )
        print("Do not invent a Medium class for a two-class dataset.", file=sys.stderr)
        return 2

    if total == 0:
        print(f"No images found under {data_dir}. Training aborted.", file=sys.stderr)
        return 2

    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError:
        print("PyTorch/torchvision are not installed. See ml/requirements.txt.", file=sys.stderr)
        return 2

    from dual_path import DualPathFreshNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    if len(dataset.classes) != args.num_classes:
        print("ImageFolder class count does not match --num-classes.", file=sys.stderr)
        return 2

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = DualPathFreshNet(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        seen = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        epoch_record = {
            "epoch": epoch,
            "loss": running_loss / max(seen, 1),
            "accuracy": correct / max(seen, 1),
            "images_seen": seen,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "freshnet.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": args.num_classes,
            "class_names": dataset.classes,
            "architecture": "DualPathFreshNet",
        },
        checkpoint_path,
    )

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "class_names": dataset.classes,
                "image_count": len(dataset),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
