"""FreshNet binary visual freshness models.

Architectures:
- context: EfficientNet-B0 on 224×224 full image
- detail: shared lightweight CNN over five 160×160 patches from a 384×384 image
- fused: projected concatenation of both branches → 2-logit head

Canonical labels: Fresh=0, Spoiled=1. Medium is a future extension only.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
    from torchvision import models
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch/torchvision are required for FreshNet models. "
        "Install packages from ml/requirements.txt."
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightPatchCNN(nn.Module):
    """Shared CNN for detail patches."""

    def __init__(self, channels: list[int] | None = None, embedding_dim: int = 256) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) or (B*N, 3, H, W)
        feats = self.features(x)
        pooled = self.pool(feats).flatten(1)
        return self.proj(pooled)


class ContextBranch(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        self.pretrained_requested = pretrained
        self.pretrained_loaded = False
        self.initialization = "scratch"
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                backbone = models.efficientnet_b0(weights=weights)
                self.pretrained_loaded = True
                self.initialization = "imagenet_efficientnet_b0_DEFAULT"
            except Exception as exc:  # noqa: BLE001
                backbone = models.efficientnet_b0(weights=None)
                self.initialization = f"scratch_after_pretrained_failure:{type(exc).__name__}"
        else:
            backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class DetailBranch(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.patch_cnn = LightweightPatchCNN(channels=channels, embedding_dim=embedding_dim)
        self.aggregate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, 5, 3, P, P)
        """
        batch, n_patches, channels, height, width = patches.shape
        flat = patches.reshape(batch * n_patches, channels, height, width)
        emb = self.patch_cnn(flat).reshape(batch, n_patches, -1)
        pooled = emb.mean(dim=1)
        return self.aggregate(pooled)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("FreshNet binary head expects num_classes=2 (Fresh, Spoiled).")
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextOnlyFreshNet(nn.Module):
    def __init__(self, num_classes: int = 2, embedding_dim: int = 256, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        self.architecture = "context"
        self.num_classes = num_classes
        self.context = ContextBranch(embedding_dim=embedding_dim, pretrained=pretrained)
        self.head = ClassificationHead(embedding_dim, num_classes=num_classes, dropout=dropout)

    @property
    def initialization(self) -> str:
        return self.context.initialization

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.context.set_backbone_trainable(trainable)

    def forward(self, context: torch.Tensor, patches: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.context(context))


class DetailOnlyFreshNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        channels: list[int] | None = None,
    ):
        super().__init__()
        self.architecture = "detail"
        self.num_classes = num_classes
        self.initialization = "scratch_detail_cnn"
        self.detail = DetailBranch(embedding_dim=embedding_dim, channels=channels)
        self.head = ClassificationHead(embedding_dim, num_classes=num_classes, dropout=dropout)

    def set_backbone_trainable(self, trainable: bool) -> None:
        return None

    def forward(self, context: torch.Tensor | None = None, patches: torch.Tensor | None = None) -> torch.Tensor:
        if patches is None:
            raise ValueError("DetailOnlyFreshNet requires patches.")
        return self.head(self.detail(patches))


class FusedFreshNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = False,
        channels: list[int] | None = None,
    ):
        super().__init__()
        self.architecture = "fused"
        self.num_classes = num_classes
        self.context = ContextBranch(embedding_dim=embedding_dim, pretrained=pretrained)
        self.detail = DetailBranch(embedding_dim=embedding_dim, channels=channels)
        self.head = ClassificationHead(embedding_dim * 2, num_classes=num_classes, dropout=dropout)

    @property
    def initialization(self) -> str:
        return f"context={self.context.initialization};detail=scratch_detail_cnn"

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.context.set_backbone_trainable(trainable)

    def forward(self, context: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        ctx = self.context(context)
        det = self.detail(patches)
        fused = torch.cat([ctx, det], dim=1)
        return self.head(fused)


def build_model(architecture: str, model_cfg: dict[str, Any], num_classes: int = 2) -> nn.Module:
    if num_classes != 2:
        raise ValueError(f"Active FreshNet task is binary; got num_classes={num_classes}.")
    embedding_dim = int(model_cfg.get("embedding_dim", 256))
    dropout = float(model_cfg.get("dropout", 0.3))
    pretrained = bool(model_cfg.get("context", {}).get("pretrained", False))
    channels = model_cfg.get("detail", {}).get("channels")
    arch = architecture.lower()
    if arch == "context":
        return ContextOnlyFreshNet(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if arch == "detail":
        return DetailOnlyFreshNet(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            channels=channels,
        )
    if arch == "fused":
        return FusedFreshNet(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained=pretrained,
            channels=channels,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


# Backward-compatible alias used by older imports/tests.
DualPathFreshNet = FusedFreshNet
