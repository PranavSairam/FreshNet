"""Produce-type classifier: EfficientNet-B0 with a new K-way linear head.

Initialised from torchvision ImageNet weights (never from a FreshNet checkpoint).
One output logit per supported produce class; the decision rule is argmax and
softmax is display-only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models


class ProduceNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.initialization = "scratch"
        if pretrained:
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.initialization = "imagenet_efficientnet_b0_DEFAULT"
        else:
            backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=False), nn.Linear(in_features, num_classes))
        self.net = backbone
        self._backbone_frozen = False

    @property
    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.net.classifier.parameters())

    @property
    def backbone_parameters(self) -> list[nn.Parameter]:
        head_ids = {id(p) for p in self.net.classifier.parameters()}
        return [p for p in self.net.parameters() if id(p) not in head_ids]

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone_parameters:
            param.requires_grad = trainable
        self._backbone_frozen = not trainable

    def train(self, mode: bool = True) -> "ProduceNet":
        super().train(mode)
        if mode and self._backbone_frozen:
            # Frozen phase: keep the pretrained BatchNorm running statistics untouched.
            self.net.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_produce_model(num_classes: int, model_cfg: dict[str, Any], pretrained: bool) -> ProduceNet:
    return ProduceNet(num_classes=num_classes, dropout=float(model_cfg.get("dropout", 0.3)), pretrained=pretrained)
