from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models import FusedFreshNet


def test_fused_forward_shape_two_classes() -> None:
    model = FusedFreshNet(num_classes=2, embedding_dim=32, pretrained=False)
    context = torch.randn(4, 3, 224, 224)
    patches = torch.randn(4, 5, 3, 160, 160)
    logits = model(context=context, patches=patches)
    assert logits.shape == (4, 2)


def test_rejects_non_binary_head() -> None:
    with pytest.raises(ValueError):
        FusedFreshNet(num_classes=3, embedding_dim=32, pretrained=False)


def test_parameter_gradients_exist() -> None:
    model = FusedFreshNet(num_classes=2, embedding_dim=32, pretrained=False)
    context = torch.randn(2, 3, 224, 224)
    patches = torch.randn(2, 5, 3, 160, 160)
    loss = model(context=context, patches=patches).sum()
    loss.backward()
    grads = [param.grad is not None for param in model.parameters() if param.requires_grad]
    assert grads
    assert all(grads)
