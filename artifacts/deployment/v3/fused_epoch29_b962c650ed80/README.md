# FreshNet v3 fused deployment bundle

Model version: `freshnet-v3-fused-epoch29-b962c650ed80`

Load `weights.pt` with `pretrained=False` and `load_state_dict(..., strict=True)`.
Do not download ImageNet weights. The trained tensors are in this bundle.
Preprocess with `training.data.eval_branch_tensors` after `open_rgb_image`.
Predict with argmax on the two logits. Softmax is for display only.

```bash
PYTHONPATH=backend:ml backend/.venv/bin/python -m app.services.inference \
  --bundle artifacts/deployment/v3/fused_epoch29_b962c650ed80 \
  --image path/to/image.jpg
```
