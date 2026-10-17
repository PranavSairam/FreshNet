# FreshNet produce-type classifier (v1)

A separate model that names the produce type in a photo. It is independent of the freshness model: its own data manifests, weights, preprocessing, version and evaluation. The freshness model, its bundle (`freshnet-v3-fused-epoch29-b962c650ed80`), its preprocessing, argmax rule and recorded evaluation were not changed or retrained. **The freshness model's 98.33 % accuracy says nothing about produce recognition, and the numbers below say nothing about freshness.**

Status as of 2026-10-16: trained, evaluated once on the test subset, exported and integrated. All numbers below are copied from `artifacts/produce/v1/metrics/eval/` (`selection.json`, `test_results.json`, `EVALUATION.md`).

## Classes and aliases

Twelve classes. Index = alphabetical position. The source data has produce type only as an image folder name; there are no specimen ids.

| Index | Class | Source folders merged into it |
| --- | --- | --- |
| 0 | apple | Apple |
| 1 | banana | Banana |
| 2 | bitter_gourd | Bittergroud (assumed misspelling of bitter gourd) |
| 3 | capsicum (bell pepper) | Bellpepper, Capciscum, Capsicum |
| 4 | carrot | Carrot |
| 5 | cucumber | Cucumber |
| 6 | mango | Mango |
| 7 | okra | Okara, Okra |
| 8 | orange | Orange |
| 9 | potato | Potato |
| 10 | strawberry | Strawberry |
| 11 | tomato | Tomato |

The alias merges (bell pepper = capsicum, Okara = okra, Bittergroud = bitter gourd) were checked against sample images. The full audit, counts and reasoning are in `PRODUCE_DATASET_V1.md` (and `produce_dataset_v1.json`).

## Data audit and exclusions

- Source: the frozen v3 manifests (`data/processed/manifests_v3_source/`), read only (hashes checked before and after). Split assignments and source-group boundaries are exactly those of v3. No source group crosses a split.
- Produce manifests: `data/processed/manifests_produce_v1/` (`train/val/test/all.csv`, `class_mapping.json`, `produce_fingerprint.json`).
- **Produce fingerprint: `9e3a27855e6e01d97a32fee61273dde59b1df7e3166ef25e764e03b97c872c66`**. Training and evaluation recompute it and refuse to run on a mismatch.
- Included: train 37,162, validation 7,853, test 8,035 (53,050 total).
- Excluded: 2 train images, both from one source group that shows a carrot, a cucumber and a cabbage together (it sits under the RottenBellpepper and RottenCarrot folders), so it has no single honest produce label.
- The dataset has no non-produce images, so the model has no "other" class.

## Architecture and training

| Setting | Value |
| --- | --- |
| Model | torchvision EfficientNet-B0, ImageNet-initialised (`EfficientNet_B0_Weights.DEFAULT`), new head `Dropout(0.3)` + `Linear(1280, 12)` |
| Loss | Cross-entropy with class weights `(N / (K * n_c)) ** 0.5` from training labels only |
| Input | `Resize((224, 224))` bilinear with antialiasing, no crop, ImageNet mean/std. Deterministic at evaluation and in the API |
| Training augmentation (train only) | RandomResizedCrop(224, scale 0.6-1.0), horizontal flip 0.5, rotation 12 degrees, small colour jitter |
| Optimiser | AdamW, weight decay 0.01. Head lr 5e-4, backbone lr 1e-4 |
| Schedule | Epochs 1-2 backbone frozen (BatchNorm kept in eval mode); epoch 3 warm-up at 0.2x; then cosine decay to 0.02x |
| Budget | Up to 20 epochs, early stopping patience 5 counted after the backbone is unfrozen |
| Selection | Highest validation macro F1 |
| Seed / batch / device | 42 / 32 / Apple MPS, float32 (torch 2.14.0, torchvision 0.29.0) |

Run outcome: stopped by early stopping after 15 epochs; best epoch 10 (validation macro F1 0.987662). Measured wall-clock time from the run log: about 160 s per epoch while the backbone was frozen (epochs 1-2) and about 460 s per epoch afterwards; 1.75 hours in total. The code is `ml/produce/`, the config is `configs/produce_v1.yaml`, and the run log is `artifacts/produce/v1/runs/20261016T175635Z/train.log`.

Before the real run, a smoke check on a 96-image subset confirmed class mapping, the 12-way output, finite loss and gradients, freeze/unfreeze, and a checkpoint round trip. Resume was tested across two processes and reproduced the uninterrupted run's epoch-2 validation F1 exactly. Smoke artifacts live under `artifacts/produce/smoke/` and the loaders refuse them.

## Selected checkpoint

- `artifacts/produce/v1/checkpoints/best.pt`, SHA-256 `e389c317c8e5df8f8434dd9025737afe4fb48b39e17d4ee63e87df78cfc759fa`, epoch 10.
- Recorded in `selection.json` **before** the test split was opened (`test_split_opened: false` at that point), together with the class mapping, preprocessing and fingerprint. `evaluate.py test` refuses to run if the checkpoint hash differs or if `test_results.json` already exists.

## Results

Task: 12-class produce type. The test subset is the produce part of the existing v3 holdout. Those images were also the freshness model's test images, so this is a **separate-task evaluation on the existing holdout, not a new independent external test**.

| Split | Images | Accuracy | Macro F1 | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Validation (used for selection) | 7,853 | 0.9929 | 0.9877 | 0.9898 |
| Test (evaluated once) | 8,035 | 0.9873 | 0.9781 | 0.9856 |

Per-class test results:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| apple | 0.9949 | 0.9790 | 0.9869 | 1000 |
| banana | 1.0000 | 0.9986 | 0.9993 | 710 |
| bitter_gourd | 1.0000 | 1.0000 | 1.0000 | 119 |
| capsicum | 0.9318 | 0.9927 | 0.9613 | 413 |
| carrot | 0.9964 | 0.9952 | 0.9958 | 1666 |
| cucumber | 0.9610 | 0.9471 | 0.9540 | 208 |
| mango | 0.9427 | 0.9737 | 0.9579 | 152 |
| okra | 0.9136 | 1.0000 | 0.9548 | 74 |
| orange | 0.9942 | 0.9840 | 0.9891 | 1565 |
| potato | 0.9307 | 0.9691 | 0.9495 | 291 |
| strawberry | 0.9919 | 1.0000 | 0.9959 | 122 |
| tomato | 0.9965 | 0.9883 | 0.9924 | 1715 |

The confusion matrix is in `EVALUATION.md`. The largest confusions: tomato predicted as capsicum (14 of 1,715), apple as potato (8), orange as potato (8), cucumber as okra (7), apple as capsicum (6). Small classes (okra 74, bitter gourd 119, strawberry 122, mango 152) have few test images, so their scores are noisy.

Fresh / Spoiled subgroups (produce-type correctness only): Fresh images 0.9930 (n = 5,309); Spoiled images 0.9762 (n = 2,726). Spoiled produce is recognised noticeably less often.

Caveats that apply to these numbers:

- Within a produce folder, images share camera, background and file style, and some classes lean mostly Fresh or mostly Spoiled. The model may use such cues. The group split removes known duplicates and augmented copies of the same source, not every near-relationship.
- Everything above is about the supported types. It says nothing about other produce, mixed scenes or non-produce photos.

## Confirmation rule ("Please confirm produce type")

The model has no "not produce" class and is not an out-of-distribution detector. Its softmax is only a comparison among the 12 trained types. A low top score is treated as a request to confirm, never as proof the item is unsupported, and a high score is not proof it is supported.

Rule, fixed before looking at results and applied to **validation only**: the smallest threshold in `[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]` whose accuracy on accepted validation images is at least 0.99. It chose **0.5**. Trade-off on validation:

| Threshold | Coverage | Selective accuracy | Asked to confirm |
| --- | --- | --- | --- |
| 0.5 (chosen) | 0.9995 | 0.9932 | 4 |
| 0.7 | 0.9955 | 0.9949 | 35 |
| 0.9 | 0.9884 | 0.9964 | 91 |
| 0.95 | 0.9838 | 0.9973 | 127 |
| 0.98 | 0.9757 | 0.9980 | 191 |

At the chosen threshold the test subset showed coverage 0.9993, selective accuracy 0.9879 and 6 images asked for confirmation (reported, not tuned on test). Because the model is already accurate on its own classes, the chosen threshold rarely fires, so in practice it mostly catches only very uncertain images. A stricter threshold is a product decision: raising it to 0.9 would send about 1.2 % of validation images to confirmation. Choosing a different threshold means re-exporting the bundle with a new validation-chosen value (the checksums are regenerated); do not retune it on the test split.

Informal observation, not an evaluation: pictures with several items or non-dataset subjects still get confident guesses. In a spot check the 1901 radish catalogue drawing was labelled carrot at 0.999 and a market stall photo bell pepper at 0.83. This is why the website lets the user correct the type and why this model must not be described as detecting unsupported items.

## Deployment bundle

`artifacts/deployment/produce_v1/effb0_epoch10_e389c317c8e5/`, model version `freshnet-produce-v1-effb0-e389c317c8e5`, schema `freshnet.produce.deployment.v1`. Files: `weights.pt`, `architecture.json`, `class_mapping.json` (with folder aliases), `preprocessing.json`, `metadata.json` (provenance, fingerprint, checkpoint hash, validation and test metrics kept separate, confirmation rule), `checksums.json`, `parity_check.json`, `README.md`.

Export parity check (`parity_check.json`): the exported bundle loaded through the backend loader and the source checkpoint were compared on 48 validation images spread across the split; maximum absolute logit difference 0.0 and 48/48 matching predictions (both on CPU).

The loader verifies every checksum, refuses smoke artifacts, requires `smoke_test: false`, a consistent class mapping and the deterministic preprocessing flags, and never downloads weights.

## How it is used by the application

- On each upload the backend runs the freshness model and the produce model on the same validated image, each with its own preprocessing. Outputs are kept apart: `freshness` (label, probabilities, version) and `produce` / `identification` (id, probabilities, version). Probabilities are never multiplied or merged.
- `produce.source` is `model_predicted`, `user_selected` or `unknown`. A user's choice always wins. If the produce model is missing, fails, or is unsure, identity falls back to Unknown / manual selection and the freshness result is unaffected.
- The combined label ("Predicted: Fresh banana") is composed on the server from the freshness class and the catalogue noun.
- After a result, the website's "Correct produce type" control updates the heading and guidance through `GET /api/guidance` without re-running either model. The filename is never used.
- `GET /api/health` reports the produce model separately (`produce_model`). `GET /api/produce` reports whether automatic identification is available, its version, threshold and supported ids.
- Setting: `FRESHNET_PRODUCE_BUNDLE_DIR` (default is the bundle above; an empty or missing path disables automatic identification and the website falls back to manual selection).

## Limitations

- Recognises only the 12 types above. No "other" class, no detection of unsupported items, no multi-item handling.
- Test numbers are from the existing holdout and a separate task, not an independent external test. Real-world photos (lighting, backgrounds, cameras) may behave differently.
- Spoiled produce is recognised less reliably than fresh produce (0.976 vs 0.993 on the test subset).
- Spoilage severity is still "Not assessed".

## Commands

```bash
# Rebuild the manifests (writes data/processed/manifests_produce_v1 and the audit docs)
python ml/produce/build_manifests.py

# Smoke check (isolated, never deployable)
backend/.venv/bin/python ml/produce/train.py --smoke

# Start or resume real training, detached from the terminal (refuses to start a second trainer)
backend/.venv/bin/python scripts/start_detached_produce.py

# Monitor
tail -f artifacts/produce/v1/runs/*/train.log
cat artifacts/produce/v1/orchestrator/train.pid        # PID
grep '"event": "epoch"' artifacts/produce/v1/runs/*/train.log

# After training: select on validation, evaluate once, export
backend/.venv/bin/python ml/produce/evaluate.py select
backend/.venv/bin/python ml/produce/evaluate.py test
backend/.venv/bin/python ml/produce/export_bundle.py

# Run the app with the produce bundle (default) or disable it
FRESHNET_PRODUCE_BUNDLE_DIR=artifacts/deployment/produce_v1/effb0_epoch10_e389c317c8e5 \
  PYTHONPATH=ml backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend

# Tests
backend/.venv/bin/python -m pytest tests
```

If training is interrupted, run the start command again: it resumes from `artifacts/produce/v1/checkpoints/last.pt` with optimiser, scheduler, early-stopping and data-order state.
