# Training & evaluation

Binary visual freshness only: **Fresh=0**, **Spoiled=1**. Medium is a documented future extension and is not trained or evaluated.

## Environment (this machine, 2026-10-01)

| Item | Value |
|------|--------|
| Venv | `backend/.venv` (reused; not recreated) |
| Python | 3.12.15 |
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| Pairing | Official: torchvision 0.29.0 is the matching / ABI-stable release for torch 2.14 ([PyTorch 2.14 blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)) |
| CUDA | Not available |
| Accelerator | Apple M5 Metal / MPS (`torch.backends.mps.is_available() == True`) |
| System RAM | 16 GB |
| Disk free | ~139–149 GB on `/System/Volumes/Data` |
| Pins | `ml/requirements.txt` keeps `torch==2.14.0` and `torchvision==0.29.0` after verification |

AMP (`training.amp`) applies on CUDA only. Full runs on this Mac use MPS float32.

```bash
# macOS / Linux from repo root
source backend/.venv/bin/activate
export PYTHONPATH="$PWD/ml"
```

```powershell
# Windows
cd backend
.\.venv\Scripts\Activate.ps1
cd ..
$env:PYTHONPATH = "$pwd\ml"
```

## Frozen experiment inputs

Expected split fingerprint (must match before training):

`005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9`

| Split | Images |
|-------|--------|
| train | 49,978 |
| val | 10,657 |
| test | 10,683 |

Read-only integrity check (2026-10-01; **no** `--repair-splits`):

```bash
export PYTHONPATH="$PWD/ml"
python ml/audit_integrity.py --config configs/dataset.yaml --skip-near-scan
```

Result: exact and known-group cross-split overlap = **0**; existing splits preserved. Near-duplicate scan remains incomplete → experiments are still **provisional** (residual leakage risk without specimen metadata).

Smoke checkpoints live only under `artifacts/checkpoints/smoke/` and `artifacts/metrics/smoke/`. `ml/export_model.py` refuses smoke exports unless `--allow-smoke` is passed. Do not treat smoke metrics as final.

Shared recorded inputs for all full runs (see each run’s `environment.json` / checkpoint metadata):

- ImageNet mean/std and sizes from `configs/train.yaml` (`context_size=224`, `detail_size=384`, `patch_size=160`)
- Class mapping Fresh=0 / Spoiled=1
- Split fingerprint above

## Training configuration (`configs/train.yaml`)

| Setting | Value |
|---------|--------|
| Max epochs | 30 |
| Early stopping | patience 7 on `val_macro_f1` |
| Optimizer | AdamW |
| Head LR | 3e-4 |
| Backbone LR (after unfreeze) | 3e-5 |
| Freeze backbone | 3 epochs (context / fused); rebuild optimizer+cosine scheduler on unfreeze |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 (chosen after MPS memory checks) |
| Seed | 42 |
| Class weights | on |
| Context / fused init | ImageNet EfficientNet-B0 `DEFAULT` when weights resolve |
| Detail init | scratch CNN |

Actual initialization is logged as `initialization` in `train_start` / `environment.json`. Current full context run recorded:

`imagenet_efficientnet_b0_DEFAULT`

## Benchmark (measured; train batches only)

Device: MPS, batch 16, `--benchmark-batches 20` (validation / IO outside timed loop not included).

| Architecture | s/batch | Est. s/epoch (train) | Est. h × 30 epochs |
|--------------|---------|----------------------|--------------------|
| context | 0.113 | 351.8 | 2.93 |
| detail | 0.080 | 250.1 | 2.08 |
| fused | 0.098 | 307.4 | 2.56 |

Artifacts: `artifacts/runs/{context,detail,fused}_20261001T16243*Z/benchmark.json`.

Live context full-run progress (not a completed epoch): step 1000/3124 reached ~135 s after `train_start` ⇒ ~7.4 steps/s ⇒ ~7+ minutes of train batches for epoch 1 before validation. Prefer logged `epoch_seconds` once the first epoch finishes.

## Launch full training (sequential: context → detail → fused)

### Started on this machine (MPS)

```bash
export FRESHNET_DEVICE=mps
export PYTHONPATH="$PWD/ml"
bash scripts/run_full_training.sh
```

| Field | Value |
|-------|--------|
| Orchestrator log | `artifacts/runs/orchestrator/full_train_20261001T163147Z.log` |
| Context run | `artifacts/runs/context_20261001T163149Z/` |
| Device | mps |
| Checkpoints | `artifacts/checkpoints/context/best.pt` and `last.pt` (epoch 5; `smoke_test=false`) |

Measured context epoch wall times (train + val) from that log:

| Epoch | backbone | epoch_seconds | val macro F1 | Spoiled recall | Spoiled→Fresh |
|-------|----------|---------------|--------------|----------------|---------------|
| 1 | frozen | 499.3 | 0.944758 | 0.938406 | 221 |
| 2 | frozen | 535.6 | 0.946179 | 0.958751 | 148 |
| 3 | frozen | 603.0 | 0.954307 | 0.957915 | 151 |
| 4 | unfrozen (2 LR groups) | 1208.5 | 0.971788 | 0.965719 | 123 |
| 5 | unfrozen | 1183.0 | 0.972950 | 0.975753 | 87 |

The first process stopped during epoch 6 (last log line step 400/3124 on 2026-10-02 21:28 local). A later resume (`resume_20261009T130807Z`) stopped at epoch 6 step 200. Neither log contains a Python traceback, and no Python crash report was found. The cause is unknown. An agent-shell exit is a possible explanation, not a verified one. Memory pressure was not ruled out.

Current detached run, started 2026-10-09T13:18:48Z:

```bash
backend/.venv/bin/python scripts/start_detached_training.py
```

That starts `scripts/continue_training.sh` with parent pid 1. It validates the split fingerprint, resumes context from `artifacts/checkpoints/context/last.pt`, then trains detail and fused. It skips an architecture only after `artifacts/checkpoints/<arch>/complete.json` exists, and it stops on the first failure. If `artifacts/runs/orchestrator/train.pid` is still alive, the starter does not launch a second job.

| Field | Value |
|-------|--------|
| Orchestrator | `artifacts/runs/orchestrator/continue_20261009T131848Z.log` |
| Context run | `artifacts/runs/context_20261009T131849Z/` |
| Session pid | 9257 (parent 1) |
| Latest completed epoch | 5, validation macro F1 0.972950 |
| In progress | epoch 6 batches; not a completed result |

Resume of the epoch-5 checkpoint restored weights, `backbone_frozen=false`, and AdamW groups at 2.960e-4 / 2.960e-5. Scheduler state, the early-stopping counter, and RNG state were not in that file. The run log records `exact_continuation: false`. New checkpoints save those fields. `best.pt` is kept for selection; `last.pt` is the continuation file.

Detail and fused have not started. The test set is untouched. Smoke weights remain under `artifacts/checkpoints/smoke/fused/` only.

If this session dies, restart with the same starter command. Do not start it while pid 9257, or whatever pid is in `train.pid`, is alive.

### Windows PowerShell alternative

```powershell
$env:FRESHNET_DEVICE = "cuda"   # or cpu / auto
.\scripts\run_full_training.ps1
```

### Google Colab (if only CPU locally)

Use `notebooks/train_freshnet_colab_gpu.ipynb`. Prefer that over multi-day CPU training. This Mac has usable MPS, so the local sequential job is the primary path.

### Resume

```bash
python ml/train_model.py --architecture context --config configs/train.yaml \
  --device mps --resume artifacts/checkpoints/context/last.pt
```

Smoke checkpoints cannot be resumed into full runs.

## Evaluate (after training finishes)

Do **not** use the test set for selection.

```bash
export PYTHONPATH="$PWD/ml"
python ml/evaluate_model.py --config configs/train_v3.yaml --checkpoint artifacts/checkpoints/v3/context/best.pt --split val
python ml/evaluate_model.py --config configs/train_v3.yaml --checkpoint artifacts/checkpoints/v3/detail/best.pt --split val
python ml/evaluate_model.py --config configs/train_v3.yaml --checkpoint artifacts/checkpoints/v3/fused/best.pt --split val
```

Compare validation macro F1, balanced accuracy, Spoiled precision / recall / F1, and Spoiled→Fresh count and rate. Select the deployment checkpoint by **validation macro F1**, with **higher Spoiled recall** as the tie-break. Fused is not selected unless it wins that rule. If context wins, the deployed model is context-only; keep the dual-path run documented separately.

Only then:

```bash
python ml/evaluate_model.py --config configs/train_v3.yaml \
  --checkpoint artifacts/checkpoints/v3/fused/best.pt --split test \
  --output-dir artifacts/metrics/v3/eval
```

The v3 test pass for that checkpoint is already recorded. The provisional original-split hold still blocks the original fingerprint. The deployment bundle is:

```bash
PYTHONPATH="$PWD/ml" backend/.venv/bin/python ml/export_model.py \
  --checkpoint artifacts/checkpoints/v3/fused/best.pt \
  --bundle-dir artifacts/deployment/v3/fused_epoch29_b962c650ed80
```

## Tests

```bash
export PYTHONPATH="$PWD/ml:$PWD/backend"
python -m pytest tests -q
```
