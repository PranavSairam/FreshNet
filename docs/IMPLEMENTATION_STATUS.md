# Implementation Status

Last updated: 2026-10-09T13:55Z

Current stage: **Provisional context training is still running on the original manifests. A verified source-image split is prepared for a later fresh run. That run has not been started.**

## Environment (verified)

- Python **3.12.15** in reused `backend/.venv`
- **torch 2.14.0** + **torchvision 0.29.0** (official matching / ABI-stable pair for torch 2.14; pins kept in `ml/requirements.txt`)
- CUDA: unavailable
- Accelerator: **Apple M5** Metal / **MPS** available; 16 GB system RAM
- Disk free: ~139–149 GB
- AMP: CUDA-only in code; this run uses MPS float32

## Dataset freeze

- Fingerprint: `005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9`
- Train 49,978 / val 10,657 / test 10,683
- Exact SHA-256 and known-group cross-split overlap: **0**. Splits were not regenerated.
- Near-duplicate stages, checked 2026-10-09:
  - Hash coverage: **complete**. `data/processed/cache/image_hashes.jsonl` has 71,318 unique paths and 71,318/71,318 lines in `near_scan_run.log`.
  - Candidate search: **complete as a banded search**, not an all-pairs search. The scan log records 418,572 candidate pairs and `complete True`.
  - Candidate review: **not done**. `near_duplicate_candidates.csv` stores 200 representative pairs only. All 200 have Hamming 0; 57 of them cross splits.
- `integrity_audit.json` currently says the near scan was skipped. That file was overwritten by a later `audit_integrity.py --skip-near-scan` run (`generated_at` 2026-10-01T16:32:37Z). It does not retract the completed hash cache or the scan log.
- The full 418,572-pair cross-split count was not saved. The representative file already shows related-looking images across splits, so these experiments stay **provisional**. Correcting splits would require new experiments. Splits were not changed.
- No specimen metadata, so residual leakage risk remains even aside from the near-duplicate candidates.
- Smoke artifacts stay under `artifacts/checkpoints/smoke/` and are excluded from export.

## Models & config

- Context / fused: ImageNet EfficientNet-B0 `DEFAULT` when weights resolve (recorded in run `environment.json`)
- Detail: scratch CNN
- Max 30 epochs; early stop patience 7 on val macro F1; AdamW; freeze 3 epochs then unfreeze with rebuilt optimizer/scheduler; batch 16; cosine schedule
- Config: `configs/train.yaml`

## Benchmarks (measured train batches on MPS, batch 16)

| Arch | Est. train s/epoch | Est. h × 30 (train only) |
|------|--------------------|---------------------------|
| context | 351.8 | 2.93 |
| detail | 250.1 | 2.08 |
| fused | 307.4 | 2.56 |

See `artifacts/runs/*_20261001T16243*Z/benchmark.json`.

Measured context full-epoch wall times (train + val) from `context_20261001T163149Z/train.log`: frozen epochs 499–603 s; unfrozen epochs 4–5 about 1183–1209 s.

## Full training status

| Item | Value |
|------|--------|
| First command | `FRESHNET_DEVICE=mps bash scripts/run_full_training.sh` |
| First orchestrator | `artifacts/runs/orchestrator/full_train_20261001T163147Z.log` |
| First context run | `artifacts/runs/context_20261001T163149Z/` |
| Init recorded | `imagenet_efficientnet_b0_DEFAULT` |
| Completed epochs | 1–5 only. Best validation macro F1 **0.972950** at epoch 5 (balanced accuracy 0.975710, Spoiled precision 0.953172, Spoiled recall 0.975753, Spoiled F1 0.964330, Spoiled→Fresh 87 / rate 0.024247). |
| Checkpoints | `artifacts/checkpoints/context/last.pt` is the continuation checkpoint (epoch 5 weights and AdamW state). `best.pt` is the same epoch and is reserved for selection. Both are non-smoke and fingerprint-matched. |
| Interruptions | The first run stopped during epoch 6 (last log line step 400/3124, mtime 2026-10-02 21:28 local). The 13:08Z resume stopped at epoch 6 step 200. Neither log has a Python traceback. No Python crash report was found under DiagnosticReports. A log sample around 18:35–18:50 local on Oct 9 showed jetsam messages for a different app, not this training process. The cause is **unknown**. Attachment to the agent shell is one possible explanation, not a verified cause. Memory pressure was not ruled out. |
| Running now | Context pid 9268 is alive inside detached bash pid 9257 (parent pid 1). Epoch 6 finished: validation macro F1 **0.970989**, which did not beat epoch 5. Epoch 7 was in progress (step 900/3124 at last check). `best.pt` is still the epoch-5 checkpoint. `last.pt` was updated after epoch 6. |
| Resume limits | Epoch-5 `last.pt` restored weights, unfrozen backbone, and two AdamW groups at learning rates 2.960e-4 and 2.960e-5. It did **not** contain scheduler state, the early-stopping counter, or RNG state. Early-stopping baseline was corrected from `best.pt` to 0.97295 and the counter reset to 0. The cosine schedule was restarted with `T_max=25`. This is not an exact uninterrupted continuation. Logged `exact_continuation: false`. |
| Current progress | Epoch 6 is in progress (batch steps logged). It is not a completed epoch. |
| Detail / fused | Not started. Test set unused. No model selected. No export. |
| Selection rule (frozen for later) | Highest validation macro F1. Tie-break: higher Spoiled recall. Do not prefer fused unless it wins that rule. |
| Abandoned | `full_train_20261001T162513Z` died after `train_start` only. `resume_20261009T130807Z` died at epoch 6 step 200. |

## Duplicate review (2026-10-09)

Active manifests were not changed. The recovered search matches the earlier log: **418,572** candidate pairs, **193,695** cross-split, using 8×8 average hash, 3-hex bands, Hamming ≤ 5, and a cap of 64 comparisons per image. That is not an all-pairs search. Files: `data/processed/reports/near_review/all_candidates.jsonl` and `cross_split_candidates.jsonl`. Provenance: `data/processed/reports/near_scan_provenance.json`. `--skip-near-scan` no longer replaces a completed provenance file with a skipped status, and the pre-check no longer writes the audit report.

Of the 57 representative cross-split pairs, decoded pixels were never identical. Reviewed labels: **47 likely related** (named salt-and-pepper or other near-copies of the same photo; RGB mean absolute error under 2), **2 distinct** (Hamming 0 was a false match), **8 unresolved**. Contact sheets: `data/processed/reports/near_review/sheets/`. Hamming-0 cross-split pairs number 24,549 and form large perceptual components (largest 313 images). Those components were **not** merged.

Because same-capture augmentations cross the active split, current context metrics are provisional. Versioned manifests that group exact SHA-256 with augmentation-stripped filename stems are in `data/processed/manifests_v2_augstem/` (fingerprint `c407ca0b…a374b0`; 50,077 / 10,805 / 10,436). Renamed near-copies such as `a_f###.png` are not all grouped by that rule. Do not evaluate the current weights on that holdout.

`artifacts/reviews/hold_downstream_training.json` makes a new detail, fused, test, or export process exit before training. The already-running context process was not restarted.

## Corrected split prepared, not trained (2026-10-09)

`data/processed/manifests/` was not modified (manifest mtime remains 2026-10-01). `manifests_v2_augstem` is only a filename candidate and is not the training split.

`data/processed/manifests_v3_source/` fingerprint `4f2534505b3ca656e72b786046d0d7f7da1088a221fe857a3b16639bd87c8f57`.

| Split | Images | Groups | Fresh images | Spoiled images |
|-------|--------|--------|--------------|----------------|
| train | 37,164 | 24,107 | 24,094 | 13,070 |
| val | 7,853 | 5,165 | 5,118 | 2,735 |
| test | 8,035 | 5,168 | 5,309 | 2,726 |

Benchmark size is 53,052 of 71,318. 18,266 images are listed in `exclusions.csv` and are not in any split. Confirmed same-capture links that remain in the benchmark cross splits: 0. Largest kept group: 31 images. Relationship log: `grouping_evidence.jsonl` (26,044 confirmed, 19,313 distinct, 50,630 unresolved proposals).

`configs/train_v3.yaml` points at this split, writes to `artifacts/checkpoints/v3` and `artifacts/runs/v3`, and sets `resume: null`. Context and fused still request ImageNet EfficientNet-B0. Detail stays a scratch CNN. The trainer refuses to resume old `artifacts/checkpoints/{context,detail,fused}` weights onto this fingerprint.

Provisional context finished on its own by early stopping at epoch 23 (best validation macro F1 0.982042 at epoch 16). Pid 9268 was already gone on 2026-10-11, so it was not signaled. Checkpoints and logs were kept. `scripts/continue_training.sh` now exits immediately, and the downstream hold file remains.

V3 training finished at 2026-10-12T08:10:25Z. Context stopped at epoch 26 (best validation macro F1 0.984695 at epoch 19). Detail ran 30 epochs (best 0.932566 at epoch 26). Fused ran 30 epochs (best 0.985118 at epoch 29). Log: `artifacts/runs/v3/orchestrator/train_20261011T135004Z.log`.

Selection record `artifacts/metrics/v3/eval/selection.json` was written before test scoring. The frozen rule picks fused epoch 29, checkpoint `artifacts/checkpoints/v3/fused/best.pt`, SHA-256 `b962c650ed807f362885e1f6c240d8abc693f824980b3afbfe8f597cafd60200`.

One test pass of that checkpoint, argmax only, on all 8,035 v3 test images: accuracy 0.983323, balanced accuracy 0.979706, macro F1 0.981337, Spoiled recall 0.968452, 86 Spoiled images predicted Fresh (rate 0.031548). Report: `artifacts/metrics/v3/eval/fused/test/REPORT.md`.

## Not done yet (honest)

- Do not train or test the original fingerprint while `artifacts/reviews/hold_downstream_training.json` exists. That hold file is still present.
- Unresolved structural pairs and conflicting-label groups stay out of the v3 benchmark. The candidate search is capped at 64 comparisons per image in a hash band. That is not a claim that every same-capture pair was found.
- The selected fused checkpoint is exported at `artifacts/deployment/v3/fused_epoch29_b962c650ed80` (model version `freshnet-v3-fused-epoch29-b962c650ed80`). The local API loads that bundle at startup and serves `POST /api/predict`. The website is the Vite app in `frontend/`.

## Produce identity, combined labels and guidance (2026-10-16)

Added on top of the unchanged freshness model. Details: `docs/PRODUCE_IDENTIFICATION_AND_GUIDANCE.md`.

- Produce identity is **selected by the user**. No produce classifier exists. There is no unknown / non-produce detector.
- Results read "Predicted: Fresh banana" or "Predicted: Spoiled apple", and "Predicted: Fresh produce" when identity is unknown.
- Spoilage severity is always "Not assessed". A proposal for genuine severity prediction is in `docs/SEVERITY_PROPOSAL.md`. Nothing from it is implemented.
- Guidance comes from `backend/app/data/guidance_catalogue.json` (version 2026-10-16.1). Reviewed storage guidance covers 11 produce types. Bitter gourd is awaiting review. Unknown / other gets general advice.
- `POST /api/predict` keeps its legacy fields and adds `produce`, `freshness`, `severity`, `guidance`, `combined_label` and `contract_version`. New endpoints: `GET /api/produce`, `GET /api/guidance`.
- Produce label audit: `docs/PRODUCE_LABEL_INVENTORY.md`, produced by `ml/audit_produce_labels.py` (metadata only).
- Tests: `tests/backend/test_produce_guidance.py`. No retraining, rescoring or threshold change took place, and the original-split hold file is untouched.

## Scripts / Colab

- `scripts/continue_training.sh` resumes `last.pt` per architecture, skips `complete.json`, and stops on failure
- `scripts/start_detached_training.py` starts that script in a new session (parent pid 1) and refuses a second start while `artifacts/runs/orchestrator/train.pid` is alive
- `scripts/run_full_training.sh` (older orchestrator; does not resume)
- `scripts/run_full_training.ps1` (Windows)
- `notebooks/train_freshnet_colab_gpu.ipynb` (GPU cloud fallback; not needed while MPS job runs)

## Honesty

- No fabricated metrics or “fused wins” claims
- Smoke metrics are not final
- Do not report training complete without checking logs + checkpoint metadata
