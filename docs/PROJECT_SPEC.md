# FreshNet Project Specification

## Title

FreshNet: Deep Learning-Based Food Freshness Classification Using Dual-Path Neural Networks

## Active task (confirmed)

**Binary visual freshness classification**

| Label | Index |
| --- | --- |
| Fresh | 0 |
| Spoiled | 1 |

`Rotten` raw folders map to **Spoiled**. **Medium is absent** from the local dataset and is a **future extension only**. It must not appear in active training, inference, API result schemas, or frontend result chips.

Outputs remain **visual freshness estimates**. An image cannot establish whether food is safe to eat. Do not claim shelf life or convert probabilities into a freshness score.

## Dataset (audited local copy)

- Path: `data/Dataset`
- Images: **71,318** readable
- Fresh: 47,341 · Rotten→Spoiled: 23,977
- Manifests under `data/processed/manifests/`
- Exact duplicates and heuristic groups must not cross splits
- Near-duplicate perceptual matches are **review candidates only**, not automatic specimen identity
- No `specimen_id` / `capture_session` metadata → residual leakage risk remains documented

## Models

1. **Context-only** — EfficientNet-B0 on 224×224 (`weights=None` offline by default)
2. **Detail-only** — five 160×160 patches (corners+center) from 384×384 with a shared lightweight CNN
3. **Fused FreshNet** — projected context + detail embeddings, LayerNorm, dropout, **2-logit** head, CrossEntropyLoss

Shared source-image augmentation is applied once before branch inputs. Val/test transforms are deterministic. Preprocessing is stored in each checkpoint.

## Training / evaluation rules

- Use prepared manifests only; never tune on the test set
- Early stop / checkpoint selection on validation **macro F1**
- Class weights from the **training split only**
- Compare architectures with identical manifests and documented budgets
- Do not claim fused is better unless measured results support it
- If the near-duplicate scan is incomplete, experiments are **provisional** — do not claim leakage-free final evaluation

## Repository layout

| Path | Role |
| --- | --- |
| `frontend/` | React UI (binary task messaging) |
| `backend/` | FastAPI + health/model status |
| `ml/models/` | Context / detail / fused networks |
| `ml/training/` | Dataset, train, metrics, evaluate |
| `ml/dataset/` | Preparation + integrity |
| `configs/train.yaml` | Training configuration |
| `configs/dataset.yaml` | Dataset preparation configuration |
| `data/processed/` | Manifests, audits, hash cache |
| `artifacts/` | Checkpoints / metrics / exports |

## Completion criteria (training stage)

1. Integrity audit reports image + unique-group counts; exact/heuristic groups do not cross splits
2. Near-duplicate policy documented; full scan complete or audit marked incomplete/provisional
3. Binary class mapping `Fresh=0`, `Spoiled=1` saved and validated
4. All three architectures train from the same manifests
5. Evaluation writes JSON + confusion figures with spoiled→fresh rates
6. Best validated (non-smoke) checkpoint can be exported for the backend
