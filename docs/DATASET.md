# FreshNet dataset preparation

## Detected local dataset

An on-disk dataset is present at:

```text
data/Dataset/
  Fresh/<food-subtype>/...
  Rotten/<food-subtype>/...
```

Inventory from this machine (Stage 2 check):

| Top-level folder | Mapped label (`configs/dataset.yaml`) | Images |
| --- | --- | --- |
| Fresh | Fresh | 47,341 |
| Rotten | Spoiled | 23,977 |
| **Total** | | **71,318** |

- **Medium is absent.** Three-class Fresh/Medium/Spoiled training is **not** allowed until you supply a properly annotated Medium class. FreshNet will not invent it.
- Two-class training (Fresh vs Spoiled) is allowed via `labels.allow_two_class_training: true`.
- No official `train` / `val` / `test` folders were found, so preparation creates reproducible **70/15/15** group-aware splits.
- No license file was bundled with this copy; record provenance yourself. This project does **not** invent a download URL.

If your dataset is elsewhere, set `dataset.source_path` in `configs/dataset.yaml` or pass `--source`.

## Place data (if missing)

Class folders:

```text
data/Dataset/
  Fresh/...
  Rotten/...      # mapped to Spoiled
  Medium/...      # only if genuinely annotated
```

Or a CSV manifest with columns:

- required: `image_path`, `label`
- optional: `food_type`, `specimen_id`, `capture_session`, `source`, `split`, `fixture`

## PowerShell commands

From the repo root (`FreshNet`):

```powershell
# 1) Python 3.10+ venv (reuse backend venv or create one)
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r ..\ml\requirements-data.txt
cd ..

# 2) Build the tiny synthetic fixture (tests only; not for real training)
python ml\dataset\build_fixture.py

# 3) Prepare the real local dataset (writes manifests + audit report)
$env:PYTHONPATH = "$pwd\ml"
python ml\prepare_dataset.py --config configs\dataset.yaml

# Optional: skip near-duplicate scan on very large trees (exact duplicates still run)
python ml\prepare_dataset.py --config configs\dataset.yaml --skip-near-duplicates

# Optional: fail if Medium is missing (documents three-class gate)
python ml\prepare_dataset.py --config configs\dataset.yaml --require-three-class
# Expected exit code: 3 on this dataset

# CSV mode
python ml\prepare_dataset.py --input-mode csv_manifest --csv path\to\manifest.csv

# 4) Run pipeline tests
$env:PYTHONPATH = "$pwd\ml;$pwd\backend"
python -m pytest tests\dataset -q
```

macOS / Linux equivalents use `source backend/.venv/bin/activate` and `export PYTHONPATH="$PWD/ml"`.

## Outputs

| Path | Purpose |
| --- | --- |
| `data/processed/manifests/train.csv` | Training rows |
| `data/processed/manifests/val.csv` | Validation rows |
| `data/processed/manifests/test.csv` | Test rows |
| `data/processed/manifests/all.csv` | All usable rows |
| `data/processed/manifests/problems.csv` | Flagged issues |
| `data/processed/manifests/split_fingerprint.json` | Experiment fingerprint |
| `data/processed/reports/dataset_audit.json` | Machine-readable audit |
| `data/processed/reports/dataset_audit.md` | Human-readable audit |
| `data/processed/transforms.json` | Train/eval transform specs |

## Leakage controls

- Groups prefer `specimen_id` / `capture_session` from CSV when present.
- Otherwise groups use food subtype + augmentation-stripped filenames, then merge exact/near-duplicate clusters.
- Images from the same group never cross splits.
- Official splits are preserved only when present and leakage-free; otherwise preparation falls back to group-aware 70/15/15 and records the reason.
- If reliable specimen metadata is absent, the audit report includes an explicit **residual leakage risk** warning.

## Synthetic fixture

`data/fixtures/synthetic_pipeline/` is **test data only**. It is excluded from real training/evaluation via config `exclude_path_parts` and fixture metadata.
