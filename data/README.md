# Data

Local datasets and preparation outputs. **Do not commit real images, labels, or databases.**

## Detected layout in this project

```text
data/Dataset/
  Fresh/<subtype>/...
  Rotten/<subtype>/...
```

Verified inventory: **71,318** images — Fresh 47,341 + Rotten 23,977.  
**Medium is not present.** Map `Rotten` → `Spoiled` via `configs/dataset.yaml`. Do not invent Medium.

## Prepare manifests

See [docs/DATASET.md](../docs/DATASET.md) for PowerShell commands.

Outputs land in `data/processed/` (gitignored).

## Synthetic fixture

`data/fixtures/synthetic_pipeline/` is **test-only** data for the preparation pipeline. Never train or evaluate the real model on it.

## Rules

- Do not invent a `Medium` class for a two-class dataset.
- Do not commit copyrighted/private images.
- Uploads: `data/uploads/` (gitignored). SQLite: `data/local/` (gitignored).
