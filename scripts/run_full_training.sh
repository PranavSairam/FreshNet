#!/usr/bin/env bash
# FreshNet full local training orchestrator (macOS/Linux).
# Runs context -> detail -> fused sequentially on the same frozen manifests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/backend/.venv/bin/python"
export PYTHONPATH="${ROOT}/ml"
DEVICE="${FRESHNET_DEVICE:-auto}"
LOG_DIR="${ROOT}/artifacts/runs/orchestrator"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/full_train_$(date -u +%Y%m%dT%H%M%SZ).log"

{
  echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "device=$DEVICE"
  echo "python=$($PYTHON -c 'import sys; print(sys.version)')"
  $PYTHON - <<'PY'
import json
from pathlib import Path
fp=json.loads(Path('data/processed/manifests/split_fingerprint.json').read_text())['split_fingerprint']
assert fp=='005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9', fp
print('fingerprint_ok', fp)
PY

  for arch in context detail fused; do
    echo "==== TRAIN $arch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
    "$PYTHON" ml/train_model.py --architecture "$arch" --config configs/train.yaml --device "$DEVICE"
    echo "==== DONE $arch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  done
  echo "all_complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$MASTER_LOG"
