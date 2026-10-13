#!/usr/bin/env bash
# Sequential FreshNet training. Stops on the first failure.
# Skips an architecture that already wrote checkpoints/<arch>/complete.json.
# Resumes from checkpoints/<arch>/last.pt when that file exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/backend/.venv/bin/python"
export PYTHONPATH="${ROOT}/ml"
DEVICE="${FRESHNET_DEVICE:-mps}"
LOG_DIR="${ROOT}/artifacts/runs/orchestrator"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/continue_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "retired $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Original-split training is closed. Checkpoints and logs were kept. The downstream hold remains."
echo "Use: backend/.venv/bin/python scripts/start_detached_training_v3.py"
exit 0

echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "pid $$"
echo "device=$DEVICE"
echo "master_log $MASTER_LOG"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
fp = json.loads(Path("data/processed/manifests/split_fingerprint.json").read_text())["split_fingerprint"]
expected = "005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9"
if fp != expected:
    raise SystemExit(f"fingerprint mismatch: {fp}")
print("fingerprint_ok", fp)
PY

train_arch() {
  local arch="$1"
  local complete="artifacts/checkpoints/${arch}/complete.json"
  local last="artifacts/checkpoints/${arch}/last.pt"
  if [[ -f "$complete" ]]; then
    echo "==== SKIP $arch already complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
    return 0
  fi
  echo "==== TRAIN $arch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  if [[ -f "$last" ]]; then
    "$PYTHON" ml/train_model.py \
      --architecture "$arch" \
      --config configs/train.yaml \
      --device "$DEVICE" \
      --resume "$last"
  else
    "$PYTHON" ml/train_model.py \
      --architecture "$arch" \
      --config configs/train.yaml \
      --device "$DEVICE"
  fi
  echo "==== DONE $arch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

train_arch context
if [[ -f artifacts/reviews/hold_downstream_training.json ]]; then
  echo "==== HOLD $(date -u +%Y-%m-%dT%H:%M:%SZ) detail, fused, test evaluation, and export stay paused ===="
  echo "Context checkpoints were left in place. Remove the hold file only after duplicate review is resolved."
  exit 0
fi
train_arch detail
train_arch fused
echo "all_complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
