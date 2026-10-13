#!/usr/bin/env bash
# Fresh context -> detail -> fused training on manifests_v3_source.
# Refuses to start while any train_model.py process is still running.
# Does not resume provisional checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if pgrep -f 'ml/train_model.py' >/dev/null 2>&1; then
  echo "Refusing to start: ml/train_model.py is still running. Let the provisional context job exit first."
  exit 1
fi
PYTHON="${ROOT}/backend/.venv/bin/python"
export PYTHONPATH="${ROOT}/ml"
DEVICE="${FRESHNET_DEVICE:-mps}"
LOG_DIR="${ROOT}/artifacts/runs/v3/orchestrator"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "pid $$ device=$DEVICE"
echo "config=configs/train_v3.yaml"
echo "initialization context/fused=imagenet_efficientnet_b0 detail=scratch no_resume"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
fp = json.loads(Path("data/processed/manifests_v3_source/split_fingerprint.json").read_text())["split_fingerprint"]
expected = "4f2534505b3ca656e72b786046d0d7f7da1088a221fe857a3b16639bd87c8f57"
if fp != expected:
    raise SystemExit(f"fingerprint mismatch: {fp}")
print("fingerprint_ok", fp)
PY

for arch in context detail fused; do
  echo "==== TRAIN $arch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  "$PYTHON" ml/train_model.py --architecture "$arch" --config configs/train_v3.yaml --device "$DEVICE"
  echo "==== DONE $arch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
done
echo "all_complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
