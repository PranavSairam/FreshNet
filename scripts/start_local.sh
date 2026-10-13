#!/usr/bin/env bash
# Start the local API and website if those ports are free.
# Does not stop processes that are already listening.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/artifacts/runs/local"

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "API already listening on http://127.0.0.1:8000"
else
  PYTHONPATH="$ROOT/backend:$ROOT/ml" \
    nohup "$ROOT/backend/.venv/bin/python" -m uvicorn app.main:app \
    --host 127.0.0.1 --port 8000 \
    >"$ROOT/artifacts/runs/local/api.log" 2>&1 &
  echo $! >"$ROOT/artifacts/runs/local/api.pid"
  disown $! || true
  echo "API starting at http://127.0.0.1:8000"
fi

if lsof -nP -iTCP:5173 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Website already listening on http://127.0.0.1:5173"
else
  (cd "$ROOT/frontend" && nohup npm run dev -- --host 127.0.0.1 --port 5173) \
    >"$ROOT/artifacts/runs/local/web.log" 2>&1 &
  echo $! >"$ROOT/artifacts/runs/local/web.pid"
  disown $! || true
  echo "Website starting at http://127.0.0.1:5173"
fi
