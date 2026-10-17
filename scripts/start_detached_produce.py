#!/usr/bin/env python3
"""Start (or resume) produce-classifier training in a new session.

Usage:
    backend/.venv/bin/python scripts/start_detached_produce.py

Starts `ml/produce/train.py --resume auto` detached from this terminal. Refuses to start a
second trainer: it checks the pidfile, any running `ml/produce/train.py`, and the trainer
also holds its own file lock. Re-running after an interruption continues from `last.pt`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts" / "produce" / "v1" / "orchestrator"
PIDFILE = STATE / "train.pid"
LOG = STATE / "detached_stdout.log"
PYTHON = ROOT / "backend" / ".venv" / "bin" / "python"
TRAINER = ROOT / "ml" / "produce" / "train.py"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _trainer_running() -> bool:
    result = subprocess.run(["pgrep", "-f", "ml/produce/train.py"], capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    if PIDFILE.is_file():
        try:
            existing = int(PIDFILE.read_text().strip())
        except ValueError:
            existing = 0
        if existing and _alive(existing):
            print(f"already_running pid={existing} log={LOG}")
            return 0
        PIDFILE.unlink(missing_ok=True)
    if _trainer_running():
        print("Refusing to start: ml/produce/train.py is already running.", file=sys.stderr)
        return 1
    if (ROOT / "artifacts" / "produce" / "v1" / "checkpoints" / "complete.json").is_file():
        print("Training already complete (complete.json present). Nothing to start.")
        return 0

    if os.fork() > 0:
        for _ in range(100):
            time.sleep(0.1)
            if PIDFILE.is_file():
                try:
                    pid = int(PIDFILE.read_text().strip())
                except ValueError:
                    continue
                if _alive(pid):
                    print(f"started pid={pid} log={LOG}")
                    return 0
        print("detached process did not record a live pid", file=sys.stderr)
        return 1

    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    PIDFILE.write_text(str(os.getpid()))
    log_handle = LOG.open("a", encoding="utf-8")
    os.dup2(log_handle.fileno(), 1)
    os.dup2(log_handle.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(ROOT)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    os.execve(str(PYTHON), [str(PYTHON), str(TRAINER), "--resume", "auto"], env)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
