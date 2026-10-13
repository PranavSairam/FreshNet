#!/usr/bin/env python3
"""Start v3 training in a new session. Refuses while any trainer is already running."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIDFILE = ROOT / "artifacts" / "runs" / "v3" / "orchestrator" / "train.pid"
LOG = ROOT / "artifacts" / "runs" / "v3" / "orchestrator" / "detached_stdout.log"
SCRIPT = ROOT / "scripts" / "continue_training_v3.sh"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _trainer_running() -> bool:
    result = subprocess.run(["pgrep", "-f", "ml/train_model.py"], capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def main() -> int:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    if PIDFILE.is_file():
        try:
            existing = int(PIDFILE.read_text().strip())
        except ValueError:
            existing = 0
        if existing and _alive(existing):
            print(f"already_running pid={existing}")
            return 0
    if _trainer_running():
        print("Refusing to start: ml/train_model.py is still running. V3 training has not started.")
        return 1

    if os.fork() > 0:
        for _ in range(50):
            time.sleep(0.1)
            if PIDFILE.is_file():
                pid = int(PIDFILE.read_text().strip())
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
    os.chdir(ROOT)
    os.execv("/bin/bash", ["bash", str(SCRIPT)])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
