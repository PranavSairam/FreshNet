#!/usr/bin/env python3
"""Start scripts/continue_training.sh in a new session so it outlives this process."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIDFILE = ROOT / "artifacts" / "runs" / "orchestrator" / "train.pid"
LOG = ROOT / "artifacts" / "runs" / "orchestrator" / "detached_stdout.log"
SCRIPT = ROOT / "scripts" / "continue_training.sh"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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

    if os.fork() > 0:
        # Parent returns immediately; the grandchild writes the pid file.
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
