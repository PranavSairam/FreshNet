#!/usr/bin/env python3
"""Export the selected FreshNet checkpoint.

The v3 deployment path writes a versioned weights-only bundle.
The provisional original-split fingerprint stays blocked by the hold file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML = Path(__file__).resolve().parent
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.export_bundle import main


if __name__ == "__main__":
    raise SystemExit(main())
