"""Legacy import path.

Prefer `from models import FusedFreshNet, build_model`.
The active task is binary Fresh/Spoiled classification.
"""

from models import DualPathFreshNet, FusedFreshNet

__all__ = ["DualPathFreshNet", "FusedFreshNet"]
