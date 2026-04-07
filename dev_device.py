"""Default Ultralytics device for local dev on Apple Silicon (M1/M2/M3)."""

from __future__ import annotations

import platform
from typing import Optional


def default_mps_device() -> Optional[str]:
    """Return 'mps' when running on macOS arm64 and MPS is available; else None (Ultralytics auto)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else None
    except Exception:
        return None
