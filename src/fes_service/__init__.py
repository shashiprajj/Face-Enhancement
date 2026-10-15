"""Face Enhancement Service: guarded NAFNet face deblurring on CPU (ONNX Runtime), research candidate."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(rel: str | os.PathLike) -> Path:
    """Resolve a repo-relative path (configs and models are referenced by repo-relative POSIX paths)."""
    return REPO_ROOT / Path(rel)
