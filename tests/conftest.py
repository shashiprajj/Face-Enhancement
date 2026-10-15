"""Shared fixtures. Face images come from tests/fixtures/ (git-ignored: FairFace and Indian-expression benchmark
images); tests needing them are skipped when they are absent. No face images are committed to the repository."""

from __future__ import annotations

import numpy as np
import pytest

from fes_service import preprocess as P
from fes_service import repo_path
from helpers import fixture_img

YUNET = repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx")


@pytest.fixture(scope="session")
def detector() -> P.YuNetDetector:
    if not YUNET.exists():
        pytest.skip("YuNet model not acquired")
    return P.YuNetDetector(YUNET, score_threshold=0.7, nms_threshold=0.3, max_side=640)


@pytest.fixture(scope="session")
def ref_faces() -> list[np.ndarray]:
    """Three accepted clean FairFace references (val pool)."""
    return [fixture_img(f"ref_{i}.png") for i in range(3)]


@pytest.fixture(scope="session")
def dev() -> dict:
    """A strongly blurred DEV face chip (gaussblur-extreme) and a full image whose face chip extends past a corner."""
    blur = fixture_img("blur_extreme.png")
    canvas = np.full((300, 300, 3), 127, np.uint8)
    canvas[:194, :194] = blur[30:, 30:]
    return {"blur": blur, "edge": canvas}


@pytest.fixture
def pcfg() -> P.PreprocessConfig:
    return P.PreprocessConfig(out_size=256)
