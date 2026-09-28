"""Shared fixtures. Face images come from the (git-ignored) P2-T3 references; tests needing them are skipped
when the dataset has not been prepared. No face images are committed to the repository."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fes_data import common as C
from fes_data import preprocess as P

YUNET = C.repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx")
REF_MANIFEST = C.repo_path("data/manifests/references_v1.csv.gz")


@pytest.fixture(scope="session")
def detector() -> P.YuNetDetector:
    if not YUNET.exists():
        pytest.skip("YuNet model not acquired")
    return P.YuNetDetector(YUNET, score_threshold=0.7, nms_threshold=0.3, max_side=640)


@pytest.fixture(scope="session")
def ref_faces() -> list[np.ndarray]:
    """Three accepted clean references (deterministic choice: first three test-pool ref_ids)."""
    if not REF_MANIFEST.exists():
        pytest.skip("references not built")
    rows = sorted((r for r in C.read_csv_gz(REF_MANIFEST) if r["accepted"] == "True" and r["pool"] == "test_pool"),
                  key=lambda r: r["ref_id"])[:3]
    imgs = []
    for r in rows:
        p = C.repo_path(r["ref_path"])
        if not p.exists():
            pytest.skip("reference files missing")
        imgs.append(cv2.imread(str(p), cv2.IMREAD_COLOR))
    return imgs


@pytest.fixture
def pcfg() -> P.PreprocessConfig:
    return P.PreprocessConfig(out_size=256)
