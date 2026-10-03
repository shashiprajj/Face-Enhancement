"""P4-T3 — second, independent TEST-ONLY identity evaluator: E07 dlib ResNet-29 face descriptor (P0 master C1).

Never part of the FES runtime and never exposed by the service. Embeddings are held in memory only.

Model     dlib_face_recognition_resnet_model_v1.dat (dlib-models @ fd81b630; README: released into the public domain)
          sha256 55533b28…(.dat) / abb1f610…(.bz2 as downloaded). Runtime: dlib 20.0.1 (BSL-1.0), official PyPI sdist
          compiled in fes-dev:cpu-p4 (docker/Dockerfile.cpu-p4, hash-pinned requirements-p4.txt).
Input     150×150 RGB face chip with dlib `get_face_chip` geometry (padding 0.25).
Routes    * chip route (E-frame images): FairFace chips ARE dlib get_face_chip outputs (padding 0.25, 224 px), so the
            dlib chip of any E-frame image is an INTER_AREA resize 224 → 150. Identical for every image of a pair
            (the fixed-alignment principle of P3-T1 §7) and uses no landmarks.
          * template route (any image): similarity fit of 5 landmarks (YuNet convention) to DLIB_CHIP_TEMPLATE_150,
            warp INTER_AREA-prereduced/INTER_LINEAR (fes_bench.geometry.warp). For real-world crops.
Output    128-d descriptor, L2-normalised → cosine similarity (dlib's native rule is Euclidean < 0.6 on the raw
          descriptor; raw norms are ≈1, recorded in validation).
"""

from __future__ import annotations

import cv2
import numpy as np

from fes_bench import geometry as G
from fes_data.common import repo_path, sha256_file
from fes_data.preprocess import umeyama_similarity

DLIB_MODEL = "models/E07_dlib_resnet/dlib_face_recognition_resnet_model_v1.dat"
DLIB_MODEL_SHA256 = "55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30"
DLIB_VERSION = "20.0.1"
CHIP = 150
# Mean YuNet 5-point landmarks of the 112 dev_pilot references (dlib chips, padding 0.25) scaled 224 → 150.
# Derived on DEV data only (P4-T3 §3); frozen here.
DLIB_CHIP_TEMPLATE_150 = np.array([[46.396, 45.712], [100.894, 45.828], [72.459, 77.261],
                                   [50.880, 102.266], [95.750, 102.370]], np.float64)


class DlibResNet:
    """E07 — TEST-ONLY evaluator. `embed_chip` for E-frame images, `embed` for landmark-aligned use."""

    id = "E07-dlib-resnet-v1"

    def __init__(self, verify_sha: bool = True):
        import dlib
        assert dlib.__version__ == DLIB_VERSION, f"dlib {dlib.__version__} != pinned {DLIB_VERSION}"
        path = repo_path(DLIB_MODEL)
        if verify_sha:
            got = sha256_file(path)
            assert got == DLIB_MODEL_SHA256, f"E07 weights SHA-256 mismatch {got}"
        self.path, self.sha256 = DLIB_MODEL, DLIB_MODEL_SHA256
        self.net = dlib.face_recognition_model_v1(str(path))

    def _desc(self, chip_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        assert chip_bgr.shape == (CHIP, CHIP, 3) and chip_bgr.dtype == np.uint8
        rgb = np.ascontiguousarray(cv2.cvtColor(chip_bgr, cv2.COLOR_BGR2RGB))
        d = np.asarray(self.net.compute_face_descriptor(rgb), np.float64)
        n = float(np.linalg.norm(d))
        return d / max(n, 1e-12), n

    def chip_from_e(self, img_e: np.ndarray) -> np.ndarray:
        assert img_e.shape[:2] == (G.E_SIZE, G.E_SIZE)
        return cv2.resize(img_e, (CHIP, CHIP), interpolation=cv2.INTER_AREA)

    def chip_from_landmarks(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        M = umeyama_similarity(np.asarray(landmarks, np.float64).reshape(5, 2), DLIB_CHIP_TEMPLATE_150)
        return G.warp(img, M, (CHIP, CHIP), up_interp=cv2.INTER_CUBIC, border_value=(0, 0, 0))

    def embed_chip(self, img_e: np.ndarray) -> np.ndarray:
        return self._desc(self.chip_from_e(img_e))[0]

    def embed(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        return self._desc(self.chip_from_landmarks(img, landmarks))[0]

    def raw_norm(self, img_e: np.ndarray) -> float:
        return self._desc(self.chip_from_e(img_e))[1]
