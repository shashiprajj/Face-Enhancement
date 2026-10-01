"""Metrics of the Phase 3 harness (P3-T1 §7). All computed in the 224×224 E-frame against the clean reference.

Image quality (P3-T5)   PSNR (RGB, 8-bit), SSIM (luma BT.601, Gaussian σ=1.5, Wang et al. constants),
                        LPIPS (AlexNet, lpips 0.1.4 'v0.1'); descriptive: sharpness ratio, mean colour shift.
Geometry (P3-T5/T7)     YuNet on the output: detected? + 5-landmark NME vs the reference landmarks
                        (normalised by the reference inter-ocular distance).
Identity (P3-T6)        SFace (E01, test-only) cosine similarity. Faces are aligned with the *reference*
                        landmarks for every image (fixed alignment: identity content is measured, not detector luck).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from fes_data.common import repo_path
from fes_data.preprocess import YuNetDetector

SFACE = "models/E01_sface/face_recognition_sface_2021dec.onnx"
YUNET = "models/D01_yunet/face_detection_yunet_2026may.onnx"


def psnr_rgb(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d2 = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = float(d2[mask].mean() if mask is not None else d2.mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def luma(img_bgr: np.ndarray) -> np.ndarray:
    b, g, r = [img_bgr[..., i].astype(np.float64) for i in range(3)]
    return 0.299 * r + 0.587 * g + 0.114 * b


def ssim_y(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    _, smap = structural_similarity(luma(a), luma(b), data_range=255.0, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    return float(smap[mask].mean() if mask is not None else smap.mean())


def sharpness_ratio(out: np.ndarray, ref: np.ndarray) -> float:
    """Laplacian variance of output / reference (≈1 matches reference detail; ≪1 over-smoothed; ≫1 over-sharpened)."""
    lv = lambda x: float(cv2.Laplacian(luma(x), cv2.CV_64F).var())  # noqa: E731
    return lv(out) / max(lv(ref), 1e-6)


def color_shift(out: np.ndarray, ref: np.ndarray) -> float:
    """Mean CIELAB ΔE76 between the image-mean colours (global tint/exposure shift, not texture)."""
    to_lab = lambda x: cv2.cvtColor(x.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)  # noqa
    return float(np.linalg.norm(to_lab(out) - to_lab(ref)))


class LPIPS:
    """lpips 0.1.4, net='alex', version '0.1'. AlexNet backbone weights come from torchvision (TORCH_HOME)."""

    def __init__(self):
        import lpips
        import torch
        self.torch = torch
        self.model = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()

    def __call__(self, outs: list[np.ndarray], ref: np.ndarray, mask: np.ndarray | None = None) -> list[float]:
        t = self.torch

        def to_t(x):
            rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return t.from_numpy(rgb.transpose(2, 0, 1).copy())

        with t.inference_mode():
            a = t.stack([to_t(o) for o in outs])
            b = to_t(ref)[None].expand_as(a)
            if mask is None or mask.all():
                return [float(v) for v in self.model(a, b).flatten()]
            m = t.from_numpy(mask.astype(np.float32))
            self.model.spatial = True
            try:
                d = self.model(a, b)[:, 0]              # N×H×W, upsampled to input size
            finally:
                self.model.spatial = False
            return [float((di * m).sum() / m.sum()) for di in d]


def face_row(landmarks: np.ndarray, score: float = 1.0) -> np.ndarray:
    """YuNet-format row (bbox, 5 landmarks, score) built from given landmarks, for FaceRecognizerSF.alignCrop."""
    lm = np.asarray(landmarks, np.float32).reshape(5, 2)
    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    return np.array([x0, y0, x1 - x0, y1 - y0, *lm.flatten(), score], np.float32)


class SFace:
    """E01 SFace (OpenCV Zoo, Apache-2.0). TEST-ONLY evaluator (P0 master C1). 128-d, cosine similarity."""

    def __init__(self):
        self.path = SFACE
        self.rec = cv2.FaceRecognizerSF.create(str(repo_path(SFACE)), "")

    def embed(self, img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        crop = self.rec.alignCrop(img_bgr, face_row(landmarks))
        f = self.rec.feature(crop).flatten().astype(np.float64)
        return f / max(np.linalg.norm(f), 1e-12)


class Geometry:
    """YuNet detection on an E-frame image; NME of the 5 landmarks against the reference's landmarks."""

    def __init__(self):
        self.det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)

    def __call__(self, img: np.ndarray, ref_lm: np.ndarray) -> tuple[bool, float | None, float | None]:
        faces = self.det.detect(img)
        if not faces:
            return False, None, None
        f = faces[0]
        lm = np.asarray(f.landmarks)
        iod = float(np.linalg.norm(ref_lm[1] - ref_lm[0]))
        return True, round(f.score, 4), float(np.linalg.norm(lm - ref_lm, axis=1).mean() / iod)


ALEXNET = "models/X01_torchvision/hub/checkpoints/alexnet-owt-7be5be79.pth"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def set_torch_home() -> None:
    """LPIPS AlexNet backbone (asset A15, P3-T2): staged from the git-ignored models/ tree to a local TORCH_HOME,
    SHA-256-checked, so torch.load never does large reads through the bind mount (P3-T2 F5)."""
    import shutil
    import tempfile
    from pathlib import Path

    from fes_bench.archs import staged_copy
    home = Path(tempfile.gettempdir()) / "fes_torch_home"
    dst = home / "hub" / "checkpoints" / Path(ALEXNET).name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_copy(repo_path(ALEXNET), ALEXNET_SHA256)), dst)
    os.environ["TORCH_HOME"] = str(home)
