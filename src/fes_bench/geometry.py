"""Shared geometry contract of the Phase 3 harness (P3-T1 §4). Applied identically to every candidate.

Frames
  * E-frame (evaluation frame): the 224×224 clean reference chip. Every candidate output is delivered in it.
  * D-frame: the degraded input, s×s with s ∈ {14, 28, 56, 112, 224}. It is the E-frame scaled by s/224
    (INTER_AREA with OpenCV's half-pixel convention), so x_E = (x_D + 0.5)·(224/s) − 0.5.
  * F-frame: FFHQ-aligned 512×512 (facexlib template), reached from the E-frame by the per-reference
    similarity `ffhq512_transform_2x3` recorded in P2-T3.

Resampling rules
  * Upsampling (scale > 1): INTER_CUBIC for face-model inputs (P1-T5 §5.1 "bicubic to 512"); INTER_LANCZOS4 when
    resizing general-SR outputs up to the E-frame (P1-T5 §5.1, Real-ESRGAN `outscale`).
  * Downsampling (scale < 1): INTER_AREA pre-reduction, then the residual transform with INTER_LINEAR
    (same anti-aliased rule as `fes_data.preprocess.align_face`).
  * Outside-source pixels in the F-frame: BORDER_CONSTANT with facexlib's fill (135, 133, 132) BGR — the value
    both face-specific candidates' native pipelines (FaceRestoreHelper.align_warp_face) use.
"""

from __future__ import annotations

import cv2
import numpy as np

E_SIZE = 224
F_SIZE = 512
FACEXLIB_BORDER_BGR = (135, 133, 132)


def _h(M: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(M, np.float64), [0.0, 0.0, 1.0]])


def d_to_e(s: int) -> np.ndarray:
    """2x3 affine D-frame (s×s) → E-frame (224×224), half-pixel-centre convention."""
    k = E_SIZE / float(s)
    return np.array([[k, 0.0, 0.5 * k - 0.5], [0.0, k, 0.5 * k - 0.5]], np.float64)


def scale_of(M: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(np.asarray(M, np.float64)[:, :2]))))


def warp(img: np.ndarray, M: np.ndarray, out_wh: tuple[int, int], up_interp: int = cv2.INTER_CUBIC,
         border_value=FACEXLIB_BORDER_BGR) -> np.ndarray:
    """Warp `img` with 2x3 `M` (src→dst). Anti-aliased when shrinking; `up_interp` when enlarging."""
    s = scale_of(M)
    src, M_eff = img, np.asarray(M, np.float64)
    if s < 1.0:
        h, w = img.shape[:2]
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        src = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        pre = np.array([[nw / w, 0, 0.5 * nw / w - 0.5], [0, nh / h, 0.5 * nh / h - 0.5], [0, 0, 1]], np.float64)
        M_eff = (_h(M) @ np.linalg.inv(pre))[:2]
        interp = cv2.INTER_LINEAR
    else:
        interp = up_interp
    out = cv2.warpAffine(src.astype(np.float32), M_eff, out_wh, flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def resize_to_e(img: np.ndarray) -> np.ndarray:
    """Bring a square image that covers the E-frame to 224×224 (area down, Lanczos up; identity at 224)."""
    n = img.shape[0]
    if n == E_SIZE:
        return img
    interp = cv2.INTER_AREA if n > E_SIZE else cv2.INTER_LANCZOS4
    return cv2.resize(img, (E_SIZE, E_SIZE), interpolation=interp)


def upsample_to_e(img: np.ndarray, interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """B0 measurement rule (as in P2-T4 input statistics): LR input → 224 with bicubic."""
    if img.shape[0] == E_SIZE:
        return img
    return cv2.resize(img, (E_SIZE, E_SIZE), interpolation=interp)


def d_to_f(s: int, M_e2f: np.ndarray) -> np.ndarray:
    """2x3 affine D-frame → F-frame."""
    return (_h(M_e2f) @ _h(d_to_e(s)))[:2]


def f_to_e(M_e2f: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_h(M_e2f))[:2]


def coverage_mask(M_e2f: np.ndarray, erode_px: int = 0) -> np.ndarray:
    """bool 224×224: E-frame pixels whose F-frame preimage lies inside the 512 canvas.

    With `erode_px` > 0 it is the *evaluation mask* (P3-T1 §4.4): identical for every candidate of a reference,
    so pixels a face model never saw are excluded from all metrics for all candidates.
    """
    ones = np.full((F_SIZE, F_SIZE), 255, np.uint8)
    back = cv2.warpAffine(ones, f_to_e(M_e2f), (E_SIZE, E_SIZE), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if erode_px > 0 and not back.all():
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        # pad with "covered" so the image border itself is not eroded
        back = cv2.erode(np.pad(back.astype(np.uint8), erode_px, constant_values=1), k)[
            erode_px:-erode_px, erode_px:-erode_px] > 0
    return back


def e_frame_coverage(M_e2f: np.ndarray) -> float:
    """Fraction of E-frame pixels covered by the F-frame canvas (1.0 = fully covered)."""
    return float(coverage_mask(M_e2f).mean())
