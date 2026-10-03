"""P4-T2 — no-reference input-quality features (runtime-computable, reference-free).

Every feature uses only the input image and the FES detection contract (`fes_data.preprocess.preprocess` in
face_crop mode). Nothing reads the clean reference, the degradation label or its parameters. The same function
serves the synthetic populations (labelled SYNTHETIC) and real-world captures (labelled REAL_WORLD).

Frames
  native   the input exactly as received (s × s for fes-syn-v1 chips)
  C-frame  the input resized to 112 × 112 (INTER_AREA down / INTER_CUBIC up). Resolution-normalised blur/contrast.

Feature definitions (QUALITY_VERSION pins these formulas; any change bumps the version)
  in_min_side                 min(h, w) of the native input, px
  det_status, det_score       FES detection contract (YuNet, face_crop mode with pad-and-relax retry)
  det_retry                   detection needed the crop retry
  iod_px                      detected inter-ocular distance in native px (NaN if no face)
  roll_deg, yaw_proxy         landmark pose proxies (P2-T6); NaN if no face
  align_resid                 RMS residual of the 5 detected landmarks after the best similarity fit to the FFHQ
                              template, / template inter-ocular distance (0 = perfect geometry)
  lap_var_c                   Laplacian variance of C-frame luma (sharpness; also falls with low resolution)
  lap_norm_c                  lap_var_c / (RMS contrast²) — sharpness decoupled from global contrast
  hf_ratio_c                  share of C-frame spectral power above 0.25 cycles/px (blur ↓)
  noise_sigma                 Immerkær (1996) fast noise estimate on native luma (8-bit units)
  blockiness                  JPEG 8×8 boundary / interior absolute gradient ratio on native luma (1 ≈ none);
                              NaN when the native side < 32
  luma_mean, luma_p05, luma_p95, dark_clip, bright_clip, rms_contrast, dyn_range   exposure / contrast (native)
  occlusion                   NOT MEASURED (no eligible occlusion model; P2-T6) — always NaN, recorded explicitly
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fes_data import preprocess as P

QUALITY_VERSION = "p4-quality-1.0.0"
C_SIZE = 112
FIELDS = ["in_min_side", "det_status", "det_score", "det_retry", "iod_px", "roll_deg", "yaw_proxy", "align_resid",
          "lap_var_c", "lap_norm_c", "hf_ratio_c", "noise_sigma", "blockiness", "luma_mean", "luma_p05",
          "luma_p95", "dark_clip", "bright_clip", "rms_contrast", "dyn_range", "occlusion"]
NAN = float("nan")


def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


def to_c_frame(img: np.ndarray) -> np.ndarray:
    n = min(img.shape[:2])
    interp = cv2.INTER_AREA if n > C_SIZE else cv2.INTER_CUBIC
    return cv2.resize(img, (C_SIZE, C_SIZE), interpolation=interp)


def noise_sigma_immerkaer(y: np.ndarray) -> float:
    """Immerkær, 'Fast Noise Variance Estimation', CVIU 1996. Image edges inflate it slightly (documented)."""
    h, w = y.shape
    if h < 3 or w < 3:
        return NAN
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    r = cv2.filter2D(y, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2.0) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


def blockiness(y: np.ndarray, block: int = 8) -> float:
    h, w = y.shape
    if min(h, w) < 32:
        return NAN
    dx = np.abs(np.diff(y, axis=1))                        # dx[:, j] = |y[:, j+1] - y[:, j]|
    dy = np.abs(np.diff(y, axis=0))
    bx = np.zeros(dx.shape[1], bool)
    bx[block - 1::block] = True                            # crossings between columns 7|8, 15|16, ...
    by = np.zeros(dy.shape[0], bool)
    by[block - 1::block] = True
    boundary = (dx[:, bx].sum() + dy[by, :].sum()) / max(1, dx[:, bx].size + dy[by, :].size)
    interior = (dx[:, ~bx].sum() + dy[~by, :].sum()) / max(1, dx[:, ~bx].size + dy[~by, :].size)
    return float(boundary / max(interior, 1e-6))


def hf_ratio(y: np.ndarray, cutoff: float = 0.25) -> float:
    y = y - y.mean()
    win = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
    p = np.abs(np.fft.fftshift(np.fft.fft2(y * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(y.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(y.shape[1]))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    tot = p.sum()
    return float(p[r > cutoff].sum() / tot) if tot > 0 else 0.0


def align_residual(lm: np.ndarray) -> float:
    T = P.FFHQ_TEMPLATE_512
    M = P.umeyama_similarity(lm, T)
    mapped = lm @ M[:, :2].T + M[:, 2]
    return float(np.sqrt(((mapped - T) ** 2).sum(1).mean()) / P.TEMPLATE_EYE_DIST_512)


def features(img: np.ndarray, detector: P.YuNetDetector) -> dict:
    """All P4-T2 features for one input image (BGR uint8). Deterministic."""
    y = _luma(img)
    yc = _luma(to_c_frame(img))
    rms = float(y.std())
    lap = float(cv2.Laplacian(yc, cv2.CV_64F).var())
    f = {"in_min_side": int(min(img.shape[:2])),
         "lap_var_c": lap, "lap_norm_c": lap / max(float(yc.std()) ** 2, 1e-6), "hf_ratio_c": hf_ratio(yc),
         "noise_sigma": noise_sigma_immerkaer(y), "blockiness": blockiness(y),
         "luma_mean": float(y.mean()), "luma_p05": float(np.percentile(y, 5)), "luma_p95": float(np.percentile(y, 95)),
         "dark_clip": float((y <= 5).mean()), "bright_clip": float((y >= 250).mean()), "rms_contrast": rms,
         "dyn_range": float(np.percentile(y, 95) - np.percentile(y, 5)), "occlusion": NAN}
    res = P.preprocess(img, detector, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    f["det_status"] = res.status
    f["det_retry"] = bool(res.meta.get("crop_retry"))
    if res.detection is not None:
        lm = np.asarray(res.detection.landmarks, np.float64)
        roll, yaw = P.pose_proxies(lm) if P.landmarks_plausible(lm) else (NAN, NAN)
        f.update(det_score=float(res.detection.score), iod_px=float(res.detection.eye_dist), roll_deg=roll,
                 yaw_proxy=yaw, align_resid=align_residual(lm))
    else:
        f.update(det_score=NAN, iod_px=NAN, roll_deg=NAN, yaw_proxy=NAN, align_resid=NAN)
    return {k: (round(v, 6) if isinstance(v, float) and not math.isnan(v) else v) for k, v in f.items()}
