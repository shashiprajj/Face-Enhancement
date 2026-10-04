"""R5-3 — real-world quality gate (p5-rwgate). Replaces the synthetic-calibrated T6 rules + T7 buckets FOR REAL INPUTS;
the frozen T6/T7 files stay unchanged and selectable (fes_service.pipeline gate="t6").

Why (R5-3 diagnosis): on 200 real photos the frozen policy routed 1 to A4 because
  * T6 blur rules require noise σ ≤ 0.28 — synthetic blur is noise-free, real blur always carries grain;
  * the T7 LOWRES bucket (IOD < 71.4 px = 2 % quantile of 224-px FairFace chips) sends every smaller face to original.

Contract (one decision per input, evaluated in this order; every outcome except ENHANCE_A4 returns the ORIGINAL):
  1 UNSAFE_OR_UNCERTAIN  T6 safety rules unchanged: no valid face / det score < 0.85 / |yaw| > 0.25 / NaN pose
  2 UNSAFE_OR_UNCERTAIN  insufficient_face_size      IOD < iod_min_recognizable
  3 UNSAFE_OR_UNCERTAIN  beyond_evidence_extreme_blur lapd1_c < lapd1_floor (blurrier than anything A4 was measured on)
  4 DO_NOT_ENHANCE       photometric / compression    A4 has no identity benefit there (P4-T4 P7) → original
  5 ENHANCE_CANDIDATE→A4 strong_noise                 noise_sigma ≥ noise_min        and IOD ≥ iod_min_a4
  6 ENHANCE_CANDIDATE→A4 strong_blur                  lapd1_c ≤ blur_max             and IOD ≥ iod_min_a4
  7 DO_NOT_ENHANCE       degraded_below_a4_size       rule 5/6 degradation but IOD < iod_min_a4
  8 DO_NOT_ENHANCE       degraded_no_a4_evidence      outside the clean-reference range (lapd1 / noise / IOD; 2 % / 98 %
                                                       quantiles of clean DEV references, no outcome data) but not in A4's
                                                       evidenced zone (e.g. moderate blur, mild noise, small face)
  9 DO_NOT_ENHANCE       acceptable                   looks like a clean reference

New feature (grain-robust sharpness, p5-rwfeat-1.0.0):
  lapd1_c = Var(Laplacian(G_σ=1 * Y_c)) / Var(Y_c)  on the 112-px C-frame luma. The σ=1 pre-filter removes pixel-level
  grain before the Laplacian: adding σ=4 grain changes the old lap_norm_c of extreme-blur chips 6× (0.0047 → 0.027)
  and lapd1_c not at all (0.0025 → 0.0026), while blur levels stay ordered (mild 0.0167 … extreme 0.0025).
Thresholds come from configs/phase5/r5_rwgate.json (SHA-256 sealed; written by scripts/r5_rwgate_calibrate.py from DEV
data only). A missing feature (NaN) never enhances.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_bench import quality as Q
from fes_bench import routing as RT
from fes_data import preprocess as P
from fes_data.common import repo_path

RWGATE_VERSION = "p5-rwgate-1.0.0"
RWFEAT_VERSION = "p5-rwfeat-1.0.0"
CONFIG = "configs/phase5/r5_rwgate.json"
A4 = "A4"


def lapd1(img: np.ndarray) -> float:
    y = cv2.cvtColor(Q.to_c_frame(img), cv2.COLOR_BGR2GRAY).astype(np.float64)
    v = float(y.var())
    if v < 1e-6:
        return float("nan")
    return float(cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_64F).var() / v)


def is_gray(img: np.ndarray, tol: int = 2) -> bool:
    """All three channels equal within `tol` grey levels at the 99th percentile (scans saved as RGB count as grey)."""
    i = img.astype(np.int16)
    d = np.maximum(np.abs(i[..., 0] - i[..., 1]), np.abs(i[..., 1] - i[..., 2]))
    return bool(np.percentile(d, 99) <= tol)


def rw_features(img: np.ndarray, det: P.YuNetDetector) -> dict:
    f = Q.features(img, det)
    f["lapd1_c"] = round(lapd1(img), 6)
    f["is_gray"] = is_gray(img)
    return f


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def load_config(path: str = CONFIG) -> dict:
    cfg = json.loads(repo_path(path).read_text(encoding="utf-8"))
    assert cfg["sha256"] == GT.gate_hash(cfg), "r5_rwgate.json modified after freeze"
    assert cfg["gate_version"] == RWGATE_VERSION and cfg["route_model"] == A4
    return cfg


def decide(f: dict, cfg: dict) -> tuple[str, str, str, str]:
    """→ (decision, reason, category, route). route ∈ {A4, B0}."""
    t = cfg["thresholds"]
    why = GT.safety(f)
    if why:
        return GT.UNSAFE, why, "uncertain_or_no_face", RT.ORIGINAL
    iod, lap, ns = f.get("iod_px"), f.get("lapd1_c"), f.get("noise_sigma")
    if _nan(iod) or iod < t["iod_min_recognizable"]:
        return GT.UNSAFE, "insufficient_face_size", "tiny", RT.ORIGINAL
    if not _nan(lap) and lap < t["lapd1_floor"]:
        return GT.UNSAFE, "beyond_evidence_extreme_blur", "unrecognizable", RT.ORIGINAL
    lm, rc, bk = f.get("luma_mean"), f.get("rms_contrast"), f.get("blockiness")
    if (not _nan(lm) and lm < t["luma_mean_lo"]) or (not _nan(rc) and rc < t["rms_contrast_lo"]):
        return GT.DNE, "photometric_no_a4_evidence", "photometric", RT.ORIGINAL
    if not _nan(bk) and bk > t["blockiness_hi"]:
        return GT.DNE, "compression_no_a4_evidence", "compression", RT.ORIGINAL
    noisy = not _nan(ns) and ns >= t["noise_min"]
    blurry = not _nan(lap) and lap <= t["blur_max"]
    if noisy or blurry:
        kind = "strong_noise" if noisy else "strong_blur"
        if iod >= t["iod_min_a4"]:
            return GT.ENH, kind, "degraded_recognizable", A4
        return GT.DNE, f"degraded_below_a4_size:{kind}", "degraded_small", RT.ORIGINAL
    # degraded in a way A4 has no identity evidence for (moderate blur, mild noise, small face): original, but say so
    kinds = [k for k, bad in (("blur", not _nan(lap) and lap < t["clean_lapd1_lo"]),
                              ("noise", not _nan(ns) and ns > t["clean_noise_hi"]),
                              ("lowres", iod < t["clean_iod_lo"])) if bad]
    if kinds:
        return GT.DNE, "degraded_no_a4_evidence:" + "+".join(kinds), "degraded_no_a4_evidence", RT.ORIGINAL
    return GT.DNE, "acceptable", "acceptable", RT.ORIGINAL


class RWRouter(RT.Router):
    """The frozen runtime router (fallbacks, timeout, output guard) with the real-world gate as its planner."""

    def __init__(self, cfg: dict, adapters: dict, features_fn, face_check, timeout_s: float = 120.0):
        super().__init__([], {}, {}, adapters, features_fn, face_check, timeout_s=timeout_s)
        self.cfg = cfg

    def plan(self, f: dict) -> RT.RouteRecord:
        d, why, cat, route = decide(f, self.cfg)
        rec = RT.RouteRecord(decision=d, reason=why, escalate=d == GT.UNSAFE, bucket=cat, route=route)
        return rec
