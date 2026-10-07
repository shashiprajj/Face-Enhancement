"""R5-5 — guarded NAFNet-GoPro route: dedicated input gate + output artifact guard (p5-nafnet-route).

    face chip → rw features (p5-rwfeat-1.0.0) + texture features (p5-nafnet-feat-1.0.0)
              → decide() ─ ineligible ──────────────────────────────→ ORIGINAL (reason NAFNET_REJECT_*)
                         └ NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR → NAFNet-GoPro (isolated worker)
                                → artifact_guard() ─ fail → ORIGINAL (reasons ARTIFACT_*)
                                                   └ pass → NAFNet output (native-size face chip)

Why a separate policy (R5-ARCHIVE, docs/phase_5_rapid/R5_Archive_Deblur_Dataset_REPORT.md): NAFNet-GoPro is the only
identity-positive deblurring model found (SFace Δ +0.16 on noise-free synthetic blur) but it is brittle — colour streaks
on 14 % of grainy blur and 6.5 % destroyed real printed / pixelated / halftone photos. So: blur only, low noise only,
and every output is checked before it is returned. The A4 policy (fes_service.rwgate) is NOT modified.

Decision order (first match wins; every outcome except ELIGIBLE returns the ORIGINAL; a missing feature never enhances):
  1 NAFNET_REJECT_NO_FACE / _LOW_DETECTION / _UNSAFE_ALIGNMENT   T6 safety rules, unchanged
  2 NAFNET_REJECT_LOW_FACE_SIZE         IOD < iod_min
  3 NAFNET_REJECT_PHOTOMETRIC           luma_mean < luma_mean_lo or rms_contrast < rms_contrast_lo
  4 NAFNET_REJECT_COMPRESSION           blockiness > blockiness_hi
  5 NAFNET_REJECT_PIXELATION            block-grid periodicity of the gradient profile > pixelation_max
  6 NAFNET_REJECT_HALFTONE              periodic high-frequency peaks AND high-frequency energy above their maxima
  7 NAFNET_REJECT_HIGH_NOISE            noise_sigma > noise_max   (strong blur + grain is ORIGINAL, never NAFNet)
  8 NAFNET_REJECT_NOT_BLUR              lapd1_c ≥ clean_lapd1_lo  (sharp / clean-looking, or a non-blur degradation)
  9 NAFNET_REJECT_MILD_BLUR             blur_max < lapd1_c < clean_lapd1_lo
 10 NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR   lapd1_c ≤ blur_max
Thresholds: configs/phase5/r5_nafnet_route.json (SHA-256 sealed; scripts/r5_nafnet_calibrate.py, DEV data only).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_data.common import repo_path

ROUTE_VERSION = "p5-nafnet-route-1.0.0"
FEAT_VERSION = "p5-nafnet-feat-1.0.0"
CONFIG = "configs/phase5/r5_nafnet_route.json"
MODEL = "NAFNET_GOPRO"
ROUTE_ELIGIBLE, ROUTE_INELIGIBLE = "STRONG_LOW_NOISE_BLUR", "NAFNET_INELIGIBLE"

ELIGIBLE = "NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR"
R_NO_FACE, R_LOW_DET, R_ALIGN = "NAFNET_REJECT_NO_FACE", "NAFNET_REJECT_LOW_DETECTION", "NAFNET_REJECT_UNSAFE_ALIGNMENT"
R_SIZE, R_PHOTO, R_COMP = "NAFNET_REJECT_LOW_FACE_SIZE", "NAFNET_REJECT_PHOTOMETRIC", "NAFNET_REJECT_COMPRESSION"
R_PIX, R_HALF, R_NOISE = "NAFNET_REJECT_PIXELATION", "NAFNET_REJECT_HALFTONE", "NAFNET_REJECT_HIGH_NOISE"
R_NOT_BLUR, R_MILD = "NAFNET_REJECT_NOT_BLUR", "NAFNET_REJECT_MILD_BLUR"
R_MISSING = "NAFNET_REJECT_MISSING_FEATURE"

A_NUMERIC, A_FACE = "ARTIFACT_NUMERIC_INVALID", "ARTIFACT_FACE_INVALID"
A_CHROMA, A_HF, A_GRAY = "ARTIFACT_CHROMA_P99", "ARTIFACT_HIGH_FREQ_COLOR", "ARTIFACT_GRAYSCALE_VIOLATION"


# ------------------------------------------------------------------ texture features (p5-nafnet-feat-1.0.0)

def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


def _periodicity(profile: np.ndarray, max_lag: int = 32) -> float:
    """Max normalised autocorrelation (lags 2..max_lag) of a detrended 1-D gradient profile. Block / grid structure
    (pixelation, heavy JPEG blocking) puts the gradient energy at regular positions → close to 1; natural or blurred
    content → low. The 9-tap moving-average detrend removes the smooth face-shape trend first."""
    t = np.convolve(profile, np.ones(9) / 9.0, "same")
    r = (profile - t)[4:-4]
    r = r - r.mean()
    v = float((r * r).sum())
    if v < 1e-9:
        return 0.0
    return float(max((r[:-k] * r[k:]).sum() / v for k in range(2, min(max_lag + 1, len(r) // 3))))


def pixelation_score(img: np.ndarray) -> float:
    """min over x/y of the gradient-profile periodicity (both axes: pixel blocks are 2-D). NaN below 24 px."""
    y = _luma(img)
    if min(y.shape) < 24:
        return float("nan")
    px = np.abs(np.diff(y, axis=1)).mean(0)
    py = np.abs(np.diff(y, axis=0)).mean(1)
    return round(min(_periodicity(px), _periodicity(py)), 6)


def halftone_scores(img: np.ndarray) -> tuple[float, float]:
    """(hf_peak_frac, hf_energy): share of the 8 strongest spectral bins (radius > 0.08 cycles/px) in the windowed
    spectrum of the high-pass residual (luma − G_σ=2 * luma), and the residual's RMS relative to the luma RMS.
    A halftone / dot screen is a strong, PERIODIC high-frequency pattern: both are high. Grain is strong but flat
    (low peak share); blur is weak (low energy); a sharp face is neither strongly periodic nor dot-dominated."""
    y = _luma(img)
    if min(y.shape) < 24:
        return float("nan"), float("nan")
    hp = y - cv2.GaussianBlur(y, (0, 0), 2.0)
    h, w = y.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft2(hp * np.outer(np.hanning(h), np.hanning(w))))) ** 2
    fy, fx = np.fft.fftshift(np.fft.fftfreq(h)), np.fft.fftshift(np.fft.fftfreq(w))
    a = spec[np.hypot(*np.meshgrid(fy, fx, indexing="ij")) > 0.08]
    tot = float(a.sum())
    peak = float(np.sort(a)[-8:].sum() / tot) if tot > 0 else 0.0
    energy = float(hp.std() / max(float(y.std()), 1e-6))
    return round(peak, 6), round(energy, 6)


def texture_features(img: np.ndarray) -> dict:
    peak, energy = halftone_scores(img)
    return {"pixelation": pixelation_score(img), "hf_peak_frac": peak, "hf_energy": energy}


# ------------------------------------------------------------------ config + input gate

def load_config(path: str = CONFIG) -> dict:
    cfg = json.loads(repo_path(path).read_text(encoding="utf-8"))
    assert cfg["sha256"] == GT.gate_hash(cfg), "r5_nafnet_route.json modified after freeze"
    assert cfg["route_version"] == ROUTE_VERSION and cfg["feature_version"] == FEAT_VERSION and cfg["model"] == MODEL
    return cfg


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


@dataclass
class NafnetDecision:
    eligible: bool
    reason: str
    decision: str                       # the project's gate vocabulary: ENHANCE_CANDIDATE | DO_NOT_ENHANCE | UNSAFE_…
    route: str                          # STRONG_LOW_NOISE_BLUR | NAFNET_INELIGIBLE

    def as_dict(self) -> dict:
        return asdict(self)


def decide(f: dict, cfg: dict) -> NafnetDecision:
    """f = rw features (+ texture_features). Pure function of the features and the sealed thresholds."""
    t = cfg["thresholds"]

    def no(reason: str, decision: str = GT.DNE) -> NafnetDecision:
        return NafnetDecision(False, reason, decision, ROUTE_INELIGIBLE)

    why = GT.safety(f)
    if why:
        r = R_NO_FACE if why.startswith("no_valid_face") else (R_LOW_DET if why == "low_detection_confidence"
                                                                else R_ALIGN)
        return no(r, GT.UNSAFE)
    need = ("iod_px", "lapd1_c", "noise_sigma", "luma_mean", "rms_contrast", "pixelation", "hf_peak_frac", "hf_energy")
    missing = [k for k in need if _nan(f.get(k))]
    if missing:
        return no(f"{R_MISSING}:{'+'.join(missing)}", GT.UNSAFE)
    if f["iod_px"] < t["iod_min"]:
        return no(R_SIZE)
    if f["luma_mean"] < t["luma_mean_lo"] or f["rms_contrast"] < t["rms_contrast_lo"]:
        return no(R_PHOTO)
    if not _nan(f.get("blockiness")) and f["blockiness"] > t["blockiness_hi"]:
        return no(R_COMP)
    if f["pixelation"] > t["pixelation_max"]:
        return no(R_PIX)
    if f["hf_peak_frac"] > t["halftone_peak_max"] and f["hf_energy"] > t["halftone_energy_max"]:
        return no(R_HALF)
    if f["noise_sigma"] > t["noise_max"]:
        return no(R_NOISE)
    if f["lapd1_c"] >= t["clean_lapd1_lo"]:
        return no(R_NOT_BLUR)
    if f["lapd1_c"] > t["blur_max"]:
        return no(R_MILD)
    return NafnetDecision(True, ELIGIBLE, GT.ENH, ROUTE_ELIGIBLE)


# ------------------------------------------------------------------ output artifact guard

@dataclass
class ArtifactGuardResult:
    passed: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)


def chroma_p99(out: np.ndarray, ref: np.ndarray) -> float:
    """99th percentile of the per-pixel Lab chroma distance (a, b) between output and its input (R5-ARCHIVE metric)."""
    a, b = _lab(out), _lab(ref)
    return float(np.percentile(np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]), 99))


def hf_color_ratio(img: np.ndarray) -> float:
    """High-frequency colour energy relative to high-frequency luma energy (residual vs G_σ=1.5). Real detail lives
    mostly in luma; rainbow / metallic streaks are high-frequency CHROMA oscillation → the ratio jumps."""
    lab = _lab(img)
    hp = lab - cv2.GaussianBlur(lab, (0, 0), 1.5)
    e_c = float((hp[..., 1] ** 2 + hp[..., 2] ** 2).mean())
    e_l = float((hp[..., 0] ** 2).mean())
    return e_c / (e_l + 1.0)


def gray_chroma_p99(img: np.ndarray) -> float:
    lab = _lab(img)
    return float(np.percentile(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0), 99))


def artifact_guard(inp: np.ndarray, out, cfg: dict, face_check, input_is_gray: bool) -> ArtifactGuardResult:
    """Every check runs on the NAFNet output vs its own input chip; any failure → ORIGINAL. Numeric validity first
    (the other metrics need a sane image)."""
    g = cfg["guard"]
    res = ArtifactGuardResult(passed=False)
    m = res.metrics
    # 1 numeric / format validity
    if not isinstance(out, np.ndarray):
        res.reasons.append(A_NUMERIC)
        m["numeric"] = f"not an array: {type(out).__name__}"
        return res
    bad = ""
    if out.dtype != np.uint8:
        bad = f"dtype {out.dtype}"
        if np.issubdtype(out.dtype, np.floating) and not np.isfinite(out).all():
            bad += " with NaN/Inf"
    elif out.shape != inp.shape:
        bad = f"shape {out.shape} != input {inp.shape}"
    elif float(out.std()) < 1.0:
        bad = "near-constant image"
    if bad:                                    # unusable format: the image metrics below would be meaningless
        res.reasons.append(A_NUMERIC)
        m["numeric"] = bad
        return res
    # channel corruption (still a NUMERIC failure, but the image checks below also run and are reported)
    o, i = out.reshape(-1, 3).astype(np.float64), inp.reshape(-1, 3).astype(np.float64)
    shift = np.abs(o.mean(0) - i.mean(0))
    dead = (o.std(0) < 1.0) & (i.std(0) >= 5.0)
    m["channel_mean_shift_max"] = round(float(shift.max()), 3)
    if dead.any() or shift.max() > g["channel_mean_shift_max"]:
        res.reasons.append(A_NUMERIC)
        m["numeric"] = "channel collapsed" if dead.any() else f"channel mean shift {shift.max():.1f}"
    # 2 face still detectable on the output
    face_ok = bool(face_check(out))
    m["face_found"] = face_ok
    if not face_ok:
        res.reasons.append(A_FACE)
    # 3 Lab chroma change (p99) vs the input chip
    cp = chroma_p99(out, inp)
    m["chroma_p99"] = round(cp, 3)
    if cp > g["chroma_p99_max"]:
        res.reasons.append(A_CHROMA)
    # 4 high-frequency colour energy: absolute level AND growth over the input
    hf_o, hf_i = hf_color_ratio(out), hf_color_ratio(inp)
    m["hf_color_ratio"], m["hf_color_ratio_in"] = round(hf_o, 4), round(hf_i, 4)
    if hf_o > g["hf_color_ratio_max"] and hf_o > g["hf_color_growth_max"] * max(hf_i, 1e-3):
        res.reasons.append(A_HF)
    # 5 grayscale in → no invented colour out
    if input_is_gray:
        gc = gray_chroma_p99(out)
        m["gray_chroma_p99"] = round(gc, 3)
        if gc > g["gray_chroma_max"]:
            res.reasons.append(A_GRAY)
    res.passed = not res.reasons
    return res


# ------------------------------------------------------------------ synthetic printed-image risks (calibration/tests only)

def synth_pixelate(img: np.ndarray, k: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)


def synth_halftone(img: np.ndarray, period: float) -> np.ndarray:
    """Clustered-dot screen at 45° (newspaper / magazine print), black dots on white."""
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    h, w = y.shape
    yy, xx = np.mgrid[0:h, 0:w]
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    u, v = (xx * c + yy * s) / period, (-xx * s + yy * c) / period
    d = np.hypot(u - np.round(u), v - np.round(v)) / 0.7071
    return cv2.cvtColor(((d > np.sqrt(1 - y)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
