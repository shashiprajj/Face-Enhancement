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

ROUTE_VERSION = "p5-nafnet-route-1.0.0"              # the live/default (R5-6) route
# R5-8 research candidate (opt-in profile "r5_8"; never the default): R5-7 guard values + framing fixes + local guard
CANDIDATE_CONFIG = "configs/phase5/r5_8_nafnet_candidate.json"
CANDIDATE_ROUTE_VERSION = "p5-nafnet-route-1.1.0"
# R5-10 research candidate (opt-in profile "r5_10"; never the default): the R5-8 candidate with ONE change — the input
# blur-eligibility boundary blur_max, recalibrated on DEV (scripts/r5_10_dev_analyze.py, pre-registered protocol)
R5_10_CONFIG = "configs/phase5/r5_10_nafnet_candidate.json"
R5_10_ROUTE_VERSION = "p5-nafnet-route-1.2.0"
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
A_CHROMA_TAIL = "ARTIFACT_CHROMA_P999"                 # candidate config only (guard key chroma_p999_max)


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
    assert cfg["route_version"] in (ROUTE_VERSION, CANDIDATE_ROUTE_VERSION, R5_10_ROUTE_VERSION, R6_ROUTE_VERSION), \
        cfg["route_version"]
    assert cfg["feature_version"] == FEAT_VERSION and cfg["model"] == MODEL
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


def chroma_dist(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a, b = _lab(out), _lab(ref)
    return np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2])


def chroma_p99(out: np.ndarray, ref: np.ndarray) -> float:
    """99th percentile of the per-pixel Lab chroma distance (a, b) between output and its input (R5-ARCHIVE metric)."""
    return float(np.percentile(chroma_dist(out, ref), 99))


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
    cd = chroma_dist(out, inp)
    cp = float(np.percentile(cd, 99))
    m["chroma_p99"] = round(cp, 3)
    if cp > g["chroma_p99_max"]:
        res.reasons.append(A_CHROMA)
    if "chroma_p999_max" in g:                            # candidate config only; absent in the live 1.0.0 config
        ct = float(np.percentile(cd, 99.9))
        m["chroma_p999"] = round(ct, 3)
        if ct > g["chroma_p999_max"]:
            res.reasons.append(A_CHROMA_TAIL)
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


# ------------------------------------------------------------------ R5-8 local-patch artifact guard (candidate only)
# p5-nafnet-local-artifact-1.0.0 — deterministic spatial statistics of the output-vs-input chroma change and of the
# high-frequency colour energy NAFNet adds; computed on the chip (≤ 224 px) with box filters + one connected-component
# pass: bounded runtime, no model, no network. Used only when the route config has a "local_artifact" section.

LOCAL_VERSION = "p5-nafnet-local-artifact-1.0.0"
A_LOCAL, A_LOCAL_ERROR = "ARTIFACT_LOCAL_PATCH", "ARTIFACT_LOCAL_GUARD_ERROR"
LOCAL_WIN = 8


def local_features(inp: np.ndarray, out: np.ndarray, win: int = LOCAL_WIN) -> dict:
    """All candidate local signals (R5-8 DEV study). inp/out: same-size uint8 BGR chips."""
    A, B = _lab(out), _lab(inp)
    d = np.hypot(A[..., 1] - B[..., 1], A[..., 2] - B[..., 2]).astype(np.float32)
    box = lambda x: cv2.blur(x, (win, win), borderType=cv2.BORDER_REFLECT_101)  # noqa: E731
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.0)  # noqa: E731
    ha, hb = hp(A), hp(B)
    ec_out = (ha[..., 1] ** 2 + ha[..., 2] ** 2).astype(np.float32)
    ec_in = (hb[..., 1] ** 2 + hb[..., 2] ** 2).astype(np.float32)
    el_out = (ha[..., 0] ** 2).astype(np.float32)
    dm = box(d)
    growth = box(np.maximum(ec_out - ec_in, 0))
    ratio = box(ec_out) / (box(el_out) + 4.0)
    f = {"l_dmean_max": float(dm.max()), "l_dmean_excess": float(dm.max() - np.median(d)),
         "l_hfc_growth_max": float(growth.max()), "l_hfc_ratio_max": float(ratio.max()),
         "l_hfc_growth_p999": float(np.percentile(growth, 99.9))}
    # "dottedness": second-difference energy present in BOTH x and y (min of the two) — a dot / grid oscillates in both
    # directions, a sharpened edge only across the edge. Output's increase over the input, 8×8 window maximum.
    k = np.array([[1.0, -2.0, 1.0]], np.float32)

    def iso(x):
        x = x.astype(np.float32)
        return np.minimum(cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101) ** 2,
                          cv2.filter2D(x, -1, k.T, borderType=cv2.BORDER_REFLECT_101) ** 2)
    dot_c = np.maximum(iso(A[..., 1]) + iso(A[..., 2]) - iso(B[..., 1]) - iso(B[..., 2]), 0)
    dot_l = np.maximum(iso(A[..., 0]) - iso(B[..., 0]), 0)
    f["l_dot_c_max"] = float(box(dot_c).max())
    f["l_dot_l_max"] = float(box(dot_l).max())
    for t in (20, 30):
        n, lab_, stats, _ = cv2.connectedComponentsWithStats((d > t).astype(np.uint8), connectivity=8)
        f[f"l_cc{t}_max"] = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        f[f"l_cc{t}_frac"] = round(f[f"l_cc{t}_max"] / float(d.size), 6)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}


def local_artifact_guard(inp: np.ndarray, out, cfg: dict, input_is_gray: bool) -> ArtifactGuardResult:
    """Candidate local-patch guard: any configured local signal above its DEV-calibrated limit → reject (ORIGINAL).
    Never modifies `out`. Invalid input → reject."""
    lc = cfg["local_artifact"]
    assert lc["version"] == LOCAL_VERSION, lc["version"]
    res = ArtifactGuardResult(passed=False)
    if not (isinstance(out, np.ndarray) and isinstance(inp, np.ndarray) and out.dtype == np.uint8
            and out.shape == inp.shape and out.ndim == 3):
        res.reasons.append(A_LOCAL_ERROR)
        res.metrics["error"] = "invalid image for the local guard"
        return res
    f = local_features(inp, out, lc.get("win", LOCAL_WIN))
    res.metrics.update(f)
    if not all(np.isfinite(v) for v in f.values()):
        res.reasons.append(A_LOCAL_ERROR)
        return res
    for k, limit in lc["limits"].items():
        if f[k] > limit:
            res.reasons.append(f"{A_LOCAL}:{k}")
    res.passed = not res.reasons
    return res


# ------------------------------------------------------------------ R6 stability-verified consensus (candidate only)
# p6-nafnet-stability-1.0.0 (R6-0: Planning/WelOrbit_R6-0_Post_Enhancement_Verifier_REPORT.md). NAFNet-GoPro is
# numerically unstable on some inputs: mirroring the input (which must not change a deblur) then makes one or more runs
# explode into a dotted / grid / noise patch, and the localized artifacts R5-6..R5-10 returned are the same failure.
# The candidate runs NAFNet on the chip and on its h / v / hv mirrors (each mirrored back), REJECTS the request
# (ORIGINAL) when any run carries the explosion tracer — colour dottedness vs the input above the DEV limit — and
# otherwise returns the per-pixel MEDIAN of the four runs with the INPUT's chrominance (luma-only enhancement: the
# output cannot change colour at all). Used only when the route config has a "stability" section (profile r6).

R6_CONFIG = "configs/phase5/r6_nafnet_candidate.json"
R6_ROUTE_VERSION = "p6-nafnet-route-1.0.0"
STABILITY_VERSION = "p6-nafnet-stability-1.0.0"
A_UNSTABLE, A_UNSTABLE_ERROR = "ARTIFACT_UNSTABLE_RUN", "ARTIFACT_STABILITY_CHECK_ERROR"
FLIPS = (None, 1, 0, -1)                               # identity, horizontal, vertical, both (cv2.flip codes)


def dot_c_max(inp: np.ndarray, out: np.ndarray, valid: np.ndarray | None = None, win: int = LOCAL_WIN) -> float:
    """Colour dottedness of `out` over `inp` (the R5-8 l_dot_c_max formula), restricted to `valid` pixels (the
    out-of-image chip area is excluded when a mask is given). Unmasked, it equals local_features()['l_dot_c_max']."""
    A, B = _lab(out).astype(np.float32), _lab(inp).astype(np.float32)
    k = np.array([[1.0, -2.0, 1.0]], np.float32)

    def iso(x):
        return np.minimum(cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101) ** 2,
                          cv2.filter2D(x, -1, k.T, borderType=cv2.BORDER_REFLECT_101) ** 2)
    d = np.maximum(iso(A[..., 1]) + iso(A[..., 2]) - iso(B[..., 1]) - iso(B[..., 2]), 0)
    if valid is not None:
        d = d * valid
    return float(cv2.blur(d.astype(np.float32), (win, win), borderType=cv2.BORDER_REFLECT_101).max())


def consensus_luma(inp: np.ndarray, runs: list) -> np.ndarray:
    """Per-pixel median of the runs (uint8 BGR, same size as inp), then the median's luminance (Y) with the input's
    chrominance (Cr, Cb). With >= 3 artifact-free runs every output pixel lies inside their range."""
    med = np.median(np.stack(runs).astype(np.float32), 0).round().clip(0, 255).astype(np.uint8)
    a, b = cv2.cvtColor(med, cv2.COLOR_BGR2YCrCb), cv2.cvtColor(inp, cv2.COLOR_BGR2YCrCb)
    a[..., 1:] = b[..., 1:]
    return cv2.cvtColor(a, cv2.COLOR_YCrCb2BGR)


def stability_check(runs_dot_c, cfg: dict) -> ArtifactGuardResult:
    """R6 stability rule: every run's colour-dottedness tracer must be <= the DEV limit (rule "any": a single unstable
    run rejects the request). Missing / malformed tracer values reject."""
    sc = cfg["stability"]
    assert sc["version"] == STABILITY_VERSION, sc["version"]
    res = ArtifactGuardResult(passed=False)
    try:
        vals = [float(v) for v in runs_dot_c]
    except (TypeError, ValueError):
        vals = []
    if len(vals) != len(FLIPS) or not all(np.isfinite(vals)):
        res.reasons.append(A_UNSTABLE_ERROR)
        res.metrics["runs_dot_c"] = runs_dot_c
        return res
    limit = float(sc["run_dot_c_max"])
    n_bad = sum(v > limit for v in vals)
    res.metrics.update(runs_dot_c=[round(v, 4) for v in vals], run_dot_c_max=round(max(vals), 4),
                       n_unstable=n_bad, limit=limit, rule=sc["rule"])
    if n_bad >= int(sc["max_unstable_runs"]) + 1:
        res.reasons.append(f"{A_UNSTABLE}:{n_bad}/{len(vals)}")
    res.passed = not res.reasons
    return res
