"""Controlled, deterministic degradation of clean reference faces (P2-T4).

Design rules
  * The clean reference is read-only input; outputs are new files.
  * Every random draw comes from numpy PCG64 seeded by sha256(global_seed | ref_id | profile | severity | version),
    so a degraded image is reproducible from (reference bytes, profile, severity, version) alone.
  * Only global, spatially uniform operators (resampling, blur, noise, JPEG, exposure, contrast) are used —
    nothing edits facial regions, so identity is not intentionally altered.
  * All exact parameters actually used are returned for the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np

DEGRADE_VERSION = "degrade-1.0.0"

# Spatial parameters are defined for the 224-px FairFace reference scale (P2-T3). P1-T7 defined them at the
# 512 scale; blur sigma and motion length are halved (224/512 ≈ 0.44, rounded to a clean 0.5 factor).
PROFILES: dict[str, dict[str, dict]] = {
    "lowres": {            # DG1: downscale factor (INTER_AREA = box anti-aliased); output stays at LR size
        "mild": {"factor": 2}, "moderate": {"factor": 4}, "severe": {"factor": 8}, "extreme": {"factor": 16}},
    "gaussblur": {         # DG2: isotropic Gaussian sigma (px @224)
        "mild": {"sigma": 0.5}, "moderate": {"sigma": 1.0}, "severe": {"sigma": 2.0}, "extreme": {"sigma": 4.0}},
    "motionblur": {        # DG3: linear kernel length (px @224), angle drawn uniformly in [0,180)
        "mild": {"length": 4}, "moderate": {"length": 8}, "severe": {"length": 13}, "extreme": {"length": 21}},
    "noise": {             # DG4: i.i.d. Gaussian noise per channel, sigma on 0-255 scale
        "mild": {"sigma": 5}, "moderate": {"sigma": 15}, "severe": {"sigma": 25}, "extreme": {"sigma": 50}},
    "jpeg": {              # DG5: IJG quality — below the FairFace Q≈75 source floor (P2-T2 finding)
        "mild": {"quality": 60}, "moderate": {"quality": 40}, "severe": {"quality": 20}, "extreme": {"quality": 10}},
    "underexposure": {     # DG6: exposure change in linear light (EV stops), then back to sRGB
        "mild": {"ev": -1.0}, "moderate": {"ev": -2.0}, "severe": {"ev": -3.0}},
    "lowcontrast": {       # DG7: scale deviations from the per-image mean luma by alpha
        "mild": {"alpha": 0.7}, "moderate": {"alpha": 0.5}, "severe": {"alpha": 0.3}},
    "combined": {          # DG8a classical BFR chain: blur → downscale → noise → JPEG (ranges sampled per image)
        "mild":     {"sigma": [0.2, 1.0], "factor": 2, "noise": [0, 5],  "quality": [50, 70]},
        "moderate": {"sigma": [0.5, 2.0], "factor": 4, "noise": [0, 10], "quality": [35, 60]},
        "severe":   {"sigma": [1.0, 4.0], "factor": 8, "noise": [0, 20], "quality": [20, 45]}},
    "realistic2": {        # DG8b second-order chain (two rounds of blur/resize/noise/JPEG); final size = 224/factor
        "mild":     {"factor": 2, "sigma": [0.2, 1.0], "noise": [0, 5],  "quality": [50, 75]},
        "moderate": {"factor": 4, "sigma": [0.3, 1.5], "noise": [0, 10], "quality": [35, 65]},
        "severe":   {"factor": 8, "sigma": [0.5, 2.0], "noise": [0, 15], "quality": [25, 50]}},
}


def derive_seed(global_seed: int, ref_id: str, profile: str, severity: str) -> int:
    key = f"{global_seed}|{ref_id}|{profile}|{severity}|{DEGRADE_VERSION}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


# ------------------------------------------------------------------ primitive operators (uint8 BGR in/out)

def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    k = 2 * math.ceil(3 * sigma) + 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalized line kernel of `length` px through the centre at `angle_deg` (anti-aliased raster, fixed supersampling)."""
    ss = 8
    size = length if length % 2 == 1 else length + 1
    big = np.zeros((size * ss, size * ss), np.float32)
    c = (size * ss - 1) / 2.0
    dx = math.cos(math.radians(angle_deg)) * (length * ss - 1) / 2.0
    dy = math.sin(math.radians(angle_deg)) * (length * ss - 1) / 2.0
    cv2.line(big, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, thickness=ss)
    k = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    return cv2.filter2D(img, -1, motion_kernel(length, angle_deg), borderType=cv2.BORDER_REFLECT_101)


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    return _to_u8(img.astype(np.float64) + rng.normal(0.0, sigma, img.shape))


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.clip(x, 0, None), 1 / 2.4) - 0.055)


def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    lin = _srgb_to_linear(img.astype(np.float64) / 255.0) * (2.0 ** ev)
    return _to_u8(_linear_to_srgb(np.clip(lin, 0, 1)) * 255.0)


def contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    x = img.astype(np.float64)
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64).mean()
    return _to_u8(mean + alpha * (x - mean))


# ------------------------------------------------------------------ profile application

def _u(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def apply_profile(ref_bgr: np.ndarray, profile: str, severity: str, seed: int) -> tuple[np.ndarray, dict]:
    """Return (degraded image, exact parameters used)."""
    spec = PROFILES[profile][severity]
    rng = np.random.Generator(np.random.PCG64(seed))
    img = ref_bgr
    p: dict = {"profile": profile, "severity": severity, "seed": seed, "version": DEGRADE_VERSION}
    if profile == "lowres":
        out = downscale(img, spec["factor"])
        p.update(factor=spec["factor"], resample="INTER_AREA")
    elif profile == "gaussblur":
        out = gaussian_blur(img, spec["sigma"])
        p.update(sigma=spec["sigma"])
    elif profile == "motionblur":
        angle = round(_u(rng, (0.0, 180.0)), 3)
        out = motion_blur(img, spec["length"], angle)
        p.update(length=spec["length"], angle_deg=angle)
    elif profile == "noise":
        out = add_gaussian_noise(img, spec["sigma"], rng)
        p.update(sigma=spec["sigma"])
    elif profile == "jpeg":
        out = jpeg(img, spec["quality"])
        p.update(quality=spec["quality"])
    elif profile == "underexposure":
        out = exposure(img, spec["ev"])
        p.update(ev=spec["ev"])
    elif profile == "lowcontrast":
        out = contrast(img, spec["alpha"])
        p.update(alpha=spec["alpha"])
    elif profile == "combined":
        sigma = round(_u(rng, spec["sigma"]), 4)
        noise = round(_u(rng, spec["noise"]), 4)
        q = int(round(_u(rng, spec["quality"])))
        x = gaussian_blur(img, sigma)
        x = downscale(x, spec["factor"])
        x = add_gaussian_noise(x, noise, rng)
        out = jpeg(x, q)
        p.update(chain="blur>downscale>noise>jpeg", sigma=sigma, factor=spec["factor"], noise_sigma=noise, quality=q)
    elif profile == "realistic2":
        h, w = img.shape[:2]
        f = spec["factor"]
        f1 = round(math.sqrt(f), 4)  # first-order reduction; second order reaches the final factor
        s1, s2 = round(_u(rng, spec["sigma"]), 4), round(_u(rng, spec["sigma"]), 4)
        n1, n2 = round(_u(rng, spec["noise"]), 4), round(_u(rng, spec["noise"]), 4)
        q1, q2 = int(round(_u(rng, spec["quality"]))), int(round(_u(rng, spec["quality"])))
        x = gaussian_blur(img, s1)
        x = cv2.resize(x, (round(w / f1), round(h / f1)), interpolation=cv2.INTER_AREA)
        x = add_gaussian_noise(x, n1, rng)
        x = jpeg(x, q1)
        x = gaussian_blur(x, s2)
        x = cv2.resize(x, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
        x = add_gaussian_noise(x, n2, rng)
        out = jpeg(x, q2)
        p.update(chain="[blur>resize>noise>jpeg]x2", factor=f, first_factor=f1, sigma1=s1, sigma2=s2,
                 noise1=n1, noise2=n2, quality1=q1, quality2=q2)
    else:
        raise KeyError(profile)
    p["out_h"], p["out_w"] = int(out.shape[0]), int(out.shape[1])
    return out, p


def params_json(p: dict) -> str:
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def all_cells() -> list[tuple[str, str]]:
    return [(prof, sev) for prof, sevs in PROFILES.items() for sev in sevs]
