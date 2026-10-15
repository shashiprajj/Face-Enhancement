"""Phase 3 — the degradation operators (pure numpy / OpenCV, deterministic for a given seed).

A variant is an ordered list of operations (`ops`) taken from degradation_matrix.json and applied left to right to the
clean reference. Images are float32 RGB in [0, 1] (sRGB-encoded) between operations and are quantised to 8 bit once,
at the end (or by a JPEG step). All pixel sizes are in pixels of the clean reference (long side 800).

Domain of each operator
  optical blurs (gaussian / motion / defocus / box / shake)  linear light: the lens and the exposure integrate photons,
                                                             so highlights spread the way they do in a real capture
  resize_down / resize_up                                    sRGB values, the way image software resamples
  exposure, white_balance, poisson_gaussian_noise            linear light (sensor side)
  gaussian_noise, contrast, saturation, jpeg                 sRGB values (after the camera's tone curve)
Filters use BORDER_REFLECT_101. Any random draw comes from numpy's PCG64 `default_rng(seed)`, where the seed is
common.stable_seed(global seed, sample_id, variant_id): a variant depends only on its original and its manifest row.
"""

from __future__ import annotations

import cv2
import numpy as np

INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC,
          "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}
BORDER = cv2.BORDER_REFLECT_101


def to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _filter_linear(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return to_srgb(cv2.filter2D(to_linear(img), -1, kernel.astype(np.float32), borderType=BORDER))


def _splat(points: np.ndarray, size: int) -> np.ndarray:
    """Kernel from sub-pixel points (x, y) relative to the kernel centre, bilinear weights, sum 1."""
    k = np.zeros((size, size), np.float64)
    c = (size - 1) / 2.0
    for x, y in points:
        fx, fy = x + c, y + c
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xi, yi = x0 + dx, y0 + dy
            if 0 <= xi < size and 0 <= yi < size:
                k[yi, xi] += (1 - abs(fx - xi)) * (1 - abs(fy - yi))
    return k / k.sum()


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Linear motion of `length` px along `angle_deg` (0 = horizontal, counter-clockwise), anti-aliased."""
    size = int(np.ceil(length)) | 1
    size += 2
    t = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, max(2, int(length * 8)))
    a = np.deg2rad(angle_deg)
    return _splat(np.stack([t * np.cos(a), -t * np.sin(a)], 1), size)


def disk_kernel(radius: float) -> np.ndarray:
    """Uniform disk (out-of-focus lens), anti-aliased by 8× supersampling."""
    size = (2 * int(np.ceil(radius)) + 1) | 1
    ss = 8
    c = (size * ss - 1) / 2.0
    y, x = np.mgrid[0:size * ss, 0:size * ss]
    m = ((x - c) ** 2 + (y - c) ** 2 <= (radius * ss) ** 2).astype(np.float64)
    k = m.reshape(size, ss, size, ss).mean((1, 3))
    return k / k.sum()


def shake_kernel(extent: int, rng: np.random.Generator) -> np.ndarray:
    """Camera-shake PSF: a smooth random trajectory (velocity random walk with inertia), scaled to fit `extent` px."""
    n = 400
    v = np.zeros(2)
    pos = np.zeros((n, 2))
    for i in range(1, n):
        v = 0.92 * v + rng.normal(0, 1, 2)
        pos[i] = pos[i - 1] + v
    pos -= (pos.max(0) + pos.min(0)) / 2.0
    pos *= ((extent - 1) / 2.0) / max(1e-9, np.abs(pos).max())
    return _splat(pos, int(extent) | 1)


# ------------------------------------------------------------------ operators: (img, params, rng, state) → img
def op_gaussian_blur(img, p, rng, st):
    k = int(2 * np.ceil(4 * p["sigma"]) + 1)
    return to_srgb(cv2.GaussianBlur(to_linear(img), (k, k), p["sigma"], borderType=BORDER))


def op_motion_blur(img, p, rng, st):
    return _filter_linear(img, motion_kernel(p["length"], p["angle"]))


def op_defocus_blur(img, p, rng, st):
    return _filter_linear(img, disk_kernel(p["radius"]))


def op_box_blur(img, p, rng, st):
    return to_srgb(cv2.blur(to_linear(img), (p["ksize"], p["ksize"]), borderType=BORDER))


def op_shake_blur(img, p, rng, st):
    return _filter_linear(img, shake_kernel(p["extent"], rng))


def op_resize_down(img, p, rng, st):
    h, w = img.shape[:2]
    st.setdefault("full_size", (w, h))
    size = (max(8, round(w / p["factor"])), max(8, round(h / p["factor"])))
    return np.clip(cv2.resize(img, size, interpolation=INTERP[p["interp"]]), 0, 1)


def op_resize_up(img, p, rng, st):
    return np.clip(cv2.resize(img, st["full_size"], interpolation=INTERP[p["interp"]]), 0, 1)


def op_gaussian_noise(img, p, rng, st):
    return np.clip(img + rng.normal(0, p["sigma"] / 255.0, img.shape).astype(np.float32), 0, 1)


def op_poisson_gaussian_noise(img, p, rng, st):
    """Shot noise (Poisson on `full_well` electrons at white) + read noise (Gaussian, fraction of white), linear light."""
    lin = to_linear(img).astype(np.float64)
    out = rng.poisson(lin * p["full_well"]) / p["full_well"] + rng.normal(0, p["read_sigma"], lin.shape)
    return to_srgb(out.astype(np.float32))


def op_exposure(img, p, rng, st):
    return to_srgb(to_linear(img) * p["gain"])


def op_white_balance(img, p, rng, st):
    return to_srgb(to_linear(img) * np.asarray(p["rgb_gain"], np.float32))


def op_contrast(img, p, rng, st):
    pivot = float(img.mean())
    return np.clip((img - pivot) * p["factor"] + pivot, 0, 1)


def op_saturation(img, p, rng, st):
    luma = (img @ np.asarray([0.299, 0.587, 0.114], np.float32))[..., None]
    return np.clip(luma + (img - luma) * p["factor"], 0, 1)


def op_jpeg(img, p, rng, st):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(quantise(img), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(p["quality"])])
    assert ok
    st["last_jpeg"] = buf.tobytes()
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


OPS = {"gaussian_blur": op_gaussian_blur, "motion_blur": op_motion_blur, "defocus_blur": op_defocus_blur,
       "box_blur": op_box_blur, "shake_blur": op_shake_blur, "resize_down": op_resize_down, "resize_up": op_resize_up,
       "gaussian_noise": op_gaussian_noise, "poisson_gaussian_noise": op_poisson_gaussian_noise,
       "exposure": op_exposure, "white_balance": op_white_balance, "contrast": op_contrast,
       "saturation": op_saturation, "jpeg": op_jpeg}


def quantise(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def apply(rgb_u8: np.ndarray, ops: list[dict], seed: int) -> tuple[bytes, str, np.ndarray]:
    """Clean RGB uint8 + ordered ops → (file bytes, extension, decoded RGB uint8 of exactly those bytes).

    When the last operation is `jpeg`, the file IS that JPEG stream (the service receives real JPEG data); otherwise
    the result is stored lossless as PNG."""
    rng = np.random.default_rng(seed)
    state: dict = {}
    img = rgb_u8.astype(np.float32) / 255.0
    for op in ops:
        img = OPS[op["op"]](img, op, rng, state)
    assert img.shape == rgb_u8.shape, f"variant changed the image size: {img.shape} vs {rgb_u8.shape}"
    out = quantise(img)
    if ops[-1]["op"] == "jpeg":
        return state["last_jpeg"], "jpg", out
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert ok
    return buf.tobytes(), "png", out
