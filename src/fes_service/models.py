"""Model factories for the isolated worker (run INSIDE the child process only).

`a4_factory` loads the pinned local A4 artefact (SwinIR-L real-SR ×4 GAN, weights SHA-256 verified on load by
fes_bench.adapters.load_net) through the unchanged Phase 3/4 adapter. No network access is used: the weights come
from models/M24_swinir (mounted), the architecture from the vendored third_party source (SHA-256 checked).
`nafnet_factory` (R5-5) loads NAFNet-GoPro width-64 (megvii-research/NAFNet @2b4af71, MIT; asset A06,
models/M28_nafnet/NAFNet-GoPro-width64.pth, SHA-256 329d3ab4…196f verified while staging, strict key match, 67.9 M
params) through the same registry (fes_bench.adapters.load_net "M28-GoPro"). Same-resolution deblurring: the output
has the input chip's size. The fake_* factories exist only for tests of the failure paths.
"""

from __future__ import annotations

import os
import time

import numpy as np


def a4_factory():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from fes_bench.adapters import build
    t0 = time.perf_counter()
    ad = build("A4")
    ad.load()
    rep = ad.load_reports[ad.key]
    info = {"model": "A4", "label": ad.label, "weights": rep["weights"], "weights_sha256": rep["sha256"],
            "strict": rep["strict"], "load_s": round(time.perf_counter() - t0, 2), "device": "cpu"}
    return ad.enhance, info


NAFNET_KEY = "M28-GoPro"


def load_nafnet(key: str = NAFNET_KEY, weights: str | None = None, sha256: str | None = None):
    """→ (net, load report). `weights` / `sha256` override the registry entry ONLY for tests of the failure paths
    (missing file, wrong hash); the service always uses the pinned registry entry. Never downloads anything."""
    from fes_bench import adapters as AD
    if weights is not None or sha256 is not None:
        asset, rel, sha = AD.WEIGHTS[key]
        saved = AD.WEIGHTS[key]
        AD.WEIGHTS[key] = (asset, weights or rel, sha256 or sha)
        try:
            return _checked_load(AD, key)
        finally:
            AD.WEIGHTS[key] = saved
    return _checked_load(AD, key)


def _checked_load(AD, key):
    from fes_data.common import repo_path
    rel = AD.WEIGHTS[key][1]
    if not repo_path(rel).exists():
        raise FileNotFoundError(f"NAFNet-GoPro checkpoint missing: {rel}")
    net, rep = AD.load_net(key)                           # SHA-256 + strict state-dict match, or it raises
    if rep["unexpected_keys"]:
        raise RuntimeError(f"{key}: unexpected checkpoint keys {rep['unexpected_keys'][:5]}")
    return net, rep


def nafnet_factory():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from fes_bench.adapters import bgr_u8_to_rgb_t, rgb_t_to_bgr_u8
    t0 = time.perf_counter()
    net, rep = load_nafnet()

    def enhance(img, ctx):
        with torch.inference_mode():
            return rgb_t_to_bgr_u8(net(bgr_u8_to_rgb_t(img)).clamp(0, 1)), {"model": "NAFNET_GOPRO"}
    info = {"model": "NAFNET_GOPRO", "label": "NAFNet-GoPro width-64 (deblur)", "asset_id": rep["asset_id"],
            "weights": rep["weights"], "weights_sha256": rep["sha256"], "strict": rep["strict"],
            "n_params": rep["n_model_params"], "load_s": round(time.perf_counter() - t0, 2), "device": "cpu",
            "pid": os.getpid()}
    return enhance, info


# ------------------------------------------------------------------ test doubles (never used by the service config)

def _to_e(img):
    import cv2
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)


def fake_ok_factory():
    """Returns the input face resized to the E-frame (passes the output guard like a real model would)."""
    return (lambda img, ctx: (_to_e(img), {"fake": True})), {"model": "FAKE_OK"}


def fake_raise_factory():
    def fn(img, ctx):
        raise RuntimeError("injected model failure")
    return fn, {"model": "FAKE_RAISE"}


def fake_hang_factory():
    def fn(img, ctx):
        time.sleep(3600)
    return fn, {"model": "FAKE_HANG"}


def fake_crash_factory():
    def fn(img, ctx):
        import signal
        os.kill(os.getpid(), signal.SIGKILL)       # what the kernel OOM killer does
    return fn, {"model": "FAKE_CRASH"}


def fake_bad_output_factory():
    return (lambda img, ctx: (np.zeros((224, 224, 3), np.uint8), {})), {"model": "FAKE_BAD_OUTPUT"}


def fake_load_fail_factory():
    raise RuntimeError("injected load failure")


# NAFNet-route doubles: same-size outputs (the NAFNet contract)

def fake_nafnet_ok_factory():
    """Mild unsharp mask of the input chip: same size, same colours (passes the artifact guard)."""
    import cv2

    def fn(img, ctx):
        b = cv2.GaussianBlur(img, (0, 0), 1.0)
        return cv2.addWeighted(img, 1.5, b, -0.5, 0), {"fake": True}
    return fn, {"model": "FAKE_NAFNET_OK", "pid": os.getpid()}


def rainbow(img: np.ndarray, amp: float = 60.0, period: float = 3.0) -> np.ndarray:
    """Synthetic NAFNet failure mode (R5-ARCHIVE): high-frequency rainbow streaks over the face."""
    h, w = img.shape[:2]
    x = np.arange(w)[None, :] + 0.5 * np.arange(h)[:, None]
    ph = 2 * np.pi * x / period
    streak = np.stack([np.sin(ph), np.sin(ph + 2.1), np.sin(ph + 4.2)], -1) * amp
    return np.clip(img.astype(np.float64) + streak, 0, 255).astype(np.uint8)


def fake_nafnet_rainbow_factory():
    return (lambda img, ctx: (rainbow(img), {})), {"model": "FAKE_NAFNET_RAINBOW"}


def fake_nafnet_bad_shape_factory():
    return (lambda img, ctx: (np.zeros((224, 224, 3), np.uint8), {})), {"model": "FAKE_NAFNET_BAD_SHAPE"}
