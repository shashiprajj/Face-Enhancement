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


def reflect_pad16(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Reflect-pad H and W (bottom/right) up to the next multiple of 16 → (padded, h, w)."""
    import cv2
    h, w = img.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    if not (ph or pw):
        return img, h, w
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101), h, w


def nafnet_factory_pad16():
    """R5-8 research candidate (profile r5_8 only): NAFNet zero-pads its input to a multiple of 16
    (NAFNet_arch.check_image_size); on the service's re-framed chips that black border halved the gain (R5-7 DEV:
    ΔSFace +0.056 → +0.113). Reflect-pad first, crop back after; inputs already /16 give bit-identical outputs."""
    enhance0, info = nafnet_factory()

    def enhance(img, ctx):
        x, h, w = reflect_pad16(img)
        out, meta = enhance0(x, ctx)
        return np.ascontiguousarray(out[:h, :w]), {**meta, "pad16": [x.shape[0] - h, x.shape[1] - w]}
    return enhance, {**info, "variant": "reflect_pad16"}


def r6_consensus(enhance1):
    """R6 research candidate core: wrap a single-run enhance(img, ctx) into the 4-run consensus enhance. Runs `enhance1`
    on the chip and its h / v / hv mirrors (each mirrored back), computes each run's colour-dottedness tracer vs the
    input, returns (median of the runs with the input's chrominance, {"runs_dot_c": [...]}) via
    nafnet_route.consensus_luma. The accept / reject decision is NOT taken here: the pipeline applies
    nafnet_route.stability_check to meta["runs_dot_c"]. ctx["oob_mask"] (full_image chips that extend past the image) =
    out-of-image pixels, set to 0 in every run and excluded from the tracer, exactly as in the R6-0 study."""
    import cv2

    from fes_service import nafnet_route as NR

    def enhance(img, ctx):
        mask = (ctx or {}).get("oob_mask")
        ref, valid = img, None
        if mask is not None:
            ref, valid = img.copy(), ~mask
            ref[mask] = 0
        runs, dots = [], []
        for fl in NR.FLIPS:
            x = img if fl is None else np.ascontiguousarray(cv2.flip(img, fl))
            y = enhance1(x, ctx)[0]
            y = y if fl is None else cv2.flip(y, fl)
            if mask is not None:
                y = y.copy()
                y[mask] = 0
            runs.append(y)
            dots.append(round(NR.dot_c_max(ref, y, valid), 4))
        return NR.consensus_luma(img, runs), {"model": NR.MODEL, "runs_dot_c": dots, "variant": "r6_consensus4"}
    return enhance


def nafnet_factory_r6():
    """R6 research candidate (profile r6 only): r6_consensus around the reflect-pad16 NAFNet (4 runs per request)."""
    enhance1, info = nafnet_factory_pad16()
    return r6_consensus(enhance1), {**info, "variant": "r6_consensus4"}


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


def _fake_unstable_core(img, ctx):
    """Deterministic test double for R6: a mild unsharp mask, plus a saturated dotted colour patch whenever the image's
    top-left pixel is darker than its bottom-right one — so it depends on orientation and appears in SOME mirror runs
    only (the NAFNet instability pattern)."""
    import cv2
    b = cv2.GaussianBlur(img, (0, 0), 1.0)
    out = cv2.addWeighted(img, 1.5, b, -0.5, 0)
    if int(img[0, 0].sum()) < int(img[-1, -1].sum()):
        out = out.copy()
        h, w = out.shape[:2]
        patch = out[h // 4:h // 4 + 16, w // 4:w // 4 + 16]
        patch[::2, ::2] = (255, 0, 255)
        patch[1::2, 1::2] = (0, 255, 0)
    return out, {}


def fake_nafnet_r6_stable_factory():
    """R6 consensus around the mild unsharp mask (every run stable)."""
    def core(img, ctx):
        import cv2
        b = cv2.GaussianBlur(img, (0, 0), 1.0)
        return cv2.addWeighted(img, 1.5, b, -0.5, 0), {}
    return r6_consensus(core), {"model": "FAKE_NAFNET_R6_STABLE", "pid": os.getpid()}


def fake_nafnet_r6_unstable_factory():
    """R6 consensus around _fake_unstable_core (some runs explode, depending on the image orientation)."""
    return r6_consensus(_fake_unstable_core), {"model": "FAKE_NAFNET_R6_UNSTABLE", "pid": os.getpid()}
