"""Model factories for the isolated workers (run INSIDE the child process only).

Profile r6 runs NAFNet-GoPro width-64 (megvii-research/NAFNet @2b4af71, MIT; asset A06); profile r6_ft runs the same
architecture fine-tuned on synthetic FACE blur (R6-FT: FairFace CC BY 4.0, EMA checkpoint of step 22,000, Kaggle run
of 2026-10-09). Both run on ONNX Runtime (CPU) from the ONNX exports of those weights, wrapped in reflect-pad16 and the
4-run stability consensus (r6_consensus). Each factory returns (enhance(img, ctx) -> (out, meta), info dict). No
network access, no PyTorch. The fake_* factories exist only for tests of the failure paths.
"""

from __future__ import annotations

import os
import time

import numpy as np

FACE_FT_SHA256 = "6e8bbf2331655bd2f5867bcd41b0c3e17f24f94c2a64eaa8b9f2d1f85a76c7fe"   # source .pth of the R6-FT export
FACE_FT_MODEL = "NAFNET_FACE_FT"


def reflect_pad16(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Reflect-pad H and W (bottom/right) up to the next multiple of 16 → (padded, h, w)."""
    import cv2
    h, w = img.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    if not (ph or pw):
        return img, h, w
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101), h, w


def _pad16(enhance0):
    """NAFNet zero-pads its input to a multiple of 16; on the service's re-framed chips that black border halved the
    gain (R5-7 DEV: ΔSFace +0.056 → +0.113). Reflect-pad first, crop back after; inputs already /16 are unchanged."""
    def enhance(img, ctx):
        x, h, w = reflect_pad16(img)
        out, meta = enhance0(x, ctx)
        return np.ascontiguousarray(out[:h, :w]), {**meta, "pad16": [x.shape[0] - h, x.shape[1] - w]}
    return enhance


def r6_consensus(enhance1):
    """Wrap a single-run enhance(img, ctx) into the 4-run consensus enhance (R6-0). Runs `enhance1` on the chip and its
    h / v / hv mirrors (each mirrored back), computes each run's colour-dottedness tracer vs the input, returns (median
    of the runs with the input's chrominance, {"runs_dot_c": [...]}) via nafnet_route.consensus_luma. The accept /
    reject decision is NOT taken here: the pipeline applies nafnet_route.stability_check to meta["runs_dot_c"].
    ctx["oob_mask"] (full_image chips that extend past the image) = out-of-image pixels, set to 0 in every run and
    excluded from the tracer, exactly as in the R6-0 study."""
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


# ------------------------------------------------------------------ ONNX Runtime backend
# The ONNX exports (scripts/r6_onnx/export_nafnet.py, kept in git history; fp32, opset 17, dynamic H / W, valid for
# sides ≤ `fes_max_side` = 384 because the export replaces NAFNetLocal's TLC average by the global mean it equals in
# that range). The files are git-ignored like every weight file; their SHA-256 and the SHA-256 of the PyTorch weights
# they were exported from (stored in the ONNX metadata) are verified on load.
ONNX_BACKEND = "onnxruntime"
ONNX_MODELS = {     # profile → (file, SHA-256 of the file, SHA-256 of the source .pth, model name, label)
    "r6": ("models/M28_nafnet/onnx/nafnet_gopro.onnx",
           "75edad38c239d64823a7ce558712695eff8ca0760e1385d7cd2527cc1276d6d1",
           "329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f", "NAFNET_GOPRO",
           "NAFNet-GoPro width-64 (deblur), ONNX"),
    "r6_ft": ("models/M28_nafnet/onnx/nafnet_face_ft.onnx",
              "f63d2d40a7f8106c2c5c96cfb1be40fef9e912870e76d6779b08f995ea788b67", FACE_FT_SHA256, FACE_FT_MODEL,
              "NAFNet width-64, face-blur fine-tune of NAFNet-GoPro (R6-FT), ONNX"),
}


def onnx_enhance(sess, max_side: int, model: str):
    """enhance(img BGR uint8 HxWx3, ctx) → (BGR uint8 HxWx3, meta) around an ONNX Runtime session with one float32
    RGB NCHW [0, 1] input and one output of the same shape. Anything unexpected RAISES (→ the worker reports a model
    failure → the pipeline returns ORIGINAL): a wrong input, an output of another shape or dtype, a non-finite value."""
    import cv2
    name = sess.get_inputs()[0].name

    def enhance(img, ctx):
        if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
            raise ValueError("ONNX input must be a uint8 HxWx3 image")
        h, w = img.shape[:2]
        if max(h, w) > max_side or h % 16 or w % 16:
            raise ValueError(f"ONNX input {w}x{h} outside the exported range (multiples of 16, side <= {max_side})")
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        y = sess.run(None, {name: np.ascontiguousarray(x.transpose(2, 0, 1))[None]})[0]
        if not (isinstance(y, np.ndarray) and y.dtype == np.float32 and y.shape == (1, 3, h, w)):
            raise RuntimeError(f"ONNX output has shape {getattr(y, 'shape', None)} / dtype {getattr(y, 'dtype', None)}")
        if not np.isfinite(y).all():
            raise RuntimeError("ONNX output is not finite")
        rgb = (np.clip(y[0], 0.0, 1.0).transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), {"model": model}
    return enhance


def load_onnx(profile: str, path: str | None = None, sha256: str | None = None):
    """→ (session, info). Verifies the file's SHA-256 and the export metadata before any inference; `path` / `sha256`
    override the pinned entry for tests of the failure paths. Raises on a missing file, a missing onnxruntime, a hash
    or metadata mismatch."""
    import hashlib

    from fes_service import repo_path
    rel, sha, src_sha, model, label = ONNX_MODELS[profile]
    rel, sha = path or rel, sha256 or sha
    file = repo_path(rel)
    if not file.exists():
        raise FileNotFoundError(f"ONNX model missing: {rel}")
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(f"the service needs onnxruntime (requirements-onnx.txt): {e}") from e
    h = hashlib.sha256()
    with file.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != sha:
        raise RuntimeError(f"{rel}: SHA-256 {h.hexdigest()[:16]}… does not match the pinned {sha[:16]}…")
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("FES_NAFNET_THREADS", "4"))
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(file), so, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if meta.get("fes_source_sha256") != src_sha:
        raise RuntimeError(f"{rel}: exported from other weights ({str(meta.get('fes_source_sha256'))[:16]}…)")
    if meta.get("fes_profile") != profile or "fes_max_side" not in meta:
        raise RuntimeError(f"{rel}: export metadata does not match profile {profile!r}")
    info = {"model": model, "label": label, "asset_id": "R6-FT" if profile == "r6_ft" else "A06",
            "backend": ONNX_BACKEND, "backend_version": ort.__version__, "provider": sess.get_providers()[0],
            "weights": rel, "weights_sha256": sha, "source_weights_sha256": meta.get("fes_source_sha256"),
            "max_side": int(meta["fes_max_side"]), "threads": so.intra_op_num_threads, "device": "cpu",
            "pid": os.getpid()}
    return sess, info


def _nafnet_factory_r6_onnx(profile: str):
    t0 = time.perf_counter()
    sess, info = load_onnx(profile)
    core = r6_consensus(_pad16(onnx_enhance(sess, info["max_side"], info["model"])))

    def enhance(img, ctx):
        out, meta = core(img, ctx)
        return out, {**meta, "backend": ONNX_BACKEND}
    return enhance, {**info, "variant": "r6_consensus4", "load_s": round(time.perf_counter() - t0, 2)}


def nafnet_factory_r6_onnx():
    """Profile r6: NAFNet-GoPro on ONNX Runtime, reflect-pad16, 4-run consensus."""
    return _nafnet_factory_r6_onnx("r6")


def nafnet_factory_r6_ft_onnx():
    """Profile r6_ft: the face-blur fine-tune on ONNX Runtime, reflect-pad16, 4-run consensus."""
    return _nafnet_factory_r6_onnx("r6_ft")


# ------------------------------------------------------------------ test doubles (never used by the service config)

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


def fake_load_fail_factory():
    raise RuntimeError("injected load failure")


def rainbow(img: np.ndarray, amp: float = 60.0, period: float = 3.0) -> np.ndarray:
    """Synthetic NAFNet failure mode (R5-ARCHIVE): high-frequency rainbow streaks over the face."""
    h, w = img.shape[:2]
    x = np.arange(w)[None, :] + 0.5 * np.arange(h)[:, None]
    ph = 2 * np.pi * x / period
    streak = np.stack([np.sin(ph), np.sin(ph + 2.1), np.sin(ph + 4.2)], -1) * amp
    return np.clip(img.astype(np.float64) + streak, 0, 255).astype(np.uint8)


def _sharpen(img):
    import cv2
    b = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, b, -0.5, 0)


def _fake_unstable_core(img, ctx):
    """Deterministic test double for R6: a mild unsharp mask, plus a saturated dotted colour patch whenever the image's
    top-left pixel is darker than its bottom-right one — so it depends on orientation and appears in SOME mirror runs
    only (the NAFNet instability pattern)."""
    out = _sharpen(img)
    if int(img[0, 0].sum()) < int(img[-1, -1].sum()):
        out = out.copy()
        h, w = out.shape[:2]
        patch = out[h // 4:h // 4 + 16, w // 4:w // 4 + 16]
        patch[::2, ::2] = (255, 0, 255)
        patch[1::2, 1::2] = (0, 255, 0)
    return out, {}


def fake_nafnet_r6_stable_factory():
    """R6 consensus around the mild unsharp mask (every run stable)."""
    return r6_consensus(lambda img, ctx: (_sharpen(img), {})), {"model": "FAKE_NAFNET_R6_STABLE", "pid": os.getpid()}


def fake_nafnet_r6_unstable_factory():
    """R6 consensus around _fake_unstable_core (some runs explode, depending on the image orientation)."""
    return r6_consensus(_fake_unstable_core), {"model": "FAKE_NAFNET_R6_UNSTABLE", "pid": os.getpid()}


# ONNX doubles: the real ONNX wrapper chain (onnx_enhance → reflect-pad16 → 4-run consensus) around a fake session

class _FakeOrtSession:
    """Minimal onnxruntime.InferenceSession stand-in: `fn` maps the float32 NCHW input to the returned output."""

    def __init__(self, fn):
        self.fn = fn

    def get_inputs(self):
        return [type("I", (), {"name": "image"})()]

    def run(self, _outputs, feed):
        return [self.fn(feed["image"])]


def _fake_onnx_r6(fn, model):
    core = r6_consensus(_pad16(onnx_enhance(_FakeOrtSession(fn), 384, model)))

    def enhance(img, ctx):
        out, meta = core(img, ctx)
        return out, {**meta, "backend": ONNX_BACKEND}
    return enhance, {"model": model, "backend": ONNX_BACKEND, "pid": os.getpid()}


def fake_onnx_identity_factory():
    return _fake_onnx_r6(lambda x: x.copy(), "FAKE_ONNX_IDENTITY")


def fake_onnx_nan_factory():
    return _fake_onnx_r6(lambda x: np.full_like(x, np.nan), "FAKE_ONNX_NAN")


def fake_onnx_bad_shape_factory():
    return _fake_onnx_r6(lambda x: x[:, :, ::2, ::2].copy(), "FAKE_ONNX_BAD_SHAPE")
