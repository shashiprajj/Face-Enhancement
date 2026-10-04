"""Model factories for the isolated worker (run INSIDE the child process only).

`a4_factory` loads the pinned local A4 artefact (SwinIR-L real-SR ×4 GAN, weights SHA-256 verified on load by
fes_bench.adapters.load_net) through the unchanged Phase 3/4 adapter. No network access is used: the weights come
from models/M24_swinir (mounted), the architecture from the vendored third_party source (SHA-256 checked).
The fake_* factories exist only for tests of the failure paths.
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
