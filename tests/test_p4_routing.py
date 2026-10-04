"""P4-T7 routing: buckets, table fitting, and runtime fallbacks under injected failures (fake adapters, not data)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from fes_bench import gate as GT
from fes_bench import routing as RT

T = {"iod_px_lo": 60.0, "noise_sigma_hi": 1.0, "lap_norm_c_lo": 0.05, "blockiness_hi": 1.6,
     "luma_mean_lo": 60.0, "rms_contrast_lo": 30.0}
BASE = {"det_status": "OK", "det_score": 0.95, "yaw_proxy": 0.0, "iod_px": 80.0, "noise_sigma": 0.5,
        "lap_norm_c": 0.1, "blockiness": 1.3, "luma_mean": 95.0, "rms_contrast": 44.0}


@pytest.mark.parametrize("upd,bucket", [
    ({}, "NONE"), ({"noise_sigma": 9.0}, "NOISE"), ({"iod_px": 20.0, "noise_sigma": 5.0}, "LOWRES"),
    ({"lap_norm_c": 0.01}, "BLUR"), ({"blockiness": 2.5}, "COMPRESS"), ({"luma_mean": 30.0}, "PHOTOMETRIC"),
    ({"rms_contrast": 10.0, "noise_sigma": 9.0}, "PHOTOMETRIC"), ({"lap_norm_c": 0.01, "blockiness": 2.5}, "MIXED"),
    ({"iod_px": math.nan, "noise_sigma": 9.0}, "NOISE")])
def test_buckets(upd, bucket):
    assert RT.bucket_of({**BASE, **upd}, T) == bucket


def test_bucket_thresholds_from_clean_quantiles():
    clean = [{**BASE, "noise_sigma": float(v)} for v in np.linspace(0.2, 0.6, 101)]
    t = RT.bucket_thresholds(clean)
    assert t["noise_sigma_hi"] == pytest.approx(0.592, abs=1e-3)


def test_fit_table_requires_positive_utility_and_support():
    b = ["NOISE"] * 12 + ["BLUR"] * 12 + ["LOWRES"] * 5
    v = {"A3": ["CONFIRMED_BENEFIT"] * 12 + ["CONFIRMED_HARM"] * 12 + ["CONFIRMED_BENEFIT"] * 5,
         "A1": ["CONFIRMED_BENEFIT"] * 12 + ["NEUTRAL"] * 12 + ["CONFIRMED_BENEFIT"] * 5}
    table, _ = RT.fit_table(b, v)
    assert table["NOISE"] == "A1"            # tie on utility → cheaper (A1 before A3 in COST_ORDER)
    assert table["BLUR"] == RT.ORIGINAL      # no positive utility
    assert table["LOWRES"] == RT.ORIGINAL    # < MIN_BUCKET_N pairs
    assert table["PHOTOMETRIC"] == RT.ORIGINAL


def _router(adapter, face_ok=True, timeout=0.5):
    rules = [GT.Rule((GT.Cond("noise_sigma", ">=", 2.0),))]
    return RT.Router(rules, T, {"NOISE": "A3"}, {"A3": adapter}, features_fn=lambda img: dict(img_feats),
                     face_check=lambda out: face_ok, timeout_s=timeout)


img_feats = {**BASE, "noise_sigma": 9.0}
IMG = np.full((56, 56, 3), 100, np.uint8)
ORIG = np.full((224, 224, 3), 100, np.uint8)
GOOD = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)


def test_success_and_original_paths():
    out, rec = _router(lambda i, c: (GOOD, {})).route(IMG, {}, ORIG)
    assert rec.used == "A3" and rec.fallback == "" and out is GOOD
    r = _router(lambda i, c: (GOOD, {}))
    global img_feats
    saved, img_feats = img_feats, {**BASE}                     # gate → DO_NOT_ENHANCE
    try:
        out, rec = r.route(IMG, {}, ORIG)
        assert rec.decision == GT.DNE and rec.used == RT.ORIGINAL and out is ORIG
        img_feats = {**BASE, "det_status": "NO_FACE_DETECTED"}
        out, rec = r.route(IMG, {}, ORIG)
        assert rec.decision == GT.UNSAFE and rec.escalate and out is ORIG
    finally:
        img_feats = saved


def _raise(e):
    def f(i, c):
        raise e
    return f


@pytest.mark.parametrize("adapter,fallback", [
    (_raise(RuntimeError("boom")), "MODEL_FAILURE"),
    (_raise(MemoryError()), "RESOURCE_FAILURE"),
    (_raise(RuntimeError("[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: not enough memory")), "RESOURCE_FAILURE"),
    (lambda i, c: (np.zeros((224, 224, 3), np.uint8), {}), "INVALID_OUTPUT"),          # constant image
    (lambda i, c: (GOOD[:100], {}), "INVALID_OUTPUT"),                                   # wrong shape
    (lambda i, c: (GOOD.astype(np.float32), {}), "INVALID_OUTPUT"),                      # wrong dtype
    (lambda i, c: (time.sleep(2.0), (GOOD, {}))[1], "TIMEOUT")])
def test_every_failure_falls_back_to_original(adapter, fallback):
    out, rec = _router(adapter).route(IMG, {}, ORIG)
    assert rec.fallback == fallback and rec.used == RT.ORIGINAL and out is ORIG


def test_no_face_on_output_is_invalid():
    out, rec = _router(lambda i, c: (GOOD, {}), face_ok=False).route(IMG, {}, ORIG)
    assert rec.fallback == "INVALID_OUTPUT" and "no face" in rec.error and out is ORIG


def test_router_recovers_after_timeout():
    calls = {"n": 0}

    def flaky(i, c):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(1.0)
        return GOOD, {}
    r = _router(flaky, timeout=0.3)
    assert r.route(IMG, {}, ORIG)[1].fallback == "TIMEOUT"
    assert r.route(IMG, {}, ORIG)[1].used == "A3"
