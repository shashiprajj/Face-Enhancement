"""R5-3 real-world gate: decision contract, grain-robust blur feature, greyscale preservation, safe fallbacks.

Unit tests use a toy threshold set (not evidence). Pipeline tests use the sealed configs/phase5/r5_rwgate.json and
Phase 4 dev chips (skipped if the local datasets are absent). No real model is run (spy / fake isolated models).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from fes_bench import gate as GT
from fes_bench import routing as RT
from fes_data import degrade as D
from fes_data.common import read_csv_gz, repo_path
from fes_service import pipeline as SP
from fes_service import rwgate as RW
from fes_service.worker import IsolatedModel

TOY = {"thresholds": {"iod_min_recognizable": 16.0, "lapd1_floor": 0.001, "luma_mean_lo": 47.0, "rms_contrast_lo": 25.0,
                      "blockiness_hi": 1.52, "noise_min": 9.8, "blur_max": 0.004, "iod_min_a4": 50.0,
                      "clean_lapd1_lo": 0.0056, "clean_noise_hi": 1.08, "clean_iod_lo": 71.0}}
GOOD = {"det_status": "OK", "det_score": 0.93, "yaw_proxy": 0.0, "iod_px": 80.0, "lapd1_c": 0.015, "noise_sigma": 1.0,
        "luma_mean": 120.0, "rms_contrast": 50.0, "blockiness": 1.0}


@pytest.mark.parametrize("upd,decision,reason,route", [
    ({}, GT.DNE, "acceptable", "B0"),
    ({"det_status": "NO_FACE_DETECTED"}, GT.UNSAFE, "no_valid_face", "B0"),
    ({"det_score": 0.6, "lapd1_c": 0.002}, GT.UNSAFE, "low_detection_confidence", "B0"),
    ({"yaw_proxy": 0.5, "noise_sigma": 20.0}, GT.UNSAFE, "alignment_unreliable", "B0"),
    ({"iod_px": 12.0, "lapd1_c": 0.002}, GT.UNSAFE, "insufficient_face_size", "B0"),         # tiny stays protected
    ({"iod_px": math.nan}, GT.UNSAFE, "insufficient_face_size", "B0"),
    ({"lapd1_c": 0.0005}, GT.UNSAFE, "beyond_evidence_extreme_blur", "B0"),                  # unrecognizable
    ({"lapd1_c": 0.002}, GT.ENH, "strong_blur", "A4"),                                       # degraded + recognizable
    ({"noise_sigma": 15.0}, GT.ENH, "strong_noise", "A4"),
    ({"lapd1_c": 0.002, "noise_sigma": 3.0}, GT.ENH, "strong_blur", "A4"),                   # real grain does not block
    ({"lapd1_c": 0.002, "iod_px": 30.0}, GT.DNE, "degraded_below_a4_size:strong_blur", "B0"),
    ({"lapd1_c": 0.002, "luma_mean": 30.0}, GT.DNE, "photometric_no_a4_evidence", "B0"),
    ({"noise_sigma": 15.0, "blockiness": 2.0}, GT.DNE, "compression_no_a4_evidence", "B0"),
    ({"lapd1_c": 0.005}, GT.DNE, "degraded_no_a4_evidence:blur", "B0"),                     # moderate blur: no A4 evidence
    ({"iod_px": 60.0, "noise_sigma": 3.0}, GT.DNE, "degraded_no_a4_evidence:noise+lowres", "B0"),
    ({"lapd1_c": math.nan, "noise_sigma": math.nan}, GT.DNE, "acceptable", "B0")])          # missing evidence never enhances
def test_decision_contract(upd, decision, reason, route):
    d, why, _, r = RW.decide({**GOOD, **upd}, TOY)
    assert (d, r) == (decision, route) and why.startswith(reason)


def test_only_a4_or_original_can_be_routed():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        f = {**GOOD, "iod_px": float(rng.uniform(5, 120)), "lapd1_c": float(rng.uniform(0, 0.03)),
             "noise_sigma": float(rng.uniform(0, 40)), "luma_mean": float(rng.uniform(20, 200)),
             "blockiness": float(rng.uniform(0.8, 3)), "det_score": float(rng.uniform(0.7, 1))}
        d, _, _, r = RW.decide(f, TOY)
        assert r in ("A4", RT.ORIGINAL) and (r == "A4") == (d == GT.ENH)


def test_lapd1_is_grain_robust_and_orders_blur(ref_faces):
    face = ref_faces[0]
    rng = np.random.default_rng(1)
    grain = lambda im, s: np.clip(im + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)  # noqa: E731
    blurred = D.gaussian_blur(face, 4.0)
    assert RW.lapd1(face) > RW.lapd1(D.gaussian_blur(face, 1.0)) > RW.lapd1(D.gaussian_blur(face, 2.0)) > RW.lapd1(blurred)
    assert abs(RW.lapd1(grain(blurred, 4)) / RW.lapd1(blurred) - 1) < 0.15          # σ=4 grain: < 15 % change
    from fes_bench.quality import to_c_frame
    y = lambda im: cv2.cvtColor(to_c_frame(im), cv2.COLOR_BGR2GRAY).astype(np.float64)  # noqa: E731
    old = lambda im: cv2.Laplacian(y(im), cv2.CV_64F).var() / y(im).var()  # noqa: E731
    assert old(grain(blurred, 4)) / old(blurred) > 2.0                              # the old feature is not


def test_is_gray():
    g = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    assert RW.is_gray(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    jpg = cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 1)
    assert RW.is_gray(jpg)
    c = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).copy()
    c[..., 2] = np.clip(c[..., 2].astype(int) + 40, 0, 255)
    assert not RW.is_gray(c)


def test_sealed_config_loads_and_is_conservative():
    cfg = RW.load_config()
    t = cfg["thresholds"]
    assert t["iod_min_recognizable"] < t["iod_min_a4"] and 0 < t["lapd1_floor"] < t["blur_max"]
    assert t["noise_min"] == pytest.approx(9.837129)
    bad = dict(cfg, thresholds={**t, "blur_max": 1.0})
    with pytest.raises(AssertionError):
        import json
        p = repo_path("experiments/phase5_rapid/_tampered_rwgate.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(bad), encoding="utf-8")
        try:
            RW.load_config("experiments/phase5_rapid/_tampered_rwgate.json")
        finally:
            p.unlink()


# ------------------------------------------------------------------ pipeline (rw gate)

class Spy:
    """Fake A4 returning a COLOUR-tinted face (to prove greyscale preservation)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img, ctx):
        self.calls += 1
        out = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.int16)
        out[..., 2] += 30
        return np.clip(out, 0, 255).astype(np.uint8), {}


@pytest.fixture(scope="module")
def blur_chip():
    pop = repo_path("data/manifests/phase4/population_p4dev.csv.gz")
    if not pop.exists():
        pytest.skip("phase 4 manifests missing")
    t = RW.load_config()["thresholds"]
    for r in sorted(read_csv_gz(pop), key=lambda r: r["pair_id"]):
        if r["cell"] == "gaussblur-extreme" and repo_path(r["degraded_path"]).exists():
            img = cv2.imread(str(repo_path(r["degraded_path"])), cv2.IMREAD_COLOR)
            # well inside the admitted blur band, so σ=3 grain cannot move it across a threshold by chance
            if 1.3 * t["lapd1_floor"] <= RW.lapd1(img) <= 0.8 * t["blur_max"]:
                return img
    pytest.skip("degraded images missing")


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_good_image_not_enhanced(ref_faces):
    spy = Spy()
    r = SP.EnhancementPipeline(spy, timeout_s=10, gate="rw").process(png(ref_faces[0]), "face_chip")
    assert r.status == SP.ORIGINAL_S and r.decision == GT.DNE and spy.calls == 0


def test_recognizable_blurred_face_reaches_a4_and_grain_does_not_block(blur_chip):
    spy = Spy()
    pipe = SP.EnhancementPipeline(spy, timeout_s=10, gate="rw")
    r = pipe.process(png(blur_chip), "face_chip")
    assert r.route == "A4" and r.status == SP.ENHANCED and r.reason == "strong_blur"
    grainy = np.clip(blur_chip + np.random.default_rng(2).normal(0, 3, blur_chip.shape), 0, 255).astype(np.uint8)
    r2 = pipe.process(png(grainy), "face_chip")
    assert r2.route == "A4"                                       # frozen T6 would block this (noise σ > 0.28)
    assert SP.EnhancementPipeline(Spy(), timeout_s=10, gate="t6").process(png(grainy), "face_chip").route == "B0"


def test_grayscale_in_grayscale_out(blur_chip):
    gray = cv2.cvtColor(cv2.cvtColor(blur_chip, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    r = SP.EnhancementPipeline(Spy(), timeout_s=10, gate="rw").process(png(gray), "face_chip")
    assert r.status == SP.ENHANCED and "grayscale_preserved" in r.warnings
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    col = SP.EnhancementPipeline(Spy(), timeout_s=10, gate="rw").process(png(blur_chip), "face_chip")
    out = cv2.imdecode(np.frombuffer(col.image, np.uint8), cv2.IMREAD_COLOR)
    assert not np.array_equal(out[..., 0], out[..., 2])           # colour inputs are not forced to grey


def test_tiny_and_unrecognizable_faces_protected(ref_faces, blur_chip):
    spy = Spy()
    pipe = SP.EnhancementPipeline(spy, timeout_s=10, gate="rw")
    canvas = np.full((300, 300, 3), 128, np.uint8)
    canvas[140:160, 140:160] = cv2.resize(ref_faces[0], (20, 20), interpolation=cv2.INTER_AREA)   # IOD ≈ 7 px
    assert pipe.process(png(canvas), "full_image").route == "B0"
    smear = D.gaussian_blur(blur_chip, 12.0)                     # far beyond any measured blur
    r = pipe.process(png(smear), "face_chip")
    assert r.route == "B0" and r.decision in (GT.UNSAFE, GT.DNE)
    assert spy.calls == 0


def test_uncertain_returns_original():
    noise = np.random.default_rng(3).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    r = SP.EnhancementPipeline(Spy(), timeout_s=10, gate="rw").process(png(noise), "full_image")
    assert r.status == SP.ORIGINAL_S and r.escalate and r.decision == GT.UNSAFE


def test_a4_failure_falls_back_under_rw_gate(blur_chip):
    m = IsolatedModel("fes_service.models:fake_raise_factory", timeout_s=5)
    try:
        r = SP.EnhancementPipeline(m, timeout_s=5, gate="rw").process(png(blur_chip), "face_chip")
    finally:
        m.close()
    assert r.route == "A4" and r.used == "B0" and r.fallback == "MODEL_FAILURE" and r.status == SP.ORIGINAL_S
