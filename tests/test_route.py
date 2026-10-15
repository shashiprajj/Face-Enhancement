"""The sealed NAFNet route (fes_service.nafnet_route): config seals, the input gate rules, the output guards, the R6
stability tracer and consensus. Pure functions; the image tests use the DEV blur fixture."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from fes_service import gate as GT
from fes_service import models as MD
from fes_service import nafnet_route as NR
from helpers import chroma, sharpen

# a feature set the gate accepts (strong, low-noise blur on a well-sized, well-lit frontal face)
GOOD = {"det_status": "OK", "det_score": 0.95, "yaw_proxy": 0.0, "iod_px": 80.0, "luma_mean": 120.0,
        "rms_contrast": 50.0, "blockiness": 1.0, "pixelation": 0.1, "hf_peak_frac": 0.05, "hf_energy": 0.05,
        "noise_sigma": 0.5, "lapd1_c": 0.004}


@pytest.fixture(scope="module")
def cfg():
    return NR.load_config()


def test_config_is_sealed_and_versioned(cfg):
    assert cfg["sha256"] == GT.gate_hash(cfg) and cfg["route_version"] == NR.R6_ROUTE_VERSION
    assert cfg["stability"]["version"] == NR.STABILITY_VERSION and cfg["stability"]["run_dot_c_max"] == pytest.approx(
        20.2863, abs=1e-4)
    assert cfg["thresholds"]["blur_max"] == cfg["thresholds"]["clean_lapd1_lo"] == 0.008   # every blurred face


def test_config_tamper_is_detected(cfg, tmp_path, monkeypatch):
    bad = {**cfg, "thresholds": {**cfg["thresholds"], "blur_max": 0.02}}
    p = tmp_path / "r6.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(NR, "repo_path", lambda _: p)
    with pytest.raises(AssertionError, match="modified after freeze"):
        NR.load_config()


def test_good_features_are_eligible(cfg):
    d = NR.decide(GOOD, cfg)
    assert d.eligible and d.reason == NR.ELIGIBLE and d.decision == GT.ENH


@pytest.mark.parametrize("over, reason", [
    ({"det_status": "NO_FACE_DETECTED"}, NR.R_NO_FACE), ({"det_score": 0.5}, NR.R_LOW_DET),
    ({"yaw_proxy": 0.6}, NR.R_ALIGN), ({"iod_px": 5.0}, NR.R_SIZE), ({"luma_mean": 5.0}, NR.R_PHOTO),
    ({"blockiness": 50.0}, NR.R_COMP), ({"pixelation": 0.99}, NR.R_PIX),
    ({"hf_peak_frac": 0.99, "hf_energy": 9.0}, NR.R_HALF), ({"noise_sigma": 50.0}, NR.R_NOISE),
    ({"lapd1_c": 0.5}, NR.R_NOT_BLUR)])
def test_each_rule_rejects(cfg, over, reason):
    d = NR.decide({**GOOD, **over}, cfg)
    assert not d.eligible and d.reason == reason


def test_missing_feature_never_enhances(cfg):
    for k in ("iod_px", "lapd1_c", "noise_sigma", "pixelation"):
        d = NR.decide({**GOOD, k: float("nan")}, cfg)
        assert not d.eligible and d.reason.startswith(NR.R_MISSING) and d.decision == GT.UNSAFE


def test_gate_on_a_real_blurred_chip(cfg, dev, detector):
    from fes_service.quality import rw_features
    f = {**rw_features(dev["blur"], detector), **NR.texture_features(dev["blur"])}
    assert NR.decide(f, cfg).eligible
    sharp = cv2.resize(cv2.resize(dev["blur"], (32, 32)), dev["blur"].shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    assert not NR.decide({**rw_features(sharp, detector), **NR.texture_features(sharp)}, cfg).eligible


# ------------------------------------------------------------------ output guards

def _guard(cfg, inp, out, gray=False, face=True):
    return NR.artifact_guard(inp, out, cfg, lambda _: face, gray)


def test_normal_output_passes(cfg, dev):
    assert _guard(cfg, dev["blur"], sharpen(dev["blur"])[0]).passed


def test_rainbow_output_fails(cfg, dev):
    g = _guard(cfg, dev["blur"], MD.rainbow(dev["blur"]))
    assert not g.passed and {NR.A_CHROMA, NR.A_HF} & set(g.reasons)


@pytest.mark.parametrize("make, why", [(lambda x: None, "not an array"), (lambda x: x.astype(np.float32), "dtype"),
                                       (lambda x: x[:50], "shape"), (lambda x: np.full_like(x, 7), "near-constant")])
def test_invalid_output_fails(cfg, dev, make, why):
    g = _guard(cfg, dev["blur"], make(dev["blur"]))
    assert not g.passed and g.reasons == [NR.A_NUMERIC] and why in g.metrics["numeric"]


def test_channel_corruption_and_lost_face_fail(cfg, dev):
    out = sharpen(dev["blur"])[0].copy()
    out[..., 0] = 0
    assert NR.A_NUMERIC in _guard(cfg, dev["blur"], out).reasons
    assert NR.A_FACE in _guard(cfg, dev["blur"], sharpen(dev["blur"])[0], face=False).reasons


def test_grayscale_colour_invention_fails(cfg, dev):
    g = cv2.cvtColor(cv2.cvtColor(dev["blur"], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    tinted = g.copy()
    tinted[..., 2] = np.clip(tinted[..., 2].astype(int) + 40, 0, 255).astype(np.uint8)
    assert NR.A_GRAY in _guard(cfg, g, tinted, gray=True).reasons


def test_local_guard_flags_a_dotted_patch(cfg, dev):
    x = dev["blur"]
    assert NR.local_artifact_guard(x, sharpen(x)[0], cfg, False).passed
    bad = sharpen(x)[0].copy()
    bad[40:56:2, 40:56:2] = (255, 0, 255)
    bad[41:57:2, 41:57:2] = (0, 255, 0)
    g = NR.local_artifact_guard(x, bad, cfg, False)
    assert not g.passed and g.reasons[0].startswith(NR.A_LOCAL)


# ------------------------------------------------------------------ R6 stability tracer and consensus

def test_tracer_equals_local_formula(dev):
    x = dev["blur"]
    for out in (sharpen(x)[0], MD.rainbow(x, amp=40, period=2.0), MD._fake_unstable_core(cv2.flip(x, 1), {})[0]):
        assert NR.dot_c_max(x, out) == pytest.approx(NR.local_features(x, out)["l_dot_c_max"], rel=1e-5, abs=1e-4)


def test_tracer_ignores_masked_area(dev):
    x = dev["blur"]
    y = MD.rainbow(x, amp=80, period=2.0)
    mask = np.ones(x.shape[:2], bool)
    mask[40:60, 40:60] = False
    y2 = sharpen(x)[0]
    y2[mask] = y[mask]                                             # rainbow only where masked out
    assert NR.dot_c_max(x, y2, ~mask) < NR.dot_c_max(x, y2) / 5


def test_consensus_keeps_input_colour_and_ignores_one_outlier(dev):
    x = dev["blur"]
    good, bad = sharpen(x)[0], MD.rainbow(x, amp=120, period=2.0)
    out = NR.consensus_luma(x, [good, good, good, bad])
    assert out.shape == x.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - NR.consensus_luma(x, [good] * 4).astype(int)).max() <= 1
    assert np.abs(chroma(out).astype(int) - chroma(x).astype(int)).max() <= 2


@pytest.mark.parametrize("vals,ok", [([1, 2, 3, 4], True), ([1, 2, 3, 20.28], True), ([1, 500, 3, 4], False),
                                     ([30, 30, 30, 30], False), ([1, 2, 3], False), ([1, float("nan"), 3, 4], False),
                                     (None, False), (["x", 1, 2, 3], False)])
def test_stability_rule(cfg, vals, ok):
    r = NR.stability_check(vals, cfg)
    assert r.passed is ok
    if not ok:
        assert r.reasons and r.reasons[0].startswith((NR.A_UNSTABLE, NR.A_UNSTABLE_ERROR))


def test_consensus_wrapper_flags_orientation_dependent_explosion(dev):
    x = dev["blur"]
    o1, m1 = MD.fake_nafnet_r6_stable_factory()[0](x, {})
    o2, m2 = MD.fake_nafnet_r6_unstable_factory()[0](x, {})
    assert len(m1["runs_dot_c"]) == len(m2["runs_dot_c"]) == 4
    assert max(m1["runs_dot_c"]) < 20 and max(m2["runs_dot_c"]) > 100
    assert 1 <= sum(v > 100 for v in m2["runs_dot_c"]) <= 3            # some mirror runs explode, not all
    for o in (o1, o2):
        assert o.shape == x.shape and np.abs(chroma(o).astype(int) - chroma(x).astype(int)).max() <= 2


def test_reflect_pad16():
    img = np.arange(20 * 33 * 3, dtype=np.uint8).reshape(20, 33, 3)
    x, h, w = MD.reflect_pad16(img)
    assert (h, w) == (20, 33) and x.shape == (32, 48, 3) and np.array_equal(x[:20, :33], img)
    same, *_ = MD.reflect_pad16(np.zeros((32, 16, 3), np.uint8))
    assert same.shape == (32, 16, 3)
