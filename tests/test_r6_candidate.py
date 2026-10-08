"""R6 research candidate (opt-in profile "r6"): 4-run stability check + median / luma-only consensus output.

Invariants: requests without `profile` (and r5_6 / r5_8 / r5_10) are unchanged; r6 is reachable only when the service
is built with it; any unstable mirror run -> ORIGINAL (fallback STABILITY_GUARD); an accepted output never changes the
input's colour (chrominance); the stability tracer is the R5-8 l_dot_c_max formula. DEV images only.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from fes_data.common import read_csv_gz, repo_path
from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP

POP = repo_path("data/manifests/phase4/population_p4dev.csv.gz")
BLUR_ID = "ff-train-13457.ref-v1.gaussblur-extreme"
REAL = pytest.mark.skipif(os.environ.get("FES_RUN_NAFNET") != "1", reason="real NAFNet: set FES_RUN_NAFNET=1")


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def sharpen(img, ctx=None):
    b = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, b, -0.5, 0), {}


def chroma(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 1:]


class Model:
    """In-process stand-in for the isolated worker: r6_consensus around `core`, records every ctx."""

    def __init__(self, core):
        self.fn, self.ctx = MD.r6_consensus(core), []

    def __call__(self, img, ctx):
        self.ctx.append(dict(ctx))
        out, meta = self.fn(img, ctx)
        return out, {**meta, "infer_s": 0.01}

    def alive(self):
        return True


class Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, img, ctx):
        self.calls += 1
        return self.fn(img)[0], {"infer_s": 0.01}

    def alive(self):
        return True


@pytest.fixture(scope="module")
def dev():
    if not POP.exists():
        pytest.skip("phase 4 manifests missing")
    r = {x["pair_id"]: x for x in read_csv_gz(POP)}[BLUR_ID]
    if not repo_path(r["degraded_path"]).exists():
        pytest.skip("DEV image missing")
    blur = cv2.imread(str(repo_path(r["degraded_path"])), cv2.IMREAD_COLOR)
    canvas = np.full((300, 300, 3), 127, np.uint8)
    canvas[:194, :194] = blur[30:, 30:]                              # full_image chip extends past the corner
    return {"blur": blur, "edge": canvas}


def build(core=sharpen, mode="auto", with_r6=True):
    a4, n6 = Spy(lambda x: (cv2.resize(x, (224, 224)), {})), Spy(sharpen)
    m = Model(core) if with_r6 else None
    pipe = SP.EnhancementPipeline(a4, timeout_s=10, gate="rw", nafnet=n6, route_mode=mode,
                                  experimental_nafnet={"r6": m} if with_r6 else None)
    return pipe, a4, n6, m


# ------------------------------------------------------------------ building blocks

def test_tracer_equals_r5_8_local_formula(dev):
    x = dev["blur"]
    for out in (sharpen(x)[0], MD.rainbow(x, amp=40, period=2.0), MD._fake_unstable_core(cv2.flip(x, 1), {})[0]):
        assert NR.dot_c_max(x, out) == pytest.approx(NR.local_features(x, out)["l_dot_c_max"], rel=1e-5, abs=1e-4)


def test_tracer_ignores_masked_area(dev):
    x = dev["blur"]
    y = MD.rainbow(x, amp=80, period=2.0)
    mask = np.zeros(x.shape[:2], bool)
    mask[:, :] = True
    mask[40:60, 40:60] = False
    y2 = sharpen(x)[0]
    y2[mask] = y[mask]                                             # rainbow only where masked out
    assert NR.dot_c_max(x, y2, ~mask) < NR.dot_c_max(x, y2) / 5


def test_consensus_keeps_input_colour_and_ignores_one_outlier(dev):
    x = dev["blur"]
    good = sharpen(x)[0]
    bad = MD.rainbow(x, amp=120, period=2.0)
    out = NR.consensus_luma(x, [good, good, good, bad])
    assert out.shape == x.shape and out.dtype == np.uint8
    ref = NR.consensus_luma(x, [good] * 4)
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1          # the outlier run has no effect
    assert np.abs(chroma(out).astype(int) - chroma(x).astype(int)).max() <= 2   # colour = input (YCrCb rounding)


def test_greyscale_input_stays_grey(dev):
    g = cv2.cvtColor(cv2.cvtColor(dev["blur"], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = NR.consensus_luma(g, [MD.rainbow(g, amp=50, period=2.0)] * 4)
    assert NR.gray_chroma_p99(out) < 2.0


@pytest.mark.parametrize("vals,ok", [([1, 2, 3, 4], True), ([1, 2, 3, 20.28], True), ([1, 500, 3, 4], False),
                                     ([30, 30, 30, 30], False), ([1, 2, 3], False), ([1, float("nan"), 3, 4], False),
                                     (None, False), (["x", 1, 2, 3], False)])
def test_stability_rule(vals, ok):
    cfg = NR.load_config(NR.R6_CONFIG)
    assert cfg["stability"]["run_dot_c_max"] == pytest.approx(20.2863, abs=1e-4)
    r = NR.stability_check(vals, cfg)
    assert r.passed is ok
    if not ok:
        assert r.reasons and r.reasons[0].startswith((NR.A_UNSTABLE, NR.A_UNSTABLE_ERROR))


def test_r6_config_is_r5_8_plus_stability_only():
    r6, r8 = NR.load_config(NR.R6_CONFIG), NR.load_config(NR.CANDIDATE_CONFIG)
    assert r6["route_version"] == NR.R6_ROUTE_VERSION and r6["stability"]["version"] == NR.STABILITY_VERSION
    same = set(r8) - {"sha256", "name", "route_version", "derived_from", "status", "stage", "thresholds"}
    assert {k: r6[k] for k in same} == {k: r8[k] for k in same}
    assert set(r6) - set(r8) == {"stability", "blur_range"}
    t6, t8 = r6["thresholds"], r8["thresholds"]
    assert t6["blur_max"] == t6["clean_lapd1_lo"] == 0.008                     # every blurred face, no mild band
    assert {k: v for k, v in t6.items() if k not in ("blur_max", "clean_lapd1_lo")} ==         {k: v for k, v in t8.items() if k not in ("blur_max", "clean_lapd1_lo")}


def test_r6_admits_mild_blur_that_r5_rejected():
    r6, r6_6 = NR.load_config(NR.R6_CONFIG), NR.load_config()
    f = {"det_status": "OK", "det_score": 0.95, "yaw_proxy": 0.0, "iod_px": 50.0, "luma_mean": 120.0, "rms_contrast": 50.0,
         "blockiness": 1.0, "pixelation": 0.1, "hf_peak_frac": 0.05, "hf_energy": 0.05, "noise_sigma": 0.5,
         "lapd1_c": 0.00433}                                                   # the owner's "less blurred" photo
    live = NR.decide(f, r6_6)
    if live.reason not in (NR.R_MILD, NR.ELIGIBLE):
        pytest.skip(f"feature set incomplete for decide(): {live.reason}")
    assert live.reason == NR.R_MILD and NR.decide(f, r6).eligible
    assert NR.decide({**f, "lapd1_c": 0.0125}, r6).reason == NR.R_NOT_BLUR      # sharp faces still left alone


def test_consensus_wrapper_flags_orientation_dependent_explosion(dev):
    x = dev["blur"]
    stable = MD.fake_nafnet_r6_stable_factory()[0]
    unstable = MD.fake_nafnet_r6_unstable_factory()[0]
    o1, m1 = stable(x, {})
    o2, m2 = unstable(x, {})
    assert len(m1["runs_dot_c"]) == len(m2["runs_dot_c"]) == 4
    assert max(m1["runs_dot_c"]) < 20 and max(m2["runs_dot_c"]) > 100
    assert 1 <= sum(v > 100 for v in m2["runs_dot_c"]) <= 3            # some mirror runs explode, not all
    for o in (o1, o2):
        assert o.shape == x.shape and np.abs(chroma(o).astype(int) - chroma(x).astype(int)).max() <= 2


# ------------------------------------------------------------------ pipeline profile

def test_default_request_unchanged_with_r6_enabled(dev):
    data = png(dev["blur"])
    plain = SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="rw", nafnet=Spy(sharpen),
                                   route_mode="auto").process(data, "face_chip")
    pipe, a4, n6, m = build()
    r = pipe.process(data, "face_chip")
    assert r.status == plain.status == SP.ENHANCED and r.image == plain.image and not m.ctx
    assert r.nafnet["route_version"] == NR.ROUTE_VERSION and "stability" not in r.nafnet


def test_r6_stable_is_enhanced_with_input_colour(dev):
    pipe, a4, n6, m = build()
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6")
    assert r.status == SP.ENHANCED and n6.calls == 0 and len(m.ctx) == 1
    assert r.nafnet["route_version"] == NR.R6_ROUTE_VERSION and r.nafnet["profile"] == "r6"
    assert r.nafnet["stability"]["passed"] and r.nafnet["stability"]["metrics"]["n_unstable"] == 0
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(chroma(out).astype(int) - chroma(dev["blur"]).astype(int)).max() <= 2


def test_r6_unstable_returns_original(dev):
    pipe, *_ = build(core=MD._fake_unstable_core)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.STABILITY_GUARD
    assert r.error.startswith(NR.A_UNSTABLE) and not r.nafnet["stability"]["passed"]


def test_r6_full_image_passes_oob_mask(dev):
    pipe, a4, n6, m = build()
    r = pipe.process(png(dev["edge"]), "full_image", profile="r6")
    assert r.nafnet is not None and r.nafnet["eligible"], r.reason
    assert m.ctx and "oob_mask" in m.ctx[0] and m.ctx[0]["oob_mask"].any()
    assert r.status == SP.ENHANCED


def test_r6_model_failure_falls_back(dev):
    def boom(img, ctx):
        raise RuntimeError("injected")
    pipe, *_ = build(core=boom)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == "MODEL_FAILURE"


def test_missing_tracer_rejects(dev):
    class NoMeta(Model):
        def __call__(self, img, ctx):
            return sharpen(img)[0], {"infer_s": 0.01}
    pipe = SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="rw", nafnet=Spy(sharpen),
                                  route_mode="auto", experimental_nafnet={"r6": NoMeta(sharpen)})
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.fallback == SP.STABILITY_GUARD and NR.A_UNSTABLE_ERROR in r.error


def test_r6_not_servable_unless_enabled(dev):
    pipe, a4, n6, _ = build(with_r6=False)
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6")
    assert r.status == SP.REJECTED and "not enabled" in r.error and n6.calls == 0


def test_process_level_r6_equals_per_request(dev):
    data = png(dev["blur"])
    a = SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="rw", nafnet=Model(sharpen),
                               route_mode="nafnet", nafnet_profile="r6").process(data, "face_chip")
    b = build()[0].process(data, "face_chip", profile="r6")
    assert a.status == b.status == SP.ENHANCED and a.image == b.image


# ------------------------------------------------------------------ API

def test_api_r6_profile(monkeypatch, dev):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_nafnet_r6_unstable_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    monkeypatch.delenv("FES_NAFNET_PROFILE", raising=False)
    monkeypatch.setenv("FES_EXPERIMENTAL_PROFILES", "r6")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        rd = c.get("/ready").json()
        assert rd["profiles"] == ["original", "r5_6", "r6"] and rd["default_profile"] == "r5_6"
        assert rd["experimental_models"]["r6"]["model"] == "FAKE_NAFNET_R6_UNSTABLE"
        post = lambda **f: c.post("/v1/enhance", files={"image": ("x.png", png(dev["blur"]), "image/png")},  # noqa: E731
                                  data={"input_mode": "face_chip", **f})
        six = post(profile="r6").json()
        assert six["status"] == SP.ORIGINAL_S and six["fallback"] == SP.STABILITY_GUARD
        d = post().json()
        assert "profile" not in d and d["nafnet"]["route_version"] == NR.ROUTE_VERSION


# ------------------------------------------------------------------ real model: service == R6-0 research output

@REAL
def test_real_r6_matches_r6_0_study_output():
    """The service factory reproduces the R6-0 study's median/luma output and tracers bit-exactly (DEV case)."""
    cache = repo_path("experiments/phase5_rapid/outputs/r6_0/cache/dev__fc__image_65276.jpg")
    if not cache.exists():
        pytest.skip("R6-0 cache missing")
    chip = cv2.imread(str(cache / "chip.png"), cv2.IMREAD_COLOR)
    runs = [cv2.imread(str(cache / f"{k}.png"), cv2.IMREAD_COLOR) for k in ("v0", "vh", "vv", "vhv")]
    fn = MD.nafnet_factory_r6()[0]
    out, meta = fn(chip, {})
    assert np.array_equal(out, NR.consensus_luma(chip, runs))
    assert meta["runs_dot_c"] == [round(NR.dot_c_max(chip, r), 4) for r in runs]
    rows = json.loads(repo_path("data/manifests/phase5_rapid/r6_0/cases.json").read_text())
    assert any(c["id"] == "dev:fc:image_65276.jpg" for c in rows)
