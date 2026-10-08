"""R5-8: research-candidate profile (opt-in) + local-patch artifact guard (p5-nafnet-local-artifact-1.0.0).

The live/default profile r5_6 must be exactly the R5-6 route; the candidate is only reachable with
nafnet_profile="r5_8" / FES_NAFNET_PROFILE=r5_8. DEV images only (p4dev pair + its reference).
"""

from __future__ import annotations

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
HAS_FINAL = "local_artifact" in NR.load_config(NR.CANDIDATE_CONFIG)


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(scope="module")
def blur():
    if not POP.exists():
        pytest.skip("phase 4 manifests missing")
    p = repo_path({r["pair_id"]: r for r in read_csv_gz(POP)}[BLUR_ID]["degraded_path"])
    if not p.exists():
        pytest.skip("DEV image missing")
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def sharpen(img):
    return MD.fake_nafnet_ok_factory()[0](img, {})[0]


def patch(img, y=100, x=100, s=8, amp=90.0):
    out = sharpen(img)
    r = MD.rainbow(img, amp=amp, period=2.0)
    out[y:y + s, x:x + s] = r[y:y + s, x:x + s]
    return out


def cfg_with(limits):
    return {"local_artifact": {"version": NR.LOCAL_VERSION, "win": NR.LOCAL_WIN, "limits": limits}}


SYNTH = cfg_with({"l_hfc_growth_max": 200.0})              # test-only limits (the real ones come from DEV)


class Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, img, ctx):
        self.calls += 1
        return self.fn(img), {"infer_s": 0.01}

    def alive(self):
        return True


# ------------------------------------------------------------------ profile isolation

def test_default_profile_is_the_live_r5_6_route(blur):
    pipe = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(sharpen), route_mode="nafnet")
    assert pipe.nafnet_profile == "r5_6" and pipe.nafnet_cfg["route_version"] == NR.ROUTE_VERSION
    assert pipe.seals["nafnet_route"] == NR.load_config()["sha256"] and not pipe.oob_fill
    assert "local_artifact" not in pipe.nafnet_cfg and "chroma_p999_max" not in pipe.nafnet_cfg["guard"]
    r = pipe.process(png(blur), "face_chip")
    assert set(r.nafnet) == {"eligible", "reason", "decision", "route", "route_version", "infer_s", "artifact"}
    assert "chroma_p999" not in r.nafnet["artifact"]["metrics"]


def test_candidate_profile_is_opt_in_and_sealed():
    cfg = NR.load_config(NR.CANDIDATE_CONFIG)
    assert cfg["route_version"] == NR.CANDIDATE_ROUTE_VERSION and "RESEARCH CANDIDATE" in cfg["status"]
    assert cfg["derived_from"]["live_route_seal"] == NR.load_config()["sha256"]
    pipe = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(sharpen), route_mode="nafnet",
                                  nafnet_profile="r5_8")
    assert pipe.oob_fill and pipe.seals["nafnet_route"] == cfg["sha256"]
    with pytest.raises(AssertionError):
        SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(sharpen), nafnet_profile="r5_7")


def test_app_defaults_to_r5_6(monkeypatch):
    pytest.importorskip("fastapi")
    from fes_service import app as A
    monkeypatch.delenv("FES_NAFNET_PROFILE", raising=False)
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    pipe = A.build_pipeline()
    assert pipe.nafnet_profile == "r5_6" and pipe.nafnet.factory == "fes_service.models:fake_nafnet_ok_factory"
    monkeypatch.delenv("FES_NAFNET_FACTORY")
    assert A.build_pipeline().nafnet.factory == "fes_service.models:nafnet_factory"
    monkeypatch.setenv("FES_NAFNET_PROFILE", "r5_8")
    p8 = A.build_pipeline()
    assert p8.nafnet_profile == "r5_8" and p8.nafnet.factory == "fes_service.models:nafnet_factory_pad16"


# ------------------------------------------------------------------ local guard (unit)

def test_legitimate_output_accepted(blur):
    g = NR.local_artifact_guard(blur, sharpen(blur), SYNTH, False)
    assert g.passed and g.reasons == []


def test_localized_artifact_blocked(blur):
    g = NR.local_artifact_guard(blur, patch(blur), SYNTH, False)
    assert not g.passed and g.reasons == [f"{NR.A_LOCAL}:l_hfc_growth_max"]


def test_destructive_output_blocked(blur):
    g = NR.local_artifact_guard(blur, MD.rainbow(blur), SYNTH, False)
    assert not g.passed


def test_grayscale_accepted_when_valid(blur):
    gray = cv2.cvtColor(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = cv2.cvtColor(cv2.cvtColor(sharpen(gray), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert NR.local_artifact_guard(gray, out, SYNTH, True).passed


@pytest.mark.parametrize("bad", [None, "x", np.zeros((10, 10, 3), np.uint8), np.zeros((224, 224, 3), np.float32)])
def test_invalid_output_rejected(blur, bad):
    g = NR.local_artifact_guard(blur, bad, SYNTH, False)
    assert not g.passed and g.reasons == [NR.A_LOCAL_ERROR]


def test_local_guard_is_deterministic_and_never_modifies_output(blur):
    out = patch(blur)
    keep = out.copy()
    a, b = NR.local_artifact_guard(blur, out, SYNTH, False), NR.local_artifact_guard(blur, out, SYNTH, False)
    assert a.as_dict() == b.as_dict() and np.array_equal(out, keep)


# ------------------------------------------------------------------ pipeline integration (candidate profile)

def candidate(nafnet_fn, monkeypatch, limits=None):
    pipe = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(nafnet_fn),
                                  route_mode="nafnet", nafnet_profile="r5_8")
    if limits is not None or "local_artifact" not in pipe.nafnet_cfg:
        monkeypatch.setitem(pipe.nafnet_cfg, "local_artifact", cfg_with(limits or {"l_hfc_growth_max": 200.0})["local_artifact"])
    return pipe


def test_candidate_local_patch_returns_original(blur, monkeypatch):
    pipe = candidate(lambda x: patch(x, s=6, amp=30.0), monkeypatch, {"l_hfc_growth_max": 200.0})
    data = png(blur)
    r = pipe.process(data, "face_chip")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.LOCAL_ARTIFACT_GUARD
    assert r.artifact_check == "FAIL" and r.nafnet["local_artifact"]["passed"] is False and r.model == NR.MODEL
    assert r.nafnet["artifact"]["passed"]                              # the global guard alone would have passed it


def test_candidate_legitimate_output_enhanced(blur, monkeypatch):
    pipe = candidate(sharpen, monkeypatch, {"l_hfc_growth_max": 200.0})
    r = pipe.process(png(blur), "face_chip")
    assert r.status == SP.ENHANCED and r.nafnet["local_artifact"]["passed"] and r.nafnet["profile"] == "r5_8"
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out, sharpen(blur))                          # accepted pixels are not modified


def test_detector_exception_returns_original(blur, monkeypatch):
    pipe = candidate(sharpen, monkeypatch, {"l_hfc_growth_max": 200.0})
    monkeypatch.setattr(NR, "local_artifact_guard", lambda *a, **k: 1 / 0)
    data = png(blur)
    r = pipe.process(data, "face_chip")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.LOCAL_ARTIFACT_GUARD
    assert NR.A_LOCAL_ERROR in r.error


def test_candidate_full_image_fill_and_black_border(blur, monkeypatch):
    canvas = np.full((300, 300, 3), 127, np.uint8)
    canvas[:194, :194] = blur[30:, 30:]                                # chip extends ~30 px past the corner
    seen = {}

    def ident(x):
        seen["x"] = x.copy()
        return x.copy()
    pipe = candidate(ident, monkeypatch, {"l_hfc_growth_max": 1e9})
    r = pipe.process(png(canvas), "full_image")
    chip = cv2.imdecode(np.frombuffer(r.chip, np.uint8), cv2.IMREAD_COLOR)
    oob = chip.max(axis=2) == 0
    assert r.nafnet["oob_fill_frac"] > 0 and oob.any()
    assert r.status == SP.ENHANCED and seen["x"][oob].max() > 0       # NAFNet saw reflected content, not black
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert out[oob].max() == 0 and np.array_equal(out[~oob], chip[~oob])


def test_live_profile_never_fills(blur):
    canvas = np.full((300, 300, 3), 127, np.uint8)
    canvas[:194, :194] = blur[30:, 30:]
    seen = {}

    def ident(x):
        seen["x"] = x.copy()
        return x.copy()
    pipe = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(ident), route_mode="nafnet")
    r = pipe.process(png(canvas), "full_image")
    if "x" in seen:
        chip = cv2.imdecode(np.frombuffer(r.chip, np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(seen["x"], chip)                         # black-filled chip, exactly as R5-6


# ------------------------------------------------------------------ reflect-pad16 factory

def test_reflect_pad16():
    img = np.random.default_rng(5).integers(0, 255, (99, 107, 3), np.uint8)
    x, h, w = MD.reflect_pad16(img)
    assert x.shape == (112, 112, 3) and (h, w) == (99, 107) and np.array_equal(x[:99, :107], img)
    same = np.zeros((128, 224, 3), np.uint8)
    assert MD.reflect_pad16(same)[0] is same


@REAL
def test_real_pad16_factory_keeps_size_and_matches_live_on_multiples_of_16(blur):
    f16, info = MD.nafnet_factory_pad16()
    f0, _ = MD.nafnet_factory()
    c99 = cv2.resize(blur, (99, 99), interpolation=cv2.INTER_AREA)
    out, meta = f16(c99, {})
    assert out.shape == c99.shape and meta["pad16"] == [13, 13] and info["variant"] == "reflect_pad16"
    c128 = cv2.resize(blur, (128, 128), interpolation=cv2.INTER_AREA)
    assert np.array_equal(f16(c128, {})[0], f0(c128, {})[0])


@pytest.mark.skipif(not HAS_FINAL, reason="final candidate config not written yet")
def test_final_candidate_config_has_dev_calibrated_local_guard():
    lc = NR.load_config(NR.CANDIDATE_CONFIG)["local_artifact"]
    assert lc["version"] == NR.LOCAL_VERSION and lc["limits"] and "dev" in lc["calibration"]["data"].lower()
