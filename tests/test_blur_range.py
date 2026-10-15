"""Admin blur range (fes_service.blur_scale): the 0-100 % blur scale, enhance_from_pct and give_up_above_pct.

Invariants: the scale is sealed, monotone and anchored to the validated blur limits; with both settings off the gate
decides exactly as before (same outputs); the settings can only make the gate STRICTER — enhance_from_pct below the
validated minimum, a give_up_above_pct not above enhance_from_pct, or a request that widens the deployment's range is
refused; a face outside the range is returned unchanged without running a model; every NAFNet answer reports the
face's blur percent and the range applied. Images: the Indian-expression benchmark (skipped when absent).
"""

from __future__ import annotations

import json
import math

import cv2
import pytest

from fes_service import blur_scale as BS
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from test_r6_best import strong
from test_r6_candidate import Model, Spy, sharpen
from test_side_face import FRONT, SIDE, load


@pytest.fixture(scope="module")
def scale():
    return BS.load_scale()


def build(settings=None):
    m6, ft = Model(sharpen), Model(strong)
    pipe = SP.EnhancementPipeline(Spy(lambda x: (cv2.resize(x, (224, 224)), {})), timeout_s=10, gate="rw",
                                  nafnet=Spy(sharpen), route_mode="auto", experimental_nafnet={"r6": m6, "r6_ft": ft},
                                  blur_settings=settings)
    return pipe, m6, ft


# ------------------------------------------------------------------ the scale

def test_scale_is_sealed_monotone_and_anchored(scale):
    for framing in BS.FRAMINGS:
        pts = scale.cfg["curves"][framing]
        assert [p["percent"] for p in pts] == [0.0, 20.0, 37.5, 62.5, 100.0]     # 25 % per Gaussian sigma
        for p in pts:
            assert scale.percent(framing, p["lapd1_c"]) == pytest.approx(p["percent"])
        assert scale.percent(framing, pts[0]["lapd1_c"] * 3) == 0.0                     # sharper than sharp: 0
        assert scale.percent(framing, pts[-1]["lapd1_c"] / 3) == 100.0                  # beyond sigma 4: 100
        xs = [pts[-1]["lapd1_c"] * 1.05 ** i for i in range(60)]
        ys = [scale.percent(framing, x) for x in xs]
        assert all(a >= b for a, b in zip(ys, ys[1:]))                                  # sharper → lower percent
    assert math.isnan(scale.percent("front", float("nan")))
    floor = scale.floor()
    sealed = {"front": NR.load_config(NR.R6_CONFIG)["thresholds"]["blur_max"],
              "side": NR.load_side_config()["thresholds"]["blur_max"]}
    for framing, v in sealed.items():                                   # the floor is stored to 3 decimals
        assert floor[framing] == pytest.approx(scale.percent(framing, v), abs=1e-3)
    assert scale.min_enhance_from() == math.ceil(max(floor.values())) == 36


# ------------------------------------------------------------------ settings validation

def test_validation_only_allows_stricter(scale):
    S = BS.BlurSettings
    assert BS.validate(S(), scale) == S()
    assert BS.validate(S(36, 80), scale) == S(36, 80)
    with pytest.raises(ValueError, match="below the validated minimum"):
        BS.validate(S(20, None), scale)
    with pytest.raises(ValueError, match="must be above"):
        BS.validate(S(50, 50), scale)
    with pytest.raises(ValueError, match="must be above"):
        BS.validate(S(None, 30), scale)                               # below the validated limit itself
    with pytest.raises(ValueError, match="below 100"):
        BS.validate(S(None, 100), scale)
    base = S(40, 80)
    assert BS.validate(S(), scale, base) == base                      # request without values: the deployment's
    assert BS.validate(S(50, 70), scale, base) == S(50, 70)           # narrower: accepted
    assert BS.validate(S(None, 70), scale, base) == S(40, 70)
    with pytest.raises(ValueError, match="only narrow"):
        BS.validate(S(38, None), scale, base)
    with pytest.raises(ValueError, match="only narrow"):
        BS.validate(S(None, 90), scale, base)
    with pytest.raises(ValueError, match="must be above"):
        BS.validate(S(None, 35), scale, base)                         # narrower than 80 but not above 40


@pytest.mark.parametrize("v", ["abc", "-1", "101", "nan", "inf"])
def test_parse_refuses_non_percentages(v):
    with pytest.raises(ValueError, match="0 to 100"):
        BS.parse(v, None)
    assert BS.parse("", "") == BS.BlurSettings() and BS.parse("40", 70) == BS.BlurSettings(40.0, 70.0)


def test_settings_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FES_SETTINGS", raising=False)
    assert BS.load_settings()[0] == BS.BlurSettings()                 # the shipped file: both null
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"blur": {"enhance_from_pct": 45, "give_up_above_pct": 85}}), encoding="utf-8")
    monkeypatch.setenv("FES_SETTINGS", str(p))
    assert BS.load_settings() == (BS.BlurSettings(45.0, 85.0), str(p))
    p.write_text(json.dumps({"blur": {"enhance_from": 45}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown blur setting"):
        BS.load_settings()
    monkeypatch.setenv("FES_SETTINGS", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="does not exist"):
        BS.load_settings()


def test_pipeline_refuses_invalid_deployment_settings():
    with pytest.raises(ValueError, match="validated minimum"):
        build(BS.BlurSettings(10, None))
    with pytest.raises(ValueError, match="need gate 'rw'"):
        SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="t6", blur_settings=BS.BlurSettings(40, None))


# ------------------------------------------------------------------ the gate rule

def test_decide_rule_is_stricter_only():
    cfg = NR.load_config(NR.R6_CONFIG)
    t = cfg["thresholds"]
    f = {"det_status": "FACE_TOO_SMALL", "det_score": 0.95, "iod_px": 80.0, "lapd1_c": t["blur_max"] / 2,
         "noise_sigma": 0.0, "luma_mean": 120.0, "rms_contrast": 40.0, "blockiness": 1.0, "pixelation": 0.1,
         "hf_peak_frac": 0.05, "hf_energy": 0.03}
    br = lambda v: {"verdict": v}  # noqa: E731
    assert NR.decide(f, cfg, NR.box_safety).eligible
    assert NR.decide(f, cfg, NR.box_safety, br(BS.IN_RANGE)).eligible
    assert NR.decide(f, cfg, NR.box_safety, br(BS.NOT_BLURRY_ENOUGH)).reason == NR.R_MILD
    assert NR.decide(f, cfg, NR.box_safety, br(BS.TOO_BLURRED)).reason == NR.R_TOO_BLURRED
    noisy = {**f, "noise_sigma": t["noise_max"] + 1}
    assert NR.decide(noisy, cfg, NR.box_safety, br(BS.TOO_BLURRED)).reason == NR.R_NOISE      # sealed rules first


# ------------------------------------------------------------------ the service

def test_default_is_unchanged_and_reports_the_blur(scale):
    data = load(FRONT)
    pipe, m6, ft = build()
    r = pipe.process(data, "full_image", profile="best")
    assert r.status == SP.ENHANCED and len(m6.ctx) == len(ft.ctx) == 1
    b = r.nafnet["blur_range"]
    assert b["framing"] == "front" and b["verdict"] == BS.IN_RANGE and b["give_up_above_pct"] is None
    assert b["enhance_from_pct"] == round(scale.floor()["front"], 1)
    assert b["source"] == {"enhance_from": "validated_limit", "give_up_above": "off"}
    assert b["blur_pct"] == round(scale.percent("front", r.quality["lapd1_c"]), 1)
    wide = build(BS.BlurSettings(36, 99))[0].process(data, "full_image", profile="best")    # range that still admits it
    assert wide.status == SP.ENHANCED and wide.image == r.image
    assert wide.nafnet["blur_range"]["source"] == {"enhance_from": "deployment", "give_up_above": "deployment"}


def test_outside_the_range_returns_the_original_without_a_model():
    data = load(FRONT)
    pipe, m6, ft = build()
    pct = pipe.process(data, "full_image", profile="best").nafnet["blur_range"]["blur_pct"]
    n6, nft = len(m6.ctx), len(ft.ctx)
    hi = pipe.process(data, "full_image", profile="best", blur=BS.BlurSettings(None, pct - 1))
    assert hi.status == SP.ORIGINAL_S and hi.image == data and hi.reason == NR.R_TOO_BLURRED
    assert hi.nafnet["blur_range"]["verdict"] == BS.TOO_BLURRED
    assert hi.nafnet["blur_range"]["source"] == {"enhance_from": "validated_limit", "give_up_above": "request"}
    lo = pipe.process(data, "full_image", profile="best", blur=BS.BlurSettings(min(pct + 1, 99), None))
    assert lo.status == SP.ORIGINAL_S and lo.reason == NR.R_MILD
    assert lo.nafnet["blur_range"]["verdict"] == BS.NOT_BLURRY_ENOUGH
    assert (len(m6.ctx), len(ft.ctx)) == (n6, nft)                                  # no model ran


def test_request_may_only_narrow_the_deployment_range():
    data = load(FRONT)
    pipe, m6, _ = build(BS.BlurSettings(40, 80))
    r = pipe.process(data, "full_image", profile="best", blur=BS.BlurSettings(None, 90))
    assert r.status == SP.REJECTED and "only narrow" in r.error and r.image is None and not m6.ctx
    r = pipe.process(data, "full_image", profile="best", blur=BS.BlurSettings(20, None))
    assert r.status == SP.REJECTED and "validated minimum" in r.error


def test_side_face_uses_the_side_scale(scale):
    data = load(SIDE)
    pipe, *_ = build()
    s = pipe.process(data, "full_image", profile="best_side")
    b = s.nafnet["blur_range"]
    assert s.nafnet["framing"] == "detection_box" and b["framing"] == "side"
    assert b["blur_pct"] == round(scale.percent("side", s.quality["lapd1_c"]), 1)
    assert b["enhance_from_pct"] == round(scale.floor()["side"], 1)
    t = pipe.process(data, "full_image", profile="best_side", blur=BS.BlurSettings(None, b["blur_pct"] - 1))
    assert t.status == SP.ORIGINAL_S and t.reason == NR.R_TOO_BLURRED


def test_other_profiles_are_unchanged():
    data = load(FRONT)
    pipe, *_ = build(BS.BlurSettings(40, 80))
    r = pipe.process(data, "full_image")                                # the default r5_6 route: no blur range
    assert r.status != SP.REJECTED and "blur_range" not in (r.nafnet or {})
    r = pipe.process(data, "full_image", blur=BS.BlurSettings(50, None))
    assert r.status == SP.REJECTED and "apply to profiles" in r.error
    assert pipe.process(data, "full_image", profile="original", blur=BS.BlurSettings(50, None)).status == SP.ORIGINAL_S


def test_api_settings_file_form_fields_and_ready(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    data = load(FRONT)
    for k, v in {"FES_EXPERIMENTAL_PROFILES": "r6,r6_ft", "FES_A4_FACTORY": "fes_service.models:fake_ok_factory",
                 "FES_NAFNET_FACTORY": "fes_service.models:fake_nafnet_ok_factory",
                 "FES_NAFNET_R6_FACTORY": "fes_service.models:fake_nafnet_r6_stable_factory",
                 "FES_NAFNET_R6_FT_FACTORY": "fes_service.models:fake_nafnet_r6_stable_factory", "FES_WARMUP": "0"}.items():
        monkeypatch.setenv(k, v)
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"blur": {"enhance_from_pct": 40, "give_up_above_pct": 95}}), encoding="utf-8")
    monkeypatch.setenv("FES_SETTINGS", str(p))
    monkeypatch.setitem(A.STATE, "ready", False)
    files = {"image": ("x.png", data, "image/png")}
    with TestClient(A.app) as c:
        rd = c.get("/ready").json()
        assert rd["ready"], rd["error"]
        bs = rd["blur_settings"]
        assert (bs["enhance_from_pct"], bs["give_up_above_pct"], bs["min_enhance_from_pct"]) == (40, 95, 36)
        assert set(bs["validated_limit_pct"]) == {"front", "side"} and bs["source"] == str(p)
        b = c.post("/v3/enhance", files=files, data={"return_image": "false"}).json()
        pct = b["nafnet"]["blur_range"]["blur_pct"]
        assert b["nafnet"]["blur_range"]["enhance_from_pct"] == 40 and b["nafnet"]["blur_range"]["give_up_above_pct"] == 95
        t = c.post("/v3/enhance", files=files, data={"give_up_above_pct": str(pct - 1), "return_image": "false"})
        assert t.status_code == 200 and t.json()["reason"] == NR.R_TOO_BLURRED
        w = c.post("/v3/enhance", files=files, data={"give_up_above_pct": "99"})
        assert w.status_code == 422 and "only narrow" in w.json()["error"]
        x = c.post("/v3/enhance", files=files, data={"enhance_from_pct": "lots"})
        assert x.status_code == 422 and "0 to 100" in x.json()["error"]
    p.write_text(json.dumps({"blur": {"enhance_from_pct": 10}}), encoding="utf-8")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:                                        # invalid deployment value: never ready
        rd = c.get("/ready")
        assert rd.status_code == 503 and "validated minimum" in rd.json()["error"]
