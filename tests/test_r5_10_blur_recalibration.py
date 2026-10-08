"""R5-10: the recalibrated NAFNet blur-eligibility boundary (research candidate profile r5_10, opt-in only).

Invariants: the r5_10 config is the sealed R5-8 candidate with ONLY thresholds.blur_max widened (DEV-selected); strong
blur stays eligible, the newly admitted moderate zone is eligible only in r5_10, mild blur above the new boundary and
clean inputs stay rejected, every other gate still blocks, both output guards are unchanged, the R5-6 / R5-8 configs are
untouched, and a request without `profile` never uses r5_10. DEV images only. Real-NAFNet tests: FES_RUN_NAFNET=1.
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

HAS_R10 = repo_path(NR.R5_10_CONFIG).exists()
pytestmark = pytest.mark.skipif(not HAS_R10, reason="R5-10 candidate config not written yet")
REAL = pytest.mark.skipif(os.environ.get("FES_RUN_NAFNET") != "1", reason="real NAFNet: set FES_RUN_NAFNET=1")
POP = repo_path("data/manifests/phase4/population_p4dev.csv.gz")
DEV_P4 = repo_path("data/manifests/phase5_rapid/r5_10/dev_p4.csv.gz")
STRONG_ID = "ff-train-13457.ref-v1.gaussblur-extreme"
R5_6_SEAL = "eb6fe101836aaa66da4edcfafadcf5463ccb6314e781bec29a9265230ceac8b9"
R5_8_SEAL = "631acd73ce084f2d48fbcd569a28658fc0261395dfa3ebc5b4bb01036817641f"
VOLATILE = ("request_id", "timings_ms")


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def sharpen(img):
    return MD.fake_nafnet_ok_factory()[0](img, {})[0]


class Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, img, ctx):
        self.calls += 1
        return self.fn(img), {"infer_s": 0.01}

    def alive(self):
        return True


@pytest.fixture(scope="module")
def cfgs():
    return NR.load_config(), NR.load_config(NR.CANDIDATE_CONFIG), NR.load_config(NR.R5_10_CONFIG)


@pytest.fixture(scope="module")
def imgs(cfgs):
    """strong = a DEV strong-blur face (eligible everywhere); moderate = the first p4dev DEV input in the newly admitted
    zone (R5-8 MILD_BLUR, lapd1 ≤ the new blur_max); clean = its sharp reference."""
    if not (POP.exists() and DEV_P4.exists()):
        pytest.skip("DEV manifests missing")
    new = cfgs[2]["thresholds"]["blur_max"]
    pop = {x["pair_id"]: x for x in read_csv_gz(POP)}
    mod = next(r for r in read_csv_gz(DEV_P4) if r["population"] == "p4dev" and
               r["fc_reason"] == NR.R_MILD and float(r["fc_lapd1_c"]) <= new)
    out = {}
    for k, p in (("strong", pop[STRONG_ID]["degraded_path"]), ("moderate", pop[mod["name"]]["degraded_path"]),
                 ("clean", pop[STRONG_ID]["ref_path"])):
        if not repo_path(p).exists():
            pytest.skip("DEV images missing")
        out[k] = cv2.imread(str(repo_path(p)), cv2.IMREAD_COLOR)
    return out


@pytest.fixture(scope="module")
def base_f(imgs):
    """Real features of an eligible DEV strong-blur face: passes every gate; tests then move ONE feature."""
    from fes_data import preprocess as P
    from fes_service import rwgate as RW
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    f = RW.rw_features(imgs["strong"], det)
    f.update(NR.texture_features(imgs["strong"]))
    return f


# ------------------------------------------------------------------ config: one change only, sealed, nothing else moved

def test_live_and_r5_8_configs_untouched(cfgs):
    assert cfgs[0]["sha256"] == R5_6_SEAL and cfgs[0]["route_version"] == NR.ROUTE_VERSION
    assert cfgs[1]["sha256"] == R5_8_SEAL and cfgs[1]["route_version"] == NR.CANDIDATE_ROUTE_VERSION


def test_r5_10_differs_from_r5_8_only_in_blur_max(cfgs):
    _, r8, r10 = cfgs
    assert r10["route_version"] == NR.R5_10_ROUTE_VERSION and r10["derived_from"]["r5_8_candidate_seal"] == R5_8_SEAL
    meta = ("sha256", "name", "route_version", "derived_from", "status", "stage", "blur_recalibration", "thresholds")
    assert {k: v for k, v in r8.items() if k not in meta} == {k: v for k, v in r10.items() if k not in meta}
    t8, t10 = dict(r8["thresholds"]), dict(r10["thresholds"])
    old, new = t8.pop("blur_max"), t10.pop("blur_max")
    assert t8 == t10 and old == 0.002617 and old < new < t10["clean_lapd1_lo"]
    assert r10["blur_recalibration"]["old_blur_max"] == old and r10["blur_recalibration"]["new_blur_max"] == new
    assert r10["guard"] == r8["guard"] and r10["local_artifact"] == r8["local_artifact"]
    assert r10["framing"] == r8["framing"] and r10["model_asset"] == r8["model_asset"]


def test_r5_10_seal_detects_tampering(cfgs, tmp_path, monkeypatch):
    import json
    bad = {**cfgs[2], "thresholds": {**cfgs[2]["thresholds"], "blur_max": 0.0055}}
    p = tmp_path / "x.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(NR, "repo_path", lambda _: p)
    with pytest.raises(AssertionError):
        NR.load_config("ignored")


# ------------------------------------------------------------------ input gate: the blur zones

def _with(f, **kw):
    return {**f, **kw}


def test_blur_zones(cfgs, base_f):
    r6, r8, r10 = cfgs
    old, new, clean = r8["thresholds"]["blur_max"], r10["thresholds"]["blur_max"], r10["thresholds"]["clean_lapd1_lo"]
    cases = {  # lapd1_c → expected reason under (r5_6, r5_8, r5_10)
        0.0010: (NR.ELIGIBLE, NR.ELIGIBLE, NR.ELIGIBLE),                 # very strong blur: unchanged
        old: (NR.ELIGIBLE, NR.ELIGIBLE, NR.ELIGIBLE),                    # old boundary still inclusive
        (old + new) / 2: (NR.R_MILD, NR.R_MILD, NR.ELIGIBLE),            # newly admitted moderate blur
        new: (NR.R_MILD, NR.R_MILD, NR.ELIGIBLE),                        # new boundary inclusive
        (new + clean) / 2: (NR.R_MILD, NR.R_MILD, NR.R_MILD),            # mild blur above the new boundary
        clean: (NR.R_NOT_BLUR,) * 3, 0.02: (NR.R_NOT_BLUR,) * 3}        # clean / sharp
    for lap, exp in cases.items():
        got = tuple(NR.decide(_with(base_f, lapd1_c=lap), c).reason for c in (r6, r8, r10))
        assert got == exp, (lap, got)


@pytest.mark.parametrize("change,reason", [
    ({"noise_sigma": 3.0}, NR.R_NOISE), ({"blockiness": 3.0}, NR.R_COMP), ({"pixelation": 0.9}, NR.R_PIX),
    ({"hf_peak_frac": 0.6, "hf_energy": 0.3}, NR.R_HALF), ({"luma_mean": 20.0}, NR.R_PHOTO),
    ({"rms_contrast": 10.0}, NR.R_PHOTO), ({"iod_px": 20.0}, NR.R_SIZE), ({"lapd1_c": float("nan")}, NR.R_MISSING)])
def test_other_gates_still_block_newly_admitted_blur(cfgs, base_f, change, reason):
    lap = (0.002617 + cfgs[2]["thresholds"]["blur_max"]) / 2
    f = _with(base_f, lapd1_c=lap)
    assert NR.decide(f, cfgs[2]).eligible
    d = NR.decide(_with(f, **change), cfgs[2])
    assert not d.eligible and d.reason.startswith(reason)


def test_safety_rules_still_block(cfgs, base_f):
    lap = (0.002617 + cfgs[2]["thresholds"]["blur_max"]) / 2
    d = NR.decide(_with(base_f, lapd1_c=lap, det_score=0.3), cfgs[2])
    assert not d.eligible and d.reason in (NR.R_LOW_DET, NR.R_ALIGN, NR.R_NO_FACE)


# ------------------------------------------------------------------ pipeline: opt-in only, default unchanged

def build(experimental=("r5_8", "r5_10"), fn=sharpen):
    a4, n6, nx = Spy(lambda x: cv2.resize(x, (224, 224))), Spy(fn), Spy(fn)
    pipe = SP.EnhancementPipeline(a4, timeout_s=10, gate="rw", nafnet=n6, route_mode="auto",
                                  experimental_nafnet={k: nx for k in experimental} or None)
    return pipe, n6, nx


def strip(m):
    return {k: v for k, v in m.items() if k not in VOLATILE}


@pytest.mark.parametrize("key", ["strong", "moderate", "clean"])
def test_default_request_unchanged_with_r5_10_enabled(imgs, key):
    data = png(imgs[key])
    old = build(experimental=())[0].process(data, "face_chip")
    pipe, n6, nx = build()
    new = pipe.process(data, "face_chip")
    assert new.image == old.image and strip(new.metadata()) == strip(old.metadata()) and nx.calls == 0
    assert pipe.seals["nafnet_route"] == R5_6_SEAL


def test_r5_10_not_servable_unless_enabled(imgs):
    pipe = build(experimental=("r5_8",))[0]
    r = pipe.process(png(imgs["moderate"]), "face_chip", profile="r5_10")
    assert r.status == SP.REJECTED and "not enabled" in r.error and "FES_EXPERIMENTAL_PROFILES=r5_10" in r.error


def test_moderate_blur_attempted_only_by_r5_10(imgs):
    pipe, n6, nx = build()
    data = png(imgs["moderate"])
    six, eight = pipe.process(data, "face_chip", profile="r5_6"), pipe.process(data, "face_chip", profile="r5_8")
    assert six.reason == NR.R_MILD and eight.reason == NR.R_MILD and nx.calls == 0 and n6.calls == 0
    ten = pipe.process(data, "face_chip", profile="r5_10")
    assert ten.nafnet["eligible"] and ten.model == NR.MODEL and nx.calls == 1
    assert ten.nafnet["route_version"] == NR.R5_10_ROUTE_VERSION and ten.nafnet["profile"] == "r5_10"
    assert ten.nafnet["local_artifact"] is not None                       # both output guards ran
    assert pipe.seals["nafnet_route_r5_10"] == NR.load_config(NR.R5_10_CONFIG)["sha256"]


def test_strong_blur_same_decision_in_r5_8_and_r5_10(imgs):
    pipe = build()[0]
    data = png(imgs["strong"])
    e, t = pipe.process(data, "face_chip", profile="r5_8"), pipe.process(data, "face_chip", profile="r5_10")
    assert e.reason == t.reason == NR.ELIGIBLE and e.image == t.image


def test_clean_reference_rejected_by_r5_10(imgs):
    r = build()[0].process(png(imgs["clean"]), "face_chip", profile="r5_10")
    assert r.status == SP.ORIGINAL_S and not r.nafnet["eligible"]


def test_r5_10_output_guards_still_return_original(imgs):
    def patchy(x):
        out = sharpen(x)
        out[60:66, 60:66] = MD.rainbow(x, amp=90.0, period=2.0)[60:66, 60:66]
        return out
    pipe = build(fn=patchy)[0]
    data = png(imgs["moderate"])
    r = pipe.process(data, "face_chip", profile="r5_10")
    assert r.status == SP.ORIGINAL_S and r.image == data
    assert r.fallback in (SP.LOCAL_ARTIFACT_GUARD, SP.ARTIFACT_GUARD)


def test_r5_10_model_failure_falls_back(imgs):
    def boom(x):
        raise RuntimeError("injected")
    data = png(imgs["moderate"])
    r = build(fn=boom)[0].process(data, "face_chip", profile="r5_10")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == "MODEL_FAILURE"


def test_r5_10_as_process_default_equals_per_request(imgs):
    data = png(imgs["moderate"])
    ref = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(sharpen), route_mode="auto",
                                 nafnet_profile="r5_10").process(data, "face_chip")
    r = build()[0].process(data, "face_chip", profile="r5_10")
    m = r.metadata()
    assert m.pop("profile") == "r5_10" and strip(m) == strip(ref.metadata()) and r.image == ref.image


# ------------------------------------------------------------------ API

def _client(monkeypatch, experimental):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setenv("FES_NAFNET_R5_8_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    monkeypatch.delenv("FES_NAFNET_PROFILE", raising=False)
    if experimental is None:
        monkeypatch.delenv("FES_EXPERIMENTAL_PROFILES", raising=False)
    else:
        monkeypatch.setenv("FES_EXPERIMENTAL_PROFILES", experimental)
    monkeypatch.setitem(A.STATE, "ready", False)
    return TestClient(A.app)


def post(c, data, **form):
    return c.post("/v1/enhance", files={"image": ("x.png", data, "image/png")}, data={"input_mode": "face_chip", **form})


def test_api_default_has_no_r5_10(monkeypatch, imgs):
    with _client(monkeypatch, None) as c:
        rd = c.get("/ready").json()
        assert rd["profiles"] == ["original", "r5_6"] and "nafnet_route_r5_10" not in rd["seals"]
        r = post(c, png(imgs["moderate"]), profile="r5_10")
        assert r.status_code == 422 and "not enabled" in r.json()["error"]


def test_api_r5_8_and_r5_10_share_one_worker(monkeypatch, imgs):
    from fes_service import app as A
    with _client(monkeypatch, "r5_8,r5_10") as c:
        rd = c.get("/ready").json()
        assert rd["profiles"] == ["original", "r5_6", "r5_8", "r5_10"] and rd["default_profile"] == "r5_6"
        assert rd["experimental_models"]["r5_8"] == rd["experimental_models"]["r5_10"]
        pipe = A.STATE["pipeline"]
        assert pipe.profiles["r5_8"].model is pipe.profiles["r5_10"].model is not pipe.nafnet
        data = png(imgs["moderate"])
        ten = post(c, data, profile="r5_10").json()
        assert ten["profile"] == "r5_10" and ten["nafnet"]["route_version"] == NR.R5_10_ROUTE_VERSION
        assert ten["nafnet"]["eligible"]
        d = post(c, data).json()
        assert "profile" not in d and d["nafnet"]["route_version"] == NR.ROUTE_VERSION and not d["nafnet"]["eligible"]


def test_api_r5_10_alone(monkeypatch, imgs):
    with _client(monkeypatch, "r5_10") as c:
        rd = c.get("/ready").json()
        assert rd["profiles"] == ["original", "r5_6", "r5_10"]
        assert post(c, png(imgs["moderate"]), profile="r5_8").status_code == 422


# ------------------------------------------------------------------ real NAFNet

@REAL
def test_real_nafnet_r5_10_moderate_blur(imgs):
    from fes_service.worker import IsolatedModel
    nx = IsolatedModel("fes_service.models:nafnet_factory_pad16", timeout_s=120, threads=4)
    try:
        nx.start()
        pipe = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=120, gate="rw", nafnet=Spy(sharpen),
                                      route_mode="auto", experimental_nafnet={"r5_10": nx})
        data = png(imgs["moderate"])
        r = pipe.process(data, "face_chip", profile="r5_10")
        assert r.nafnet["eligible"] and r.model == NR.MODEL and r.nafnet["artifact"] is not None
        assert r.status in (SP.ENHANCED, SP.ORIGINAL_S)
        if r.status == SP.ENHANCED:
            out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
            assert out.shape == imgs["moderate"].shape and r.nafnet["local_artifact"]["passed"]
    finally:
        nx.close()
