"""R5-5 guarded NAFNet-GoPro route: input gate, artifact guard, pipeline routing, fail-safe worker paths, API.

Model calls use spies / the fake factories in fes_service.models. The real NAFNet tests (checkpoint, determinism,
CPU time, no reload) need torch: they run with FES_RUN_NAFNET=1 inside fes-dev:cpu-p5 (≈ 30 s).
DEV data only (p4dev pair + its clean reference); no held-out evaluation image is used.
"""

from __future__ import annotations

import os
import socket

import cv2
import numpy as np
import pytest

from fes_bench import gate as GT
from fes_bench import routing as RT
from fes_data import preprocess as P
from fes_data.common import read_csv_gz, repo_path
from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from fes_service import rwgate as RW
from fes_service.worker import IsolatedModel

POP = repo_path("data/manifests/phase4/population_p4dev.csv.gz")
BLUR_ID = "ff-train-13457.ref-v1.gaussblur-extreme"          # DEV strong low-noise blur (224 px chip)
REAL = pytest.mark.skipif(os.environ.get("FES_RUN_NAFNET") != "1", reason="real NAFNet: set FES_RUN_NAFNET=1")


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(scope="module")
def cfg() -> dict:
    return NR.load_config()


@pytest.fixture(scope="module")
def det():
    return P.YuNetDetector(repo_path(SP.YUNET), 0.7, 0.3, max_side=640)


@pytest.fixture(scope="module")
def dev(det) -> dict:
    """{'blur': DEV strong-blur chip, 'clean': its clean reference, 'noise': the first DEV noise chip (sorted pair id)
    that the p5-rwgate routes strong_noise → A4}."""
    if not POP.exists():
        pytest.skip("phase 4 manifests missing")
    pop = {r["pair_id"]: r for r in read_csv_gz(POP)}
    r = pop[BLUR_ID]
    out = {}
    for k, p in (("blur", r["degraded_path"]), ("clean", r["ref_path"])):
        if not repo_path(p).exists():
            pytest.skip("DEV images missing")
        out[k] = cv2.imread(str(repo_path(p)), cv2.IMREAD_COLOR)
    rw = RW.load_config()
    for k in sorted(pop):
        p = pop[k]
        if p["profile"] == "noise" and p["severity"] in ("moderate", "severe") and p["out_w"] == "224":
            img = cv2.imread(str(repo_path(p["degraded_path"])), cv2.IMREAD_COLOR)
            if RW.decide(RW.rw_features(img, det), rw)[1] == "strong_noise":
                out["noise"] = img
                break
    return out


def feats(img, det) -> dict:
    f = RW.rw_features(img, det)
    f.update(NR.texture_features(img))
    return f


def ok_features(**kw) -> dict:
    """A feature dict that is ELIGIBLE under the sealed thresholds; tests override one signal at a time."""
    f = {"det_status": "OK", "det_score": 0.95, "yaw_proxy": 0.0, "iod_px": 80.0, "lapd1_c": 0.0015,
         "noise_sigma": 0.3, "luma_mean": 120.0, "rms_contrast": 50.0, "blockiness": 1.0, "pixelation": 0.2,
         "hf_peak_frac": 0.05, "hf_energy": 0.05, "is_gray": False}
    f.update(kw)
    return f


# ================================================================== config

def test_config_is_sealed_and_versioned(cfg):
    assert cfg["route_version"] == NR.ROUTE_VERSION and cfg["model"] == NR.MODEL
    assert cfg["model_asset"]["sha256"].startswith("329d3ab4") and "MIT" in cfg["model_asset"]["license"]
    assert cfg["rwgate_seal"] == RW.load_config()["sha256"]
    t = cfg["thresholds"]
    assert t["blur_max"] < t["clean_lapd1_lo"] and t["noise_max"] < RW.load_config()["thresholds"]["noise_min"]


def test_config_tamper_is_detected(cfg, tmp_path, monkeypatch):
    import json
    bad = dict(cfg, thresholds={**cfg["thresholds"], "noise_max": 50.0})
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(NR, "repo_path", lambda _: p)
    with pytest.raises(AssertionError, match="modified"):
        NR.load_config()


# ================================================================== input gate

def test_strong_low_noise_blur_is_eligible(dev, det, cfg):
    d = NR.decide(feats(dev["blur"], det), cfg)
    assert d.eligible and d.reason == NR.ELIGIBLE and d.route == NR.ROUTE_ELIGIBLE and d.decision == GT.ENH


def test_high_noise_blur_is_rejected(dev, det, cfg):
    rng = np.random.default_rng(0)
    grainy = np.clip(dev["blur"].astype(np.float64) + rng.normal(0, 6, dev["blur"].shape), 0, 255).astype(np.uint8)
    f = feats(grainy, det)
    assert f["lapd1_c"] <= cfg["thresholds"]["blur_max"]              # still strong blur (grain-robust measure) …
    d = NR.decide(f, cfg)
    assert not d.eligible and d.reason == NR.R_NOISE                   # … but grainy → ORIGINAL


def test_clean_input_is_rejected(dev, det, cfg):
    assert NR.decide(feats(dev["clean"], det), cfg).reason == NR.R_NOT_BLUR


def test_mild_blur_is_rejected(dev, det, cfg):
    reasons = {NR.decide(feats(cv2.GaussianBlur(dev["clean"], (0, 0), s), det), cfg).reason for s in (0.8, 1.5, 2.2)}
    assert NR.ELIGIBLE not in reasons and reasons <= {NR.R_NOT_BLUR, NR.R_MILD}
    t = cfg["thresholds"]
    mid = (t["blur_max"] + t["clean_lapd1_lo"]) / 2
    assert NR.decide(ok_features(lapd1_c=mid), cfg).reason == NR.R_MILD


def test_pure_noise_is_rejected(dev, det, cfg):
    assert not NR.decide(feats(dev["noise"], det), cfg).eligible


def test_pixelated_input_is_rejected(dev, det, cfg):
    d = NR.decide(feats(NR.synth_pixelate(dev["clean"], 3), det), cfg)
    assert d.reason == NR.R_PIX
    # a strongly blurred, then pixelated face: blur alone would admit it; the texture rule must not
    blurred_pix = NR.synth_pixelate(dev["blur"], 6)
    assert not NR.decide(feats(blurred_pix, det), cfg).eligible


def test_halftone_is_rejected_where_detectable(dev, cfg):
    t = cfg["thresholds"]
    for p in (3, 4, 6, 8):
        ht = NR.synth_halftone(dev["clean"], p)
        f = NR.texture_features(ht)
        # never eligible: a dot screen is far too sharp to count as strong blur (and YuNet rarely finds the face)
        assert RW.lapd1(ht) > t["clean_lapd1_lo"] and not NR.decide(ok_features(**f, lapd1_c=RW.lapd1(ht)), cfg).eligible
    # the dedicated rule (defence in depth; its measured sensitivity is in the calibration report): strong periodic
    # high-frequency peaks with high-frequency energy → HALFTONE even if every other signal looked eligible
    assert NR.decide(ok_features(hf_peak_frac=t["halftone_peak_max"] * 1.2, hf_energy=0.7), cfg).reason == NR.R_HALF
    assert NR.decide(ok_features(hf_peak_frac=t["halftone_peak_max"] * 1.2, hf_energy=0.01), cfg).eligible
    assert NR.texture_features(dev["blur"])["hf_energy"] < cfg["thresholds"]["halftone_energy_max"]


def test_low_face_size_is_rejected(cfg, dev, det):
    assert NR.decide(ok_features(iod_px=cfg["thresholds"]["iod_min"] - 1), cfg).reason == NR.R_SIZE
    small = cv2.resize(dev["blur"], (64, 64), interpolation=cv2.INTER_AREA)       # IOD ≈ 23 px
    assert not NR.decide(feats(small, det), cfg).eligible


@pytest.mark.parametrize("over,reason", [
    ({"det_status": "NO_FACE_DETECTED"}, NR.R_NO_FACE), ({"det_score": 0.6}, NR.R_LOW_DET),
    ({"det_score": float("nan")}, NR.R_LOW_DET), ({"yaw_proxy": 0.6}, NR.R_ALIGN),
    ({"yaw_proxy": float("nan")}, NR.R_ALIGN)])
def test_bad_detection_or_alignment_is_unsafe(cfg, over, reason):
    d = NR.decide(ok_features(**over), cfg)
    assert d.reason == reason and d.decision == GT.UNSAFE and not d.eligible


@pytest.mark.parametrize("over,reason", [
    ({"luma_mean": 20.0}, NR.R_PHOTO), ({"rms_contrast": 10.0}, NR.R_PHOTO), ({"blockiness": 3.0}, NR.R_COMP),
    ({"pixelation": 0.9}, NR.R_PIX), ({"hf_peak_frac": 0.6, "hf_energy": 0.5}, NR.R_HALF),
    ({"noise_sigma": 4.0}, NR.R_NOISE), ({"lapd1_c": 0.02}, NR.R_NOT_BLUR)])
def test_each_rule_rejects(cfg, over, reason):
    assert NR.decide(ok_features(), cfg).eligible
    assert NR.decide(ok_features(**over), cfg).reason == reason


def test_missing_feature_never_enhances(cfg):
    for k in ("lapd1_c", "noise_sigma", "pixelation", "iod_px"):
        d = NR.decide(ok_features(**{k: float("nan")}), cfg)
        assert not d.eligible and d.reason.startswith(NR.R_MISSING)


def test_gate_is_deterministic(dev, det, cfg):
    a, b = feats(dev["blur"], det), feats(dev["blur"], det)
    assert a == b and NR.decide(a, cfg) == NR.decide(b, cfg)


# ================================================================== artifact guard

def guard(inp, out, cfg, det, gray=False):
    return NR.artifact_guard(inp, out, cfg, lambda x: len(det.detect(x)) > 0, gray)


def test_normal_output_passes(dev, cfg, det):
    fn, _ = MD.fake_nafnet_ok_factory()
    g = guard(dev["blur"], fn(dev["blur"], {})[0], cfg, det)
    assert g.passed and g.reasons == [] and g.metrics["chroma_p99"] < cfg["guard"]["chroma_p99_max"]


def test_rainbow_output_fails(dev, cfg, det):
    g = guard(dev["blur"], MD.rainbow(dev["blur"]), cfg, det)
    assert not g.passed and NR.A_CHROMA in g.reasons and NR.A_HF in g.reasons


def test_high_frequency_colour_alone_fails(dev, cfg, det):
    """Fine chroma checkerboard: small per-pixel colour change (below the chroma p99 limit) but pure HF colour."""
    lab = cv2.cvtColor(dev["blur"], cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w = lab.shape[:2]
    cb = np.where((np.add.outer(np.arange(h), np.arange(w)) % 2) == 0, 1.0, -1.0) * 4.0
    lab[..., 1] += cb
    lab[..., 2] -= cb
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    g = guard(dev["blur"], out, cfg, det)
    assert NR.A_HF in g.reasons and NR.A_CHROMA not in g.reasons and not g.passed


@pytest.mark.parametrize("make,why", [
    (lambda x: x.astype(np.float32) / 255.0, "dtype"),
    (lambda x: np.full(x.shape, np.nan, np.float32), "NaN"),
    (lambda x: cv2.resize(x, (100, 100)), "shape"),
    (lambda x: np.full_like(x, 128), "near-constant"),
    (lambda x: "not an image", "not an array")])
def test_invalid_output_fails(dev, cfg, det, make, why):
    g = guard(dev["blur"], make(dev["blur"]), cfg, det)
    assert not g.passed and g.reasons == [NR.A_NUMERIC] and why in g.metrics["numeric"]


def test_channel_corruption_fails(dev, cfg, det):
    out = dev["blur"].copy()
    out[..., 2] = 0                                          # red channel lost
    g = guard(dev["blur"], out, cfg, det)
    assert not g.passed and NR.A_NUMERIC in g.reasons and "channel" in g.metrics["numeric"]


def test_output_without_face_fails(dev, cfg, det):
    noise = np.random.default_rng(3).integers(0, 255, dev["blur"].shape, np.uint8)
    g = guard(dev["blur"], noise, cfg, det)
    assert not g.passed and NR.A_FACE in g.reasons


def test_grayscale_colour_invention_fails(dev, cfg, det):
    gray = cv2.cvtColor(cv2.cvtColor(dev["blur"], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert RW.is_gray(gray)
    assert guard(gray, gray, cfg, det, gray=True).passed
    tinted = gray.astype(np.int16)
    tinted[..., 2] += 18                                     # a warm cast the input never had
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)
    g = guard(gray, tinted, cfg, det, gray=True)
    assert not g.passed and NR.A_GRAY in g.reasons


# ================================================================== pipeline routing

class Spy:
    def __init__(self, fn=None):
        self.calls, self.fn = 0, fn

    def __call__(self, img, ctx):
        self.calls += 1
        out = self.fn(img) if self.fn else cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return out, {"infer_s": 0.01}

    def alive(self):
        return True


def sharpen(img):
    return MD.fake_nafnet_ok_factory()[0](img, {})[0]


def make(mode="auto", nafnet=None):
    a4, nf = Spy(), nafnet if nafnet is not None else Spy(sharpen)
    return SP.EnhancementPipeline(a4, timeout_s=10, gate="rw", nafnet=nf, route_mode=mode), a4, nf


@pytest.mark.parametrize("mode", ["nafnet", "auto"])
def test_eligible_input_invokes_nafnet_only(dev, mode):
    pipe, a4, nf = make(mode)
    r = pipe.process(png(dev["blur"]), "face_chip")
    assert r.status == SP.ENHANCED and nf.calls == 1 and a4.calls == 0
    assert (r.used, r.model, r.route, r.artifact_check) == (NR.MODEL, NR.MODEL, NR.MODEL, "PASS")
    assert r.reason == NR.ELIGIBLE and r.bucket == NR.ROUTE_ELIGIBLE and r.decision == GT.ENH
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == dev["blur"].shape and np.array_equal(out, sharpen(dev["blur"]))
    m = r.metadata()
    assert m["output"] == "nafnet_face_chip_native" and m["nafnet"]["artifact"]["passed"] and m["route_mode"] == mode


@pytest.mark.parametrize("key", ["clean", "noise"])
def test_ineligible_input_does_not_invoke_nafnet(dev, key):
    pipe, a4, nf = make("nafnet")
    data = png(dev[key])
    r = pipe.process(data, "face_chip")
    assert r.status == SP.ORIGINAL_S and r.image == data and nf.calls == 0 and a4.calls == 0
    assert r.model is None and r.reason.startswith("NAFNET_REJECT_") and r.nafnet["eligible"] is False
    assert r.artifact_check == "NOT_RUN" and r.route == RT.ORIGINAL


def test_auto_keeps_a4_only_for_strong_noise(dev):
    pipe, a4, nf = make("auto")
    r = pipe.process(png(dev["noise"]), "face_chip")
    assert r.reason == "strong_noise" and r.used == "A4" and r.model == SP.A4_MODEL
    assert a4.calls == 1 and nf.calls == 0 and r.nafnet["reason"] == NR.R_NOISE
    c = pipe.process(png(dev["clean"]), "face_chip")              # neither route: ORIGINAL, no model at all
    assert c.status == SP.ORIGINAL_S and a4.calls == 1 and nf.calls == 0 and c.model is None


def test_a4_route_remains_independently_callable(dev):
    pipe, a4, nf = make("auto")
    r = pipe.process(png(dev["blur"]), "face_chip", route_mode="a4")       # the R5-3 rwgate: strong_blur → A4
    assert r.used == "A4" and r.model == SP.A4_MODEL and a4.calls == 1 and nf.calls == 0
    assert r.metadata()["output"] == "a4_face_chip_224" and r.nafnet is None and r.route_mode == "a4"
    f = pipe.process(png(dev["clean"]), "face_chip", force_a4=True)        # research override still works
    assert f.forced and f.used == "A4" and a4.calls == 2 and nf.calls == 0


def test_bad_route_mode_and_unconfigured_nafnet_are_rejected(dev):
    pipe, _, _ = make("auto")
    assert "route_mode" in pipe.process(png(dev["blur"]), "face_chip", route_mode="both").error
    legacy = SP.EnhancementPipeline(Spy(), timeout_s=10, gate="rw")        # no NAFNet configured
    r = legacy.process(png(dev["blur"]), "face_chip", route_mode="nafnet")
    assert r.status == SP.REJECTED and "not configured" in r.error
    with pytest.raises(AssertionError):
        SP.EnhancementPipeline(Spy(), timeout_s=10, gate="t6", nafnet=Spy(), route_mode="nafnet")


@pytest.mark.parametrize("factory,fallback", [
    ("fake_raise_factory", "MODEL_FAILURE"), ("fake_crash_factory", "RESOURCE_FAILURE"),
    ("fake_hang_factory", "TIMEOUT"), ("fake_load_fail_factory", "MODEL_FAILURE")])
def test_nafnet_failures_return_original(dev, factory, fallback):
    m = IsolatedModel(f"fes_service.models:{factory}", timeout_s=3.0, load_timeout_s=60)
    pipe, a4, _ = make("nafnet", nafnet=m)
    data = png(dev["blur"])
    try:
        r = pipe.process(data, "face_chip")
        assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == fallback
        assert r.model == NR.MODEL and r.used == RT.ORIGINAL and r.artifact_check == "NOT_RUN" and a4.calls == 0
        if fallback in ("TIMEOUT", "RESOURCE_FAILURE"):
            assert not m.alive()
            assert pipe.process(data, "face_chip").fallback == fallback and m.restarts >= 1   # recovered, retried
    finally:
        m.close()


@pytest.mark.parametrize("factory,reason", [("fake_nafnet_rainbow_factory", NR.A_CHROMA),
                                            ("fake_nafnet_bad_shape_factory", NR.A_NUMERIC)])
def test_artifact_failure_returns_original(dev, factory, reason):
    m = IsolatedModel(f"fes_service.models:{factory}", timeout_s=10)
    pipe, _, _ = make("nafnet", nafnet=m)
    data = png(dev["blur"])
    try:
        r = pipe.process(data, "face_chip")
    finally:
        m.close()
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.ARTIFACT_GUARD
    assert r.artifact_check == "FAIL" and r.model == NR.MODEL and reason in r.nafnet["artifact"]["reasons"]
    assert reason in r.error


def test_guard_exception_counts_as_failure(dev, monkeypatch):
    pipe, _, _ = make("nafnet")
    monkeypatch.setattr(NR, "artifact_guard", lambda *a, **k: 1 / 0)
    r = pipe.process(png(dev["blur"]), "face_chip")
    assert r.status == SP.ORIGINAL_S and r.artifact_check == "FAIL" and r.fallback == SP.ARTIFACT_GUARD


def test_grayscale_input_stays_grayscale(dev):
    gray = cv2.cvtColor(cv2.cvtColor(dev["blur"], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    pipe, _, nf = make("nafnet")
    r = pipe.process(png(gray), "face_chip")
    assert r.status == SP.ENHANCED and "grayscale_preserved" in r.warnings
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert RW.is_gray(out)


def test_full_image_mode_reaches_nafnet(dev):
    canvas = np.full((420, 620, 3), 127, np.uint8)
    canvas[100:324, 200:424] = dev["blur"]
    pipe, _, nf = make("nafnet")
    r = pipe.process(png(canvas), "full_image")
    assert r.n_faces == 1 and r.chip_px > 0 and r.nafnet is not None
    assert r.status in (SP.ENHANCED, SP.ORIGINAL_S) and (nf.calls == 1) == r.nafnet["eligible"]


def test_route_and_output_are_deterministic(dev):
    pipe, _, _ = make("auto")
    a, b = pipe.process(png(dev["blur"]), "face_chip"), pipe.process(png(dev["blur"]), "face_chip")
    strip = lambda m: {k: v for k, v in m.items() if k not in ("request_id", "timings_ms")}  # noqa: E731
    assert strip(a.metadata()) == strip(b.metadata()) and a.image == b.image


def test_worker_is_loaded_once_not_per_request(dev):
    m = IsolatedModel("fes_service.models:fake_nafnet_ok_factory", timeout_s=10)
    pipe, _, _ = make("nafnet", nafnet=m)
    try:
        info = m.start()
        pid = m._proc.pid
        for _ in range(3):
            assert pipe.process(png(dev["blur"]), "face_chip").status == SP.ENHANCED
        assert m._proc.pid == pid == info["pid"] and m.restarts == 0
    finally:
        m.close()


def test_validation_record_for_r5_6(dev):
    pipe, _, _ = make("auto")
    rec = pipe.process(png(dev["blur"]), "face_chip").validation_record("dev-1")
    for k in ("input_id", "route_mode", "status", "decision", "reason", "route", "model", "fallback",
              "nafnet_eligible", "nafnet_infer_s", "artifact_check", "artifact_chroma_p99",
              "artifact_hf_color_ratio", "output_valid", "q_lapd1_c", "q_noise_sigma", "q_pixelation", "total_ms"):
        assert k in rec, k
    assert rec["model"] == NR.MODEL and rec["output_valid"] and rec["nafnet_eligible"] is True
    assert not {"landmarks", "bbox", "embedding"} & set(rec)


# ================================================================== API

@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        yield c


def test_api_reports_real_nafnet_decision(client, dev):
    rd = client.get("/ready").json()
    assert rd["route_mode"] == "auto" and rd["nafnet_worker_alive"] and rd["worker_alive"]
    assert rd["nafnet_model"]["model"] == "FAKE_NAFNET_OK" and "nafnet_route" in rd["seals"]
    j = client.post("/v1/enhance", files={"image": ("x.png", png(dev["blur"]), "image/png")},
                    data={"input_mode": "face_chip"}).json()
    assert j["status"] == SP.ENHANCED and j["model"] == NR.MODEL and j["artifact_check"] == "PASS"
    assert j["nafnet"]["reason"] == NR.ELIGIBLE and j["output"] == "nafnet_face_chip_native"
    o = client.post("/v1/enhance", files={"image": ("x.png", png(dev["clean"]), "image/png")},
                    data={"input_mode": "face_chip"}).json()
    assert o["status"] == SP.ORIGINAL_S and o["model"] is None and o["reason"] == NR.R_NOT_BLUR
    a = client.post("/v1/enhance", files={"image": ("x.png", png(dev["blur"]), "image/png")},
                    data={"input_mode": "face_chip", "route_mode": "a4"}).json()
    assert a["model"] == SP.A4_MODEL and a["used"] == "A4" and a["nafnet"] is None


def test_api_survives_nafnet_load_failure(monkeypatch, dev):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_load_fail_factory")
    monkeypatch.setenv("FES_ROUTE_MODE", "a4")                 # NAFNet only on demand: loads lazily, fails
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        assert c.get("/ready").status_code == 200
        r = c.post("/v1/enhance", files={"image": ("x.png", png(dev["blur"]), "image/png")},
                   data={"input_mode": "face_chip", "route_mode": "nafnet"})
        assert r.status_code == 200 and r.json()["status"] == SP.ORIGINAL_S and r.json()["fallback"] == "MODEL_FAILURE"
        assert c.get("/health").status_code == 200


# ================================================================== real NAFNet-GoPro (torch; FES_RUN_NAFNET=1)

@REAL
def test_real_checkpoint_loads_strict_and_offline(monkeypatch):
    def no_net(*a, **k):
        raise OSError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    net, rep = MD.load_nafnet()
    assert rep["sha256"].startswith("329d3ab4") and rep["strict"] and not rep["missing_keys"]
    assert not rep["unexpected_keys"] and rep["n_model_params"] == 67888835
    assert rep["weights"] == "models/M28_nafnet/NAFNet-GoPro-width64.pth"


@REAL
def test_real_missing_or_wrong_checkpoint_fails_clearly():
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        MD.load_nafnet(weights="models/M28_nafnet/does-not-exist.pth")
    with pytest.raises(Exception, match="checksum mismatch"):
        MD.load_nafnet(sha256="0" * 64)
    with pytest.raises(Exception, match="checksum mismatch"):                    # the SIDD variant is NOT a substitute
        MD.load_nafnet(weights="models/M28_nafnet/NAFNet-SIDD-width64.pth")


@REAL
def test_real_nafnet_through_pipeline_is_deterministic_and_timed(dev):
    import time
    m = IsolatedModel("fes_service.models:nafnet_factory", timeout_s=120, threads=4)
    try:
        t0 = time.perf_counter()
        info = m.start()
        load_s = time.perf_counter() - t0
        pipe = SP.EnhancementPipeline(Spy(), timeout_s=120, gate="rw", nafnet=m, route_mode="auto")
        runs = [pipe.process(png(dev["blur"]), "face_chip") for _ in range(3)]
        pid = m._proc.pid
    finally:
        m.close()
    assert info["weights_sha256"].startswith("329d3ab4") and info["model"] == NR.MODEL
    assert all(r.model == NR.MODEL and r.status == SP.ENHANCED and r.artifact_check == "PASS" for r in runs)
    assert runs[0].image == runs[1].image == runs[2].image                       # bit-identical on CPU
    infer = [r.nafnet["infer_s"] for r in runs]
    print(f"\nNAFNet-GoPro 224px chip: load {load_s:.1f}s, infer {infer} s, pid {pid}, restarts {m.restarts}")
    assert m.restarts == 0 and max(infer) < 30
