"""R5-1 standalone service: validation, detection fallbacks, frozen routing, process-isolated A4 failures, API.

Model calls use the fake factories in fes_service.models (real A4 only with FES_RUN_A4=1, ≈ 1 min on CPU).
Requires fes-dev:cpu-p5 (FastAPI) for the API tests; they are skipped elsewhere.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os

import cv2
import numpy as np
import pytest

from fes_bench import gate as GT
from fes_bench import routing as RT
from fes_data.common import read_csv_gz, repo_path
from fes_service import pipeline as SP
from fes_service.worker import IsolatedModel

LOG = repo_path("data/manifests/phase4/t7/decision_log_p4dev.csv.gz")
POP = repo_path("data/manifests/phase4/population_p4dev.csv.gz")


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


class Spy:
    """Fake A4: returns the input face in the E-frame (so the frozen output guard finds a face)."""

    def __init__(self):
        self.calls, self.out = 0, None

    def __call__(self, img, ctx):
        self.calls += 1
        self.out = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return self.out, {}

    def alive(self):
        return True


@pytest.fixture(scope="module")
def samples() -> dict:
    """Phase 4 dev chips by frozen outcome (A4-routed NOISE / BLUR, DO_NOT_ENHANCE mild, UNSAFE)."""
    if not (LOG.exists() and POP.exists()):
        pytest.skip("phase 4 manifests missing")
    log = {r["pair_id"]: r for r in read_csv_gz(LOG)}
    pop = {r["pair_id"]: r for r in read_csv_gz(POP)}
    pick = {}
    for k in sorted(log):
        r = log[k]
        key = (f"A4_{r['bucket']}" if r["used"] == "A4" else
               ("DNE_mild" if r["decision"] == GT.DNE and pop[k]["severity"] == "mild" else
                ("UNSAFE" if r["decision"] == GT.UNSAFE else None)))
        if key and key not in pick:
            p = repo_path(pop[k]["degraded_path"])
            if not p.exists():
                pytest.skip("degraded images missing")
            pick[key] = (k, p.read_bytes(), r)
    return pick


@pytest.fixture(scope="module")
def pipe_spy():
    spy = Spy()
    return SP.EnhancementPipeline(spy, timeout_s=10), spy


# ------------------------------------------------------------------ validation → REJECTED

@pytest.mark.parametrize("data,msg", [
    (b"", "empty"), (b"not an image at all" * 10, "decodable"),
    (png(np.zeros((4, 4, 3), np.uint8)), "smaller"), (png(np.zeros((10, 200, 3), np.uint8)), "aspect")])
def test_invalid_inputs_rejected(pipe_spy, data, msg):
    pipe, spy = pipe_spy
    r = pipe.process(data)
    assert r.status == SP.REJECTED and msg in r.error and r.image is None and spy.calls == 0


def test_truncated_and_oversized_rejected(samples):
    pipe = SP.EnhancementPipeline(Spy(), timeout_s=10, limits=SP.Limits(max_bytes=50_000, max_pixels=1_000_000))
    good = samples["A4_NOISE"][1]
    assert pipe.process(good[: len(good) // 3]).status in (SP.REJECTED, SP.ORIGINAL_S)   # never ENHANCED
    assert "bytes" in pipe.process(b"\x89PNG" + b"0" * 60_000).error
    bomb = png(np.zeros((2000, 2000, 3), np.uint8))           # tiny file, 4 MP > limit: rejected from the header
    assert len(bomb) < 50_000 and "pixels" in pipe.process(bomb).error


def test_bad_mode_and_non_square_chip_rejected(pipe_spy):
    pipe, _ = pipe_spy
    assert pipe.process(png(np.zeros((64, 64, 3), np.uint8)), "nope").status == SP.REJECTED
    assert "square" in pipe.process(png(np.zeros((64, 80, 3), np.uint8)), "face_chip").error
    assert "square" in pipe.process(png(np.zeros((300, 300, 3), np.uint8)), "face_chip").error


# ------------------------------------------------------------------ safety behaviour

def test_no_face_returns_original_bytes_and_escalates(pipe_spy):
    pipe, spy = pipe_spy
    data = png(np.random.default_rng(1).integers(0, 255, (300, 400, 3), np.uint8))
    r = pipe.process(data)
    assert r.status == SP.ORIGINAL_S and r.image == data and r.escalate and r.decision == GT.UNSAFE
    assert r.reason.startswith("no_valid_face") and spy.calls == 0


@pytest.mark.parametrize("key", ["DNE_mild", "UNSAFE"])
def test_good_and_unsafe_inputs_return_original(samples, key):
    spy = Spy()
    pipe = SP.EnhancementPipeline(spy, timeout_s=10)
    k, data, logged = samples[key]
    r = pipe.process(data, "face_chip")
    assert r.status == SP.ORIGINAL_S and r.image == data and spy.calls == 0
    assert (r.decision, r.route) == (logged["decision"], logged["route"])
    assert r.escalate == (key == "UNSAFE")


@pytest.mark.parametrize("key", ["A4_NOISE", "A4_BLUR"])
def test_approved_degradation_goes_to_a4_and_matches_phase4(samples, key):
    spy = Spy()
    pipe = SP.EnhancementPipeline(spy, timeout_s=10)
    k, data, logged = samples[key]
    r = pipe.process(data, "face_chip")
    assert r.status == SP.ENHANCED and r.used == "A4" and spy.calls == 1
    assert (r.decision, r.reason, r.bucket) == (logged["decision"], logged["reason"], logged["bucket"])
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out, spy.out)


def test_full_image_mode_detects_and_processes_largest_face(samples, ref_faces):
    spy = Spy()
    pipe = SP.EnhancementPipeline(spy, timeout_s=10)
    a, b = ref_faces[0], cv2.resize(ref_faces[1], (150, 150), interpolation=cv2.INTER_AREA)
    canvas = np.full((400, 700, 3), 120, np.uint8)
    canvas[80:304, 40:264], canvas[120:270, 450:600] = a, b
    r = pipe.process(png(canvas), "full_image")
    assert r.n_faces == 2 and "multiple_faces" in r.warnings and r.status in (SP.ORIGINAL_S, SP.ENHANCED)
    assert 200 <= r.chip_px <= 224                         # the 224-px face (largest), not the 150-px one


def test_face_chip_crop_is_integer_and_native_scale(ref_faces, detector):
    from fes_data import preprocess as P
    face = cv2.resize(ref_faces[0], (160, 160), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(face, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(120, 120, 120))
    fr = P.preprocess(img, detector, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    chip, side = SP.face_chip(img, np.asarray(fr.detection.landmarks))
    assert chip.shape == (side, side, 3) and 140 <= side <= 180
    # every chip pixel is an original pixel (no resampling): find it in the source
    res = cv2.matchTemplate(img, chip, cv2.TM_SQDIFF)
    assert res.min() == 0


def test_deterministic(samples, pipe_spy):
    pipe, _ = pipe_spy
    _, data, _ = samples["A4_BLUR"]
    a, b = pipe.process(data, "face_chip"), pipe.process(data, "face_chip")
    strip = lambda m: {k: v for k, v in m.items() if k not in ("request_id", "timings_ms")}  # noqa: E731
    assert strip(a.metadata()) == strip(b.metadata()) and a.image == b.image


def test_busy_when_inflight_slots_exhausted(pipe_spy):
    pipe = SP.EnhancementPipeline(Spy(), timeout_s=10, max_inflight=1)
    pipe._slots.acquire()
    with pytest.raises(SP.Busy):
        pipe.process(b"x")
    pipe._slots.release()


def test_metadata_has_no_sensitive_fields(samples, pipe_spy):
    pipe, _ = pipe_spy
    m = pipe.process(samples["A4_NOISE"][1], "face_chip").metadata()
    assert not {"landmarks", "bbox", "det_landmarks", "embedding", "transform_2x3"} & set(m)


# ------------------------------------------------------------------ force_a4 (R5-UI research option)

def test_force_a4_overrides_route_only_and_reports_gate_verdict(samples):
    spy = Spy()
    pipe = SP.EnhancementPipeline(spy, timeout_s=10, gate="rw")
    _, data, _ = samples["DNE_mild"]
    gated = pipe.process(data, "face_chip")
    assert gated.status == SP.ORIGINAL_S and spy.calls == 0 and not gated.forced and gated.chip
    forced = pipe.process(data, "face_chip", force_a4=True)
    assert forced.status == SP.ENHANCED and forced.used == "A4" and spy.calls == 1 and forced.forced
    assert (forced.decision, forced.reason, forced.bucket) == (gated.decision, gated.reason, gated.bucket)
    assert forced.gate_route == gated.route == RT.ORIGINAL and forced.chip == gated.chip
    assert "plan" not in vars(pipe.router)                       # the override never outlives the request
    again = pipe.process(data, "face_chip")                       # next unforced request: frozen behaviour
    assert again.status == SP.ORIGINAL_S and spy.calls == 1


def test_force_a4_on_gate_approved_input_is_not_marked_forced(samples):
    pipe = SP.EnhancementPipeline(Spy(), timeout_s=10)
    r = pipe.process(samples["A4_NOISE"][1], "face_chip", force_a4=True)
    assert r.status == SP.ENHANCED and not r.forced and r.gate_route == "A4"


def test_force_a4_cannot_enhance_without_a_face(pipe_spy):
    pipe, spy = pipe_spy
    calls = spy.calls
    data = png(np.random.default_rng(2).integers(0, 255, (300, 400, 3), np.uint8))
    r = pipe.process(data, force_a4=True)
    assert r.status == SP.ORIGINAL_S and r.image == data and r.chip is None and spy.calls == calls


def test_force_a4_keeps_failure_fallbacks(samples):
    m = IsolatedModel("fes_service.models:fake_bad_output_factory", timeout_s=10)
    try:
        r = SP.EnhancementPipeline(m, timeout_s=10).process(samples["DNE_mild"][1], "face_chip", force_a4=True)
        assert r.status == SP.ORIGINAL_S and r.fallback == "INVALID_OUTPUT" and r.forced
    finally:
        m.close()


# ------------------------------------------------------------------ process-isolated model failures

@pytest.mark.parametrize("factory,fallback", [
    ("fake_raise_factory", "MODEL_FAILURE"),
    ("fake_crash_factory", "RESOURCE_FAILURE"),
    ("fake_bad_output_factory", "INVALID_OUTPUT"),
    ("fake_hang_factory", "TIMEOUT")])
def test_a4_failures_fall_back_to_original(samples, factory, fallback):
    m = IsolatedModel(f"fes_service.models:{factory}", timeout_s=3.0, load_timeout_s=60)
    pipe = SP.EnhancementPipeline(m, timeout_s=3.0)
    _, data, _ = samples["A4_NOISE"]
    try:
        r = pipe.process(data, "face_chip")
        assert r.route == "A4" and r.used == RT.ORIGINAL and r.fallback == fallback
        assert r.status == SP.ORIGINAL_S and r.image == data
        if fallback in ("TIMEOUT", "RESOURCE_FAILURE"):
            assert not m.alive()                           # hung / dead worker was killed, not abandoned
            r2 = pipe.process(data, "face_chip")           # next request restarts the worker
            assert r2.fallback == fallback and m.restarts >= 1
    finally:
        m.close()


def test_worker_success_path_and_load_failure(samples):
    m = IsolatedModel("fes_service.models:fake_ok_factory", timeout_s=10)
    try:
        r = SP.EnhancementPipeline(m, timeout_s=10).process(samples["A4_NOISE"][1], "face_chip")
        assert r.status == SP.ENHANCED and m.alive()
    finally:
        m.close()
    with pytest.raises(RuntimeError, match="load"):
        IsolatedModel("fes_service.models:fake_load_fail_factory").start()


@pytest.mark.skipif(os.environ.get("FES_RUN_A4") != "1", reason="real A4 (≈1 min CPU); set FES_RUN_A4=1")
def test_real_a4_reproduces_phase4_pixels(samples):
    k, data, _ = samples["A4_BLUR"]
    runs = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/runs/p4dev/A4.csv.gz"))}
    m = IsolatedModel("fes_service.models:a4_factory", timeout_s=300)
    try:
        r = SP.EnhancementPipeline(m, timeout_s=300).process(data, "face_chip")
    finally:
        m.close()
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert r.status == SP.ENHANCED
    assert hashlib.sha256(out.tobytes()).hexdigest() == runs[k]["output_pixel_sha256"]


# ------------------------------------------------------------------ HTTP API

@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        yield c


def test_api_health_ready_and_enhance(client, samples):
    assert client.get("/health").json()["status"] == "alive"
    rd = client.get("/ready")
    assert rd.status_code == 200 and rd.json()["routing_table"]["NOISE"] == "A4" and rd.json()["worker_alive"]
    bad = client.post("/v1/enhance", files={"image": ("x.png", b"garbage", "image/png")})
    assert bad.status_code == 422 and bad.json()["status"] == SP.REJECTED
    ok = client.post("/v1/enhance", files={"image": ("x.png", samples["A4_NOISE"][1], "image/png")},
                     data={"input_mode": "face_chip"})
    j = ok.json()
    assert ok.status_code == 200 and j["status"] == SP.ENHANCED and j["enhanced"] and j["image_format"] == "png"
    assert cv2.imdecode(np.frombuffer(base64.b64decode(j["image_b64"]), np.uint8), 1).shape == (224, 224, 3)
    orig = client.post("/v1/enhance", files={"image": ("x.png", samples["DNE_mild"][1], "image/png")},
                       data={"input_mode": "face_chip", "return_image": "false"})
    assert orig.json()["status"] == SP.ORIGINAL_S and orig.json()["image_b64"] is None
    # regression (R5-1 e2e): UNSAFE inputs carry NaN quality features → must still be valid JSON, not HTTP 500
    uns = client.post("/v1/enhance", files={"image": ("x.png", samples["UNSAFE"][1], "image/png")},
                      data={"input_mode": "face_chip"})
    assert uns.status_code == 200 and uns.json()["status"] == SP.ORIGINAL_S and uns.json()["escalate"]
    # R5-UI: chip_b64 + opt-in force_a4 (default off keeps the gate's route)
    assert orig.json()["chip_b64"] is None and orig.json()["forced"] is False       # return_image=false → no pixels
    f = client.post("/v1/enhance", files={"image": ("x.png", samples["DNE_mild"][1], "image/png")},
                    data={"input_mode": "face_chip", "force_a4": "true"}).json()
    assert f["status"] == SP.ENHANCED and f["forced"] and f["gate_route"] == RT.ORIGINAL and f["chip_b64"]


def test_api_not_ready_when_model_cannot_load(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_load_fail_factory")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        assert c.get("/ready").status_code == 503 and "load" in c.get("/ready").json()["error"]
        assert c.post("/v1/enhance", files={"image": ("x.png", b"x", "image/png")}).status_code == 503
