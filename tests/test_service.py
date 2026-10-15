"""The service around the pipeline: the ONNX model wrapper, the isolated worker (failures → ORIGINAL, restart), the
real ONNX exports, and the HTTP API (/health, /ready, /v1, /v2, /v3, the demo page)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from fes_service import repo_path
from fes_service.worker import IsolatedModel
from helpers import png


# ------------------------------------------------------------------ ONNX wrapper

def test_onnx_conversion_is_rgb_nchw_unit_range():
    img = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    seen = {}

    def net(x):
        seen["x"] = x
        return (x * 1.3 - 0.1).astype(np.float32)                       # leaves [0, 1]: the output must be clipped
    out, meta = MD.onnx_enhance(MD._FakeOrtSession(net), 384, "M")(img, {})
    assert seen["x"].dtype == np.float32 and seen["x"].shape == (1, 3, 48, 64)
    assert np.allclose(seen["x"][0].transpose(1, 2, 0), img[..., ::-1] / 255.0)
    want = (np.clip(img[..., ::-1] / 255.0 * 1.3 - 0.1, 0, 1) * 255.0).round().astype(np.uint8)[..., ::-1]
    assert out.dtype == np.uint8 and np.abs(out.astype(int) - want.astype(int)).max() <= 1 and meta == {"model": "M"}


@pytest.mark.parametrize("fn, needle", [
    (lambda x: np.full_like(x, np.nan), "not finite"), (lambda x: np.full_like(x, np.inf), "not finite"),
    (lambda x: x[:, :, :8], "shape"), (lambda x: x[0], "shape"), (lambda x: x.astype(np.float64), "dtype"),
    (lambda x: None, "shape")])
def test_invalid_onnx_output_is_rejected(fn, needle):
    with pytest.raises(RuntimeError, match=needle):
        MD.onnx_enhance(MD._FakeOrtSession(fn), 384, "M")(np.zeros((16, 16, 3), np.uint8), {})


@pytest.mark.parametrize("img", [np.zeros((400, 400, 3), np.uint8), np.zeros((20, 16, 3), np.uint8),
                                 np.zeros((16, 16), np.uint8), np.zeros((16, 16, 3), np.float32)])
def test_input_outside_the_exported_range_is_rejected(img):
    with pytest.raises(ValueError):
        MD.onnx_enhance(MD._FakeOrtSession(lambda x: x), 384, "M")(img, {})


def test_missing_model_file_fails_clearly():
    with pytest.raises(FileNotFoundError, match="ONNX model missing"):
        MD.load_onnx("r6", path="models/M28_nafnet/onnx/does_not_exist.onnx")


def _have_onnx_files():
    return all(repo_path(v[0]).exists() for v in MD.ONNX_MODELS.values())


def test_missing_onnxruntime_fails_clearly(monkeypatch):
    if not _have_onnx_files():
        pytest.skip("ONNX exports missing")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)                 # import onnxruntime → ImportError
    with pytest.raises(RuntimeError, match="needs onnxruntime"):
        MD.nafnet_factory_r6_onnx()


def test_wrong_hash_or_wrong_export_is_refused():
    pytest.importorskip("onnxruntime")
    if not _have_onnx_files():
        pytest.skip("ONNX exports missing")
    with pytest.raises(RuntimeError, match="SHA-256"):
        MD.load_onnx("r6", sha256="0" * 64)
    ft_file, ft_sha = MD.ONNX_MODELS["r6_ft"][:2]
    with pytest.raises(RuntimeError, match="exported from other weights"):
        MD.load_onnx("r6", path=ft_file, sha256=ft_sha)                   # right file hash, wrong model for the profile


@pytest.mark.parametrize("profile", ["r6", "r6_ft"])
def test_real_onnx_models_load_and_deblur(dev, profile):
    pytest.importorskip("onnxruntime")
    if not _have_onnx_files():
        pytest.skip("ONNX exports missing")
    fn, info = (MD.nafnet_factory_r6_onnx if profile == "r6" else MD.nafnet_factory_r6_ft_onnx)()
    assert info["backend"] == MD.ONNX_BACKEND and info["provider"] == "CPUExecutionProvider"
    assert info["weights_sha256"] == MD.ONNX_MODELS[profile][1] and info["variant"] == "r6_consensus4"
    out, meta = fn(dev["blur"], {})
    assert out.shape == dev["blur"].shape and out.dtype == np.uint8 and len(meta["runs_dot_c"]) == 4
    assert NR.stability_check(meta["runs_dot_c"], NR.load_config()).passed
    from fes_service.quality import lapd1
    assert lapd1(out) > lapd1(dev["blur"])                                # it really deblurs


# ------------------------------------------------------------------ isolated workers: failures → ORIGINAL

def worker_pipe(r6, r6_ft="fake_onnx_identity_factory", timeout_s=10.0):
    models = {n: IsolatedModel(f"fes_service.models:{f}", timeout_s=timeout_s, load_timeout_s=60)
              for n, f in (("r6", r6), ("r6_ft", r6_ft))}
    return SP.EnhancementPipeline(models), models


@pytest.mark.parametrize("factory, fallback, timeout_s", [
    ("fake_onnx_nan_factory", "MODEL_FAILURE", 10.0), ("fake_onnx_bad_shape_factory", "MODEL_FAILURE", 10.0),
    ("fake_raise_factory", "MODEL_FAILURE", 10.0), ("fake_load_fail_factory", "MODEL_FAILURE", 10.0),
    ("fake_crash_factory", "RESOURCE_FAILURE", 10.0), ("fake_hang_factory", "TIMEOUT", 2.0)])
def test_worker_failure_returns_original(dev, factory, fallback, timeout_s):
    pipe, models = worker_pipe(factory, timeout_s=timeout_s)
    data = png(dev["blur"])
    try:
        r = pipe.process(data, "face_chip", profile="r6")
        assert r.status == SP.ORIGINAL_S and r.image == data and r.image_format == "as_received"
        assert r.fallback == fallback and r.used == SP.ORIGINAL_ROUTE and r.error
        if factory in ("fake_crash_factory", "fake_hang_factory"):          # the worker is gone and comes back
            assert not models["r6"].alive()
            r2 = pipe.process(data, "face_chip", profile="r6")
            assert r2.status == SP.ORIGINAL_S and r2.image == data and models["r6"].restarts == 1
    finally:
        for m in models.values():
            m.close()


def test_best_with_real_workers_never_returns_a_failed_candidate(dev):
    data = png(dev["blur"])
    for r6_ft in ("fake_onnx_identity_factory", "fake_onnx_bad_shape_factory"):
        pipe, models = worker_pipe("fake_onnx_nan_factory", r6_ft)
        try:
            r = pipe.process(data, "face_chip", profile="best")
            assert r.selection["candidates"]["r6"]["fallback"] == "MODEL_FAILURE"
            if r.selection["chosen"] is None:
                assert r.status == SP.ORIGINAL_S and r.image == data
            else:
                assert r.selection["chosen"] == "r6_ft" and r.status == SP.ENHANCED
                assert r.nafnet["backend"] == MD.ONNX_BACKEND
        finally:
            for m in models.values():
                m.close()


# ------------------------------------------------------------------ HTTP API

@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fes_service import app as A
    monkeypatch.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_nafnet_r6_stable_factory")
    monkeypatch.setenv("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:fake_nafnet_r6_unstable_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    settings = tmp_path / "settings.json"              # validated limits, no give-up: the extreme-blur fixture qualifies
    settings.write_text('{"blur": {"enhance_from_pct": null, "give_up_above_pct": null}}', encoding="utf-8")
    monkeypatch.setenv("FES_SETTINGS", str(settings))
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        yield c


def test_api_health_ready_and_ui(client):
    assert client.get("/health").json()["status"] == "alive"
    rd = client.get("/ready").json()
    assert rd["ready"], rd["error"]
    assert rd["profiles"] == ["r6", "r6_ft", "best", "best_side"] and rd["default_profile"] == "best"
    assert rd["models"]["r6"]["model"] == "FAKE_NAFNET_R6_STABLE" and all(w["alive"] for w in rd["workers"].values())
    assert set(rd["seals"]) == {"nafnet_route", "side_face_route", "blur_scale"} and rd["limits"]["max_bytes"] > 0
    for path in ("/", "/ui", "/ui/"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == "/ui/best.html"
    page = client.get("/ui/best.html")
    assert page.status_code == 200 and "Content-Security-Policy" in page.headers
    assert client.get("/ui/compare.html").status_code == 404 and client.get("/ui/index.html").status_code == 404


def test_api_endpoints_and_profiles(client, dev):
    files = {"image": ("x.png", png(dev["blur"]), "image/png")}
    post = lambda path, **f: client.post(path, files=files, data={"input_mode": "face_chip", **f})  # noqa: E731
    v3 = post("/v3/enhance").json()
    assert v3["status"] == SP.ENHANCED and v3["profile"] == "best" and v3["selection"]["chosen"] == "r6"
    assert v3["selection"]["candidates"]["r6_ft"]["fallback"] == SP.STABILITY_GUARD and v3["image_b64"]
    v1 = post("/v1/enhance").json()
    assert v1["profile"] == "best" and v1["image_b64"] == v3["image_b64"]       # /v1 default = best
    v2 = post("/v2/enhance").json()
    assert v2["profile"] == "r6_ft" and v2["status"] == SP.ORIGINAL_S and v2["fallback"] == SP.STABILITY_GUARD
    r6 = post("/v1/enhance", profile="r6").json()
    assert r6["status"] == SP.ENHANCED and "selection" not in r6
    bad = post("/v1/enhance", profile="r5_6")
    assert bad.status_code == 422 and "unknown profile" in bad.json()["error"]
    lean = post("/v3/enhance", return_image="false").json()
    assert lean["image_b64"] is None and lean["chip_b64"] is None


def test_api_not_ready_when_a_model_cannot_load(monkeypatch, dev):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fes_service import app as A
    monkeypatch.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_nafnet_r6_stable_factory")
    monkeypatch.setenv("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:fake_load_fail_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        rd = c.get("/ready")
        assert rd.status_code == 503 and "injected load failure" in rd.json()["error"]
        r = c.post("/v3/enhance", files={"image": ("x.png", png(dev["blur"]), "image/png")})
        assert r.status_code == 503 and r.json()["status"] == "NOT_READY"
