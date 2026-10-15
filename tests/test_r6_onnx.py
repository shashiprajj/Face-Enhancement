"""R6-ONNX: ONNX Runtime as an opt-in inference backend for the r6 / r6_ft workers (FES_NAFNET_BACKEND=onnx).

Invariants: PyTorch stays the default and an omitted setting changes nothing; the backend is chosen only by the
explicit setting (an unknown value, onnx without an ONNX-backed profile, a missing model or a missing onnxruntime are
errors — never a silent switch); the tensor conversion equals the PyTorch path's; every ONNX failure (exception,
non-finite / wrong-shape output, timeout, crash, load failure) returns the ORIGINAL bytes with the existing fallback
codes; only the network differs (gate, stability check, guards, selection are the pipeline's). DEV images only.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

from fes_data.common import repo_path
from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from fes_service.worker import IsolatedModel
from test_r6_candidate import Model, Spy, dev, png, sharpen  # noqa: F401 — `dev` is a fixture

REAL = pytest.mark.skipif(os.environ.get("FES_RUN_NAFNET") != "1", reason="real NAFNet: set FES_RUN_NAFNET=1")
ENV = ("FES_NAFNET_BACKEND", "FES_EXPERIMENTAL_PROFILES", "FES_NAFNET_PROFILE", "FES_NAFNET_FACTORY",
       "FES_NAFNET_R6_FACTORY", "FES_NAFNET_R6_FT_FACTORY", "FES_NAFNET_R5_8_FACTORY", "FES_A4_FACTORY", "FES_GATE",
       "FES_ROUTE_MODE")


@pytest.fixture
def env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def factories(pipe):
    return {n: p.model.factory for n, p in pipe.profiles.items()}


# ------------------------------------------------------------------ selection

def test_default_backend_is_torch_and_factories_are_unchanged(env):
    from fes_service import app as A
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r5_8,r6,r6_ft")
    assert A.nafnet_backend() == "torch"
    pipe = A.build_pipeline()
    assert factories(pipe) == {"r5_6": "fes_service.models:nafnet_factory",
                               "r5_8": "fes_service.models:nafnet_factory_pad16",
                               "r6": "fes_service.models:nafnet_factory_r6",
                               "r6_ft": "fes_service.models:nafnet_factory_r6_ft"}
    env.setenv("FES_NAFNET_BACKEND", "torch")                                   # explicit torch = omitted
    assert factories(A.build_pipeline()) == factories(pipe)


def test_onnx_backend_changes_only_the_r6_workers(env):
    from fes_service import app as A
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r5_8,r6,r6_ft")
    env.setenv("FES_NAFNET_BACKEND", "onnx")
    pipe = A.build_pipeline()
    assert factories(pipe) == {"r5_6": "fes_service.models:nafnet_factory",
                               "r5_8": "fes_service.models:nafnet_factory_pad16",
                               "r6": "fes_service.models:nafnet_factory_r6_onnx",
                               "r6_ft": "fes_service.models:nafnet_factory_r6_ft_onnx"}
    assert pipe.a4.factory == "fes_service.models:a4_factory" and pipe.nafnet_profile == "r5_6"
    assert pipe.profiles["r6"].model is not pipe.profiles["r6_ft"].model          # one session per profile, no more
    assert pipe.profiles["r6"].cfg["sha256"] == NR.load_config(NR.R6_CONFIG)["sha256"]   # same sealed route config


def test_onnx_as_the_default_profile_and_explicit_factory_override(env):
    from fes_service import app as A
    env.setenv("FES_NAFNET_BACKEND", "onnx")
    env.setenv("FES_NAFNET_PROFILE", "r6")
    assert A.build_pipeline().nafnet.factory == "fes_service.models:nafnet_factory_r6_onnx"
    env.delenv("FES_NAFNET_PROFILE")
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6")
    env.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_onnx_identity_factory")
    assert factories(A.build_pipeline())["r6"] == "fes_service.models:fake_onnx_identity_factory"


@pytest.mark.parametrize("value, profiles, needle", [
    ("ONNX", "r6,r6_ft", "unknown backend"), ("tensorrt", "r6", "unknown backend"), ("", "r6", "unknown backend"),
    ("onnx", "", "no ONNX-backed profile"), ("onnx", "r5_8", "no ONNX-backed profile")])
def test_invalid_backend_configuration_fails_clearly(env, value, profiles, needle):
    from fes_service import app as A
    env.setenv("FES_NAFNET_BACKEND", value)
    env.setenv("FES_EXPERIMENTAL_PROFILES", profiles)
    with pytest.raises(ValueError, match=needle):
        A.build_pipeline()


def test_api_invalid_backend_is_not_ready_and_serves_nothing(env, dev):  # noqa: F811
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    env.setenv("FES_NAFNET_BACKEND", "onnxruntime")
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6,r6_ft")
    env.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        rd = c.get("/ready")
        assert rd.status_code == 503 and "FES_NAFNET_BACKEND" in rd.json()["error"] and not rd.json()["ready"]
        r = c.post("/v3/enhance", files={"image": ("x.png", png(dev["blur"]), "image/png")})
        assert r.status_code == 503 and r.json()["status"] == "NOT_READY"


def test_api_onnx_worker_that_cannot_load_is_not_ready(env):
    """ONNX selected but the model cannot be loaded: the service reports not-ready; it does not start on PyTorch."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    env.setenv("FES_NAFNET_BACKEND", "onnx")
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6")
    env.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    env.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    env.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_load_fail_factory")
    env.setenv("FES_WARMUP", "0")
    env.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        rd = c.get("/ready")
        assert rd.status_code == 503 and "model load failed" in rd.json()["error"]


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_api_reports_the_backend(env, dev, backend):  # noqa: F811
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    onnx = backend == "onnx"
    env.setenv("FES_NAFNET_BACKEND", backend)
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6,r6_ft")
    env.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    env.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    fake = "fake_onnx_identity_factory" if onnx else "fake_nafnet_r6_stable_factory"
    env.setenv("FES_NAFNET_R6_FACTORY", f"fes_service.models:{fake}")
    env.setenv("FES_NAFNET_R6_FT_FACTORY", f"fes_service.models:{fake}")
    env.setenv("FES_WARMUP", "0")
    env.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        rd = c.get("/ready").json()
        assert rd["ready"] and rd["nafnet_backend"] == backend and "best" in rd["profiles"]
        files = {"image": ("x.png", png(dev["blur"]), "image/png")}
        b = c.post("/v1/enhance", files=files, data={"input_mode": "face_chip", "profile": "r6"}).json()
        assert b["status"] in (SP.ENHANCED, SP.ORIGINAL_S)
        if b["nafnet"]["infer_s"] is not None:                              # the worker ran
            assert b["nafnet"].get("backend") == (MD.ONNX_BACKEND if onnx else None)
        d = c.post("/v1/enhance", files=files, data={"input_mode": "face_chip"}).json()
        assert "backend" not in d["nafnet"] and d["nafnet"]["route_version"] == NR.ROUTE_VERSION   # default unchanged


# ------------------------------------------------------------------ tensor conversion and output validation

def test_conversion_matches_the_torch_path():
    pytest.importorskip("torch")
    from fes_bench.adapters import bgr_u8_to_rgb_t, rgb_t_to_bgr_u8
    img = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    seen = {}

    def net(x):
        seen["x"] = x
        return (x * 1.3 - 0.1).astype(np.float32)                       # leaves [0, 1]: the clip must match too
    out, meta = MD.onnx_enhance(MD._FakeOrtSession(net), 384, "M")(img, {})
    t = bgr_u8_to_rgb_t(img)
    assert seen["x"].dtype == np.float32 and seen["x"].shape == (1, 3, 48, 64)
    assert np.array_equal(seen["x"], t.numpy())                           # RGB, NCHW, [0, 1]
    want = rgb_t_to_bgr_u8((t * 1.3 - 0.1).clamp(0, 1))
    assert out.dtype == np.uint8 and out.shape == img.shape and np.array_equal(out, want)
    assert meta == {"model": "M"}
    red = np.zeros((16, 16, 3), np.uint8)
    red[..., 2] = 255                                                     # BGR red → input channel 0
    MD.onnx_enhance(MD._FakeOrtSession(net), 384, "M")(red, {})
    assert seen["x"][0, 0].min() == 1.0 and seen["x"][0, 1:].max() == 0.0


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


def test_missing_onnxruntime_fails_clearly_and_does_not_use_torch(monkeypatch):
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


# ------------------------------------------------------------------ failures → ORIGINAL (real isolated worker)

def onnx_pipe(r6, r6_ft=None, timeout_s=10.0):
    exp = {"r6": IsolatedModel(f"fes_service.models:{r6}", timeout_s=timeout_s, load_timeout_s=60)}
    if r6_ft:
        exp["r6_ft"] = IsolatedModel(f"fes_service.models:{r6_ft}", timeout_s=timeout_s, load_timeout_s=60)
    pipe = SP.EnhancementPipeline(Spy(lambda x: (cv2.resize(x, (224, 224)), {})), timeout_s=10, gate="rw",
                                  nafnet=Spy(sharpen), route_mode="auto", experimental_nafnet=exp)
    return pipe, exp


@pytest.mark.parametrize("factory, fallback, timeout_s", [
    ("fake_onnx_nan_factory", "MODEL_FAILURE", 10.0), ("fake_onnx_bad_shape_factory", "MODEL_FAILURE", 10.0),
    ("fake_raise_factory", "MODEL_FAILURE", 10.0), ("fake_load_fail_factory", "MODEL_FAILURE", 10.0),
    ("fake_crash_factory", "RESOURCE_FAILURE", 10.0), ("fake_hang_factory", "TIMEOUT", 2.0)])
def test_onnx_worker_failure_returns_original(dev, factory, fallback, timeout_s):  # noqa: F811
    pipe, exp = onnx_pipe(factory, timeout_s=timeout_s)
    data = png(dev["blur"])
    try:
        r = pipe.process(data, "face_chip", profile="r6")
        assert r.status == SP.ORIGINAL_S and r.image == data and r.image_format == "as_received"
        assert r.fallback == fallback and r.used == SP.RT.ORIGINAL and r.error
        if factory in ("fake_crash_factory", "fake_hang_factory"):          # the worker is gone and comes back
            assert not exp["r6"].alive()
            r2 = pipe.process(data, "face_chip", profile="r6")
            assert r2.status == SP.ORIGINAL_S and r2.image == data and exp["r6"].restarts == 1
    finally:
        for m in exp.values():
            m.close()


def test_best_never_returns_a_failed_candidate(dev):  # noqa: F811
    data = png(dev["blur"])
    pipe, exp = onnx_pipe("fake_onnx_nan_factory", "fake_onnx_identity_factory")
    try:
        r = pipe.process(data, "face_chip", profile="best")
        assert r.selection["candidates"]["r6"]["fallback"] == "MODEL_FAILURE"
        assert r.selection["chosen"] in (None, "r6_ft")
        if r.selection["chosen"] is None:
            assert r.status == SP.ORIGINAL_S and r.image == data
        else:
            assert r.status == SP.ENHANCED and r.nafnet["backend"] == MD.ONNX_BACKEND
    finally:
        for m in exp.values():
            m.close()
    pipe, exp = onnx_pipe("fake_onnx_nan_factory", "fake_onnx_bad_shape_factory")
    try:
        r = pipe.process(data, "face_chip", profile="best")
        assert r.status == SP.ORIGINAL_S and r.image == data and r.selection["chosen"] is None
    finally:
        for m in exp.values():
            m.close()


def test_backend_is_reported_only_when_the_worker_names_one(dev):  # noqa: F811
    class Onnx(Model):
        def __call__(self, img, ctx):
            out, meta = super().__call__(img, ctx)
            return out, {**meta, "backend": MD.ONNX_BACKEND}
    pipe = SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="rw", nafnet=Spy(sharpen),
                                  route_mode="auto", experimental_nafnet={"r6": Model(sharpen), "r6_ft": Onnx(sharpen)})
    data = png(dev["blur"])
    a, b = pipe.process(data, "face_chip", profile="r6"), pipe.process(data, "face_chip", profile="r6_ft")
    assert a.status == b.status == SP.ENHANCED and "backend" not in a.nafnet and b.nafnet["backend"] == MD.ONNX_BACKEND
    ma, mb = a.metadata(), b.metadata()
    assert set(ma) == set(mb) and set(mb["nafnet"]) - set(ma["nafnet"]) == {"backend"}     # nothing else differs


# ------------------------------------------------------------------ real models: parity with PyTorch

@REAL
@pytest.mark.parametrize("profile", ["r6", "r6_ft"])
def test_real_onnx_output_matches_torch(dev, profile):  # noqa: F811
    pytest.importorskip("onnxruntime")
    if not _have_onnx_files():
        pytest.skip("ONNX exports missing")
    t_fn, t_info = (MD.nafnet_factory_r6 if profile == "r6" else MD.nafnet_factory_r6_ft)()
    o_fn, o_info = (MD.nafnet_factory_r6_onnx if profile == "r6" else MD.nafnet_factory_r6_ft_onnx)()
    assert o_info["backend"] == MD.ONNX_BACKEND and o_info["provider"] == "CPUExecutionProvider"
    assert o_info["model"] == t_info["model"] and o_info["variant"] == t_info["variant"] == "r6_consensus4"
    assert o_info["source_weights_sha256"] == t_info["weights_sha256"]     # the export of exactly these weights
    assert o_info["weights_sha256"] == MD.ONNX_MODELS[profile][1] and not os.path.isabs(o_info["weights"])
    a, ma = t_fn(dev["blur"], {})
    b, mb = o_fn(dev["blur"], {})
    assert a.shape == b.shape == dev["blur"].shape and a.dtype == b.dtype == np.uint8
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.max() <= 2 and d.mean() < 0.01, (d.max(), d.mean())            # R6-ONNX report: final outputs within 1
    cfg = NR.load_config(NR.R6_CONFIG)
    assert NR.stability_check(ma["runs_dot_c"], cfg).passed == NR.stability_check(mb["runs_dot_c"], cfg).passed
    assert mb["backend"] == MD.ONNX_BACKEND and "backend" not in ma
