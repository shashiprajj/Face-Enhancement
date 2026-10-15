"""R6-FT research candidate (opt-in profile "r6_ft", POST /v2/enhance): the r6 route with the face-blur fine-tuned
NAFNet weights.

Invariants: requests without `profile` and every other profile are unchanged; r6_ft is reachable only when the service
is built with it; it uses the r6 route config unchanged (same gate, stability check, guards) and reports the fine-tuned
model as the one invoked; /v2/enhance = /v1/enhance with profile r6_ft. DEV images only.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from fes_data.common import repo_path
from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from test_r6_candidate import Model, Spy, chroma, dev, png, sharpen  # noqa: F401 — `dev` is a fixture

REAL = pytest.mark.skipif(os.environ.get("FES_RUN_NAFNET") != "1", reason="real NAFNet: set FES_RUN_NAFNET=1")


def build(core=sharpen, with_ft=True):
    a4, n6, m6 = Spy(lambda x: (cv2.resize(x, (224, 224)), {})), Spy(sharpen), Model(sharpen)
    ft = Model(core) if with_ft else None
    exp = {"r6": m6, **({"r6_ft": ft} if with_ft else {})}
    pipe = SP.EnhancementPipeline(a4, timeout_s=10, gate="rw", nafnet=n6, route_mode="auto", experimental_nafnet=exp)
    return pipe, n6, m6, ft


def test_r6_ft_uses_the_r6_route_config():
    assert SP.PROFILE_CONFIG["r6_ft"] == SP.PROFILE_CONFIG["r6"] == NR.R6_CONFIG
    assert "r6_ft" in SP.EXPERIMENTAL_PROFILES and "r6_ft" in SP.REQUEST_PROFILES
    assert SP.PROFILE_MODEL["r6_ft"] == MD.FACE_FT_MODEL != NR.MODEL


def test_r6_ft_runs_its_own_model_and_names_it(dev):  # noqa: F811
    pipe, n6, m6, ft = build()
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6_ft")
    assert r.status == SP.ENHANCED and n6.calls == 0 and not m6.ctx and len(ft.ctx) == 1
    assert r.model == MD.FACE_FT_MODEL and r.used == NR.MODEL and r.metadata()["output"] == "nafnet_face_chip_native"
    assert r.nafnet["profile"] == "r6_ft" and r.nafnet["route_version"] == NR.R6_ROUTE_VERSION
    assert r.nafnet["stability"]["passed"]
    out = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(chroma(out).astype(int) - chroma(dev["blur"]).astype(int)).max() <= 2


def test_other_profiles_unchanged_with_r6_ft_enabled(dev):  # noqa: F811
    data = png(dev["blur"])
    pipe, n6, m6, ft = build()
    plain = SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="rw", nafnet=Spy(sharpen),
                                   route_mode="auto", experimental_nafnet={"r6": Model(sharpen)})
    for profile in (None, "r6"):
        a, b = pipe.process(data, "face_chip", profile=profile), plain.process(data, "face_chip", profile=profile)
        assert a.status == b.status == SP.ENHANCED and a.image == b.image and a.model == b.model == NR.MODEL
    assert not ft.ctx


def test_r6_ft_unstable_returns_original(dev):  # noqa: F811
    pipe, *_ = build(core=MD._fake_unstable_core)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r6_ft")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.STABILITY_GUARD


def test_r6_ft_not_servable_unless_enabled(dev):  # noqa: F811
    pipe, n6, m6, _ = build(with_ft=False)
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6_ft")
    assert r.status == SP.REJECTED and "not enabled" in r.error and "FES_EXPERIMENTAL_PROFILES=r6_ft" in r.error
    assert n6.calls == 0 and not m6.ctx


@pytest.mark.parametrize("enabled", [True, False])
def test_api_v2_enhance(monkeypatch, dev, enabled):  # noqa: F811
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setenv("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:fake_nafnet_r6_stable_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    monkeypatch.delenv("FES_NAFNET_PROFILE", raising=False)
    monkeypatch.setenv("FES_EXPERIMENTAL_PROFILES", "r6_ft" if enabled else "")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        rd = c.get("/ready").json()
        files = {"image": ("x.png", png(dev["blur"]), "image/png")}
        v2 = c.post("/v2/enhance", files=files, data={"input_mode": "face_chip"})
        d = c.post("/v1/enhance", files=files, data={"input_mode": "face_chip"}).json()
        assert "profile" not in d and d["nafnet"]["route_version"] == NR.ROUTE_VERSION      # /v1 default unchanged
        if not enabled:
            assert rd["profiles"] == ["original", "r5_6"]
            assert v2.status_code == 422 and "not enabled" in v2.json()["error"]
            return
        assert rd["profiles"] == ["original", "r5_6", "r6_ft"] and rd["default_profile"] == "r5_6"
        assert rd["experimental_models"]["r6_ft"]["model"] == "FAKE_NAFNET_R6_STABLE"
        b = v2.json()
        assert v2.status_code == 200 and b["status"] == SP.ENHANCED and b["profile"] == "r6_ft"
        assert b["model"] == MD.FACE_FT_MODEL and b["nafnet"]["stability"]["passed"]
        same = c.post("/v1/enhance", files=files, data={"input_mode": "face_chip", "profile": "r6_ft"}).json()
        assert same["image_b64"] == b["image_b64"]


@REAL
def test_real_fine_tuned_weights_load_and_differ_from_gopro(dev):  # noqa: F811
    if not repo_path(MD.FACE_FT_WEIGHTS).exists():
        pytest.skip("fine-tuned weights missing")
    fn, info = MD.nafnet_factory_r6_ft()
    assert info["weights"] == MD.FACE_FT_WEIGHTS and info["weights_sha256"] == MD.FACE_FT_SHA256 and info["strict"]
    assert info["model"] == MD.FACE_FT_MODEL and info["variant"] == "r6_consensus4"
    out, meta = fn(dev["blur"], {})
    ref = MD.nafnet_factory_r6()[0](dev["blur"], {})[0]
    assert out.shape == dev["blur"].shape and len(meta["runs_dot_c"]) == 4
    assert np.abs(out.astype(int) - ref.astype(int)).mean() > 0.5              # really other weights
    dc = np.abs(chroma(out).astype(int) - chroma(dev["blur"]).astype(int))     # colour = input, up to gamut clipping
    assert dc.mean() < 0.5 and dc.max() <= 6
