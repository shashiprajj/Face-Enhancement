"""R6-BEST (profile "best", POST /v3/enhance): 1 input -> 1 output out of the r6 and r6_ft candidates.

Invariants: both candidates run their unchanged guarded routes; the sharper accepted output (higher lapd1_c) is
returned and `selection` reports the candidates and the choice; a rejected candidate is never returned; a face the
shared gate finds not eligible stays ORIGINAL and the second candidate is not run; other profiles are unchanged; the
profile needs both candidates enabled. DEV images only.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from fes_service.rwgate import lapd1
from test_r6_candidate import Model, Spy, dev, png, sharpen  # noqa: F401 — `dev` is a fixture


def strong(img, ctx=None):
    b = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 2.5, b, -1.5, 0), {}


def build(core6=sharpen, core_ft=strong, with_ft=True):
    n6, m6 = Spy(sharpen), Model(core6)
    ft = Model(core_ft) if with_ft else None
    pipe = SP.EnhancementPipeline(Spy(lambda x: (cv2.resize(x, (224, 224)), {})), timeout_s=10, gate="rw", nafnet=n6,
                                  route_mode="auto", experimental_nafnet={"r6": m6, **({"r6_ft": ft} if with_ft else {})})
    return pipe, m6, ft


def test_select_best_prefers_the_sharper_output(dev):  # noqa: F811
    x = dev["blur"]
    a, b = sharpen(x)[0], strong(x)[0]
    assert lapd1(b) > lapd1(a) > lapd1(x)
    assert NR.select_best(x, {"r6": a, "r6_ft": b})[0] == "r6_ft"
    assert NR.select_best(x, {"r6": b, "r6_ft": a})[0] == "r6"
    name, scores = NR.select_best(x, {"r6": a, "r6_ft": a.copy()})
    assert name == "r6" and scores["r6"] == scores["r6_ft"]               # tie: the earlier candidate


@pytest.mark.parametrize("core6,core_ft,want", [(sharpen, strong, "r6_ft"), (strong, sharpen, "r6")])
def test_best_returns_the_sharper_candidate(dev, core6, core_ft, want):  # noqa: F811
    pipe, m6, ft = build(core6, core_ft)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="best")
    assert r.status == SP.ENHANCED and len(m6.ctx) == len(ft.ctx) == 1
    assert r.selection["chosen"] == want and r.selection["rule"] == NR.SELECT_RULE
    assert r.nafnet["profile"] == want and r.profile == "best" and r.metadata()["selection"] == r.selection
    c = r.selection["candidates"]
    assert set(c) == {"r6", "r6_ft"} and c[want]["score"] == max(v["score"] for v in c.values())
    assert r.image == pipe.process(data, "face_chip", profile=want).image          # exactly that profile's output


def test_best_never_returns_a_rejected_candidate(dev):  # noqa: F811
    pipe, m6, ft = build(core_ft=MD._fake_unstable_core)
    r = pipe.process(png(dev["blur"]), "face_chip", profile="best")
    assert r.status == SP.ENHANCED and r.selection["chosen"] == "r6"
    assert r.selection["candidates"]["r6_ft"] == {"status": SP.ORIGINAL_S, "fallback": SP.STABILITY_GUARD,
                                                  "model": MD.FACE_FT_MODEL, "score": None}


def test_best_all_rejected_is_original(dev):  # noqa: F811
    pipe, *_ = build(MD._fake_unstable_core, MD._fake_unstable_core)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="best")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.STABILITY_GUARD
    assert r.selection["chosen"] is None


def test_best_not_eligible_runs_no_model():
    pipe, m6, ft = build()
    data = png(np.random.default_rng(0).integers(0, 255, (160, 160, 3), dtype=np.uint8))    # no face: gate says no
    r = pipe.process(data, "face_chip", profile="best")
    assert r.status == SP.ORIGINAL_S and r.image == data and not m6.ctx and not ft.ctx
    assert r.selection["chosen"] is None and list(r.selection["candidates"]) == ["r6"]


def test_best_needs_both_candidates_and_changes_nothing_else(dev):  # noqa: F811
    data = png(dev["blur"])
    pipe, m6, _ = build(with_ft=False)
    r = pipe.process(data, "face_chip", profile="best")
    assert r.status == SP.REJECTED and "not enabled" in r.error and "r6,r6_ft" in r.error and not m6.ctx
    full, *_ = build()
    for profile in (None, "r6", "r6_ft"):
        assert "selection" not in full.process(data, "face_chip", profile=profile).metadata()


def test_api_v3_enhance(monkeypatch, dev):  # noqa: F811
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_nafnet_r6_stable_factory")
    monkeypatch.setenv("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:fake_nafnet_r6_unstable_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    monkeypatch.delenv("FES_NAFNET_PROFILE", raising=False)
    monkeypatch.setenv("FES_EXPERIMENTAL_PROFILES", "r6,r6_ft")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        assert c.get("/ready").json()["profiles"] == ["original", "r5_6", "r6", "r6_ft", "best", "best_side"]
        files = {"image": ("x.png", png(dev["blur"]), "image/png")}
        r = c.post("/v3/enhance", files=files, data={"input_mode": "face_chip"})
        b = r.json()
        assert r.status_code == 200 and b["status"] == SP.ENHANCED and b["profile"] == "best"
        assert b["selection"]["chosen"] == "r6" and b["selection"]["candidates"]["r6_ft"]["fallback"] == SP.STABILITY_GUARD
        d = c.post("/v1/enhance", files=files, data={"input_mode": "face_chip"}).json()
        assert "selection" not in d and "profile" not in d and d["nafnet"]["route_version"] == NR.ROUTE_VERSION
