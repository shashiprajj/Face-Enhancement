"""R5-9: per-request profile selection (original | r5_6 | r5_8) for the research comparison page.

Invariants: a request without `profile` behaves exactly as before; r5_6 is that same live route; r5_8 is the SAVED R5-8
candidate (identical to the process-level FES_NAFNET_PROFILE=r5_8 pipeline) and is reachable only when the service was
built with it (FES_EXPERIMENTAL_PROFILES=r5_8); original never invokes a model and returns the input bytes.
DEV images only. Requires fes-dev:cpu-p5 (FastAPI) for the API tests; they are skipped elsewhere.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytest

from fes_bench import routing as RT
from fes_data.common import read_csv_gz, repo_path
from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP

POP = repo_path("data/manifests/phase4/population_p4dev.csv.gz")
BLUR_ID = "ff-train-13457.ref-v1.gaussblur-extreme"
UI = repo_path("src/fes_service/ui")
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
def dev():
    if not POP.exists():
        pytest.skip("phase 4 manifests missing")
    r = {x["pair_id"]: x for x in read_csv_gz(POP)}[BLUR_ID]
    out = {}
    for k, p in (("blur", r["degraded_path"]), ("clean", r["ref_path"])):
        if not repo_path(p).exists():
            pytest.skip("DEV images missing")
        out[k] = cv2.imread(str(repo_path(p)), cv2.IMREAD_COLOR)
    canvas = np.full((300, 300, 3), 127, np.uint8)
    canvas[:194, :194] = out["blur"][30:, 30:]                     # full_image chip extends past the corner
    out["edge"] = canvas
    return out


def build(fn=sharpen, fn8=None, experimental=True, mode="auto"):
    a4, n6 = Spy(lambda x: cv2.resize(x, (224, 224))), Spy(fn)
    n8 = Spy(fn8 or fn) if experimental else None
    pipe = SP.EnhancementPipeline(a4, timeout_s=10, gate="rw", nafnet=n6, route_mode=mode,
                                  experimental_nafnet={"r5_8": n8} if experimental else None)
    return pipe, a4, n6, n8


def strip(m):
    return {k: v for k, v in m.items() if k not in VOLATILE}


# ------------------------------------------------------------------ default unchanged

@pytest.mark.parametrize("key,mode", [("blur", "face_chip"), ("clean", "face_chip"), ("edge", "full_image")])
def test_default_request_is_unchanged_by_the_experimental_profile(dev, key, mode):
    """A pipeline WITH r5_8 enabled answers a request without `profile` exactly like one without it (R5-6)."""
    data = png(dev[key])
    old = build(experimental=False)[0].process(data, mode)
    pipe, a4, n6, n8 = build()
    new = pipe.process(data, mode)
    assert new.image == old.image and strip(new.metadata()) == strip(old.metadata())
    assert "profile" not in new.metadata() and n8.calls == 0             # never touches the candidate worker


def test_explicit_r5_6_equals_default(dev):
    pipe, _, n6, n8 = build()
    d, r = pipe.process(png(dev["blur"]), "face_chip"), pipe.process(png(dev["blur"]), "face_chip", profile="r5_6")
    assert r.image == d.image and r.status == SP.ENHANCED and n6.calls == 2 and n8.calls == 0
    m = r.metadata()
    assert m.pop("profile") == "r5_6" and strip(m) == strip(d.metadata())
    assert r.nafnet["route_version"] == NR.ROUTE_VERSION and "local_artifact" not in r.nafnet


def test_default_build_has_no_r5_8(dev):
    pipe = build(experimental=False)[0]
    assert set(pipe.profiles) == {"r5_6"} and "nafnet_route_r5_8" not in pipe.seals
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r5_8")
    assert r.status == SP.REJECTED and "not enabled" in r.error and r.image is None


# ------------------------------------------------------------------ r5_8 = the saved candidate, per request

@pytest.mark.parametrize("key,mode", [("blur", "face_chip"), ("edge", "full_image")])
def test_r5_8_per_request_equals_saved_candidate_profile(dev, key, mode):
    data = png(dev[key])
    ref = SP.EnhancementPipeline(Spy(lambda x: x), timeout_s=10, gate="rw", nafnet=Spy(sharpen), route_mode="auto",
                                 nafnet_profile="r5_8").process(data, mode)
    pipe, a4, n6, n8 = build()
    r = pipe.process(data, mode, profile="r5_8")
    m = r.metadata()
    assert m.pop("profile") == "r5_8" and strip(m) == strip(ref.metadata()) and r.image == ref.image
    assert n6.calls == 0 and n8.calls == (1 if r.model == NR.MODEL else 0)
    assert r.nafnet["route_version"] == NR.CANDIDATE_ROUTE_VERSION and r.nafnet["profile"] == "r5_8"
    assert pipe.seals["nafnet_route_r5_8"] == NR.load_config(NR.CANDIDATE_CONFIG)["sha256"]
    assert pipe.seals["nafnet_route"] == NR.load_config()["sha256"]   # default seal still the live one


def test_r5_8_local_detector_fallback_is_reported(dev):
    def patchy(x):
        out = sharpen(x)
        out[100:106, 100:106] = MD.rainbow(x, amp=90.0, period=2.0)[100:106, 100:106]
        return out
    pipe, _, n6, n8 = build(fn8=patchy)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r5_8")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback in (SP.LOCAL_ARTIFACT_GUARD, SP.ARTIFACT_GUARD)
    assert r.model == NR.MODEL and n8.calls == 1 and n6.calls == 0
    six = pipe.process(data, "face_chip", profile="r5_6")                # r5_6 uses its own (clean) worker
    assert six.status == SP.ENHANCED and n6.calls == 1


# ------------------------------------------------------------------ original

@pytest.mark.parametrize("mode", ["full_image", "face_chip"])
def test_original_returns_input_bytes_and_invokes_nothing(dev, mode):
    pipe, a4, n6, n8 = build()
    data = png(dev["blur"])
    r = pipe.process(data, mode, profile="original")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.image_format == "as_received"
    assert (a4.calls, n6.calls, n8.calls) == (0, 0, 0) and r.model is None and r.used == RT.ORIGINAL
    assert r.reason == "profile_original" and r.nafnet is None and "total" in r.timings_ms
    jpg = cv2.imencode(".jpg", dev["clean"])[1].tobytes()
    assert pipe.process(jpg, mode, profile="original").image == jpg     # bytes, not re-encoded pixels


def test_original_still_rejects_invalid_input():
    pipe = build()[0]
    r = pipe.process(b"not an image" * 20, "full_image", profile="original")
    assert r.status == SP.REJECTED and r.image is None


# ------------------------------------------------------------------ invalid requests fail safely

@pytest.mark.parametrize("profile", ["r5_7", "R5_8", "default", "production", "a4"])
def test_unknown_profile_rejected(dev, profile):
    pipe, a4, n6, n8 = build()
    r = pipe.process(png(dev["blur"]), "face_chip", profile=profile)
    assert r.status == SP.REJECTED and "unknown profile" in r.error and (a4.calls, n6.calls, n8.calls) == (0, 0, 0)


def test_profile_with_force_a4_rejected(dev):
    pipe, a4, *_ = build()
    r = pipe.process(png(dev["blur"]), "face_chip", force_a4=True, profile="r5_8")
    assert r.status == SP.REJECTED and a4.calls == 0


def test_r5_8_cannot_be_added_to_a_non_default_setup():
    with pytest.raises(AssertionError):
        SP.EnhancementPipeline(Spy(sharpen), timeout_s=10, gate="rw", nafnet=Spy(sharpen), route_mode="auto",
                               nafnet_profile="r5_8", experimental_nafnet={"r5_8": Spy(sharpen)})
    with pytest.raises(AssertionError):
        SP.EnhancementPipeline(Spy(sharpen), timeout_s=10, gate="rw", nafnet=Spy(sharpen), route_mode="auto",
                               experimental_nafnet={"r5_7": Spy(sharpen)})


def test_nafnet_failure_on_r5_8_falls_back_with_reason(dev):
    def boom(x):
        raise RuntimeError("injected")
    pipe = build(fn8=boom)[0]
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r5_8")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == "MODEL_FAILURE"


# ------------------------------------------------------------------ API

def _client(monkeypatch, experimental: str | None):
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


def test_api_default_config_cannot_serve_r5_8(monkeypatch, dev):
    from fes_service import app as A
    with _client(monkeypatch, None) as c:
        rd = c.get("/ready").json()
        assert rd["profiles"] == ["original", "r5_6"] and rd["default_profile"] == "r5_6"
        assert "nafnet_route_r5_8" not in rd["seals"] and rd["experimental_models"] == {}
        assert set(A.STATE["pipeline"].profiles) == {"r5_6"}
        r = post(c, png(dev["blur"]), profile="r5_8")
        assert r.status_code == 422 and r.json()["status"] == SP.REJECTED and "not enabled" in r.json()["error"]
        d = post(c, png(dev["blur"])).json()
        assert d["status"] == SP.ENHANCED and "profile" not in d and d["nafnet"]["route_version"] == NR.ROUTE_VERSION


def test_api_experimental_profiles(monkeypatch, dev):
    with _client(monkeypatch, "r5_8") as c:
        rd = c.get("/ready").json()
        assert rd["profiles"] == ["original", "r5_6", "r5_8"] and rd["default_profile"] == "r5_6"
        assert rd["experimental_models"]["r5_8"]["model"] == "FAKE_NAFNET_OK"
        data = png(dev["blur"])
        o = post(c, data, profile="original").json()
        assert o["status"] == SP.ORIGINAL_S and o["model"] is None and o["profile"] == "original"
        import base64
        assert base64.b64decode(o["image_b64"]) == data
        six = post(c, data, profile="r5_6").json()
        eight = post(c, data, profile="r5_8").json()
        assert six["profile"] == "r5_6" and six["nafnet"]["route_version"] == NR.ROUTE_VERSION
        assert eight["profile"] == "r5_8" and eight["nafnet"]["route_version"] == NR.CANDIDATE_ROUTE_VERSION
        assert eight["nafnet"]["local_artifact"] is not None
        d = post(c, data).json()                                         # default stays r5_6 with r5_8 enabled
        assert "profile" not in d and d["nafnet"]["route_version"] == NR.ROUTE_VERSION
        bad = post(c, data, profile="r5_9")
        assert bad.status_code == 422 and "unknown profile" in bad.json()["error"]


def test_api_unknown_experimental_value_fails_closed(monkeypatch):
    with _client(monkeypatch, "r5_7") as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "FES_EXPERIMENTAL_PROFILES" in r.json()["error"]


# ------------------------------------------------------------------ comparison page

def test_compare_page_served_and_locked(monkeypatch):
    with _client(monkeypatch, None) as c:
        p = c.get("/ui/compare.html")
        assert p.status_code == 200 and "default-src 'self'" in p.headers["content-security-policy"]
        for name in ("compare.js", "fes_compare.js", "compare.css"):
            assert c.get(f"/ui/{name}").status_code == 200, name
        idx = c.get("/ui/").text                                         # the existing A4 demo page is kept
        assert "Force A4" in idx and "Enhance Image" in idx and 'href="compare.html"' in idx


def test_compare_page_labels_and_no_inline_code():
    html = (UI / "compare.html").read_text(encoding="utf-8")
    assert "REFERENCE — NOT PRODUCTION-APPROVED" in html and "EXPERIMENTAL — NOT PRODUCTION-APPROVED" in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and not re.search(r"\son[a-z]+=", html)
    js = (UI / "compare.js").read_text(encoding="utf-8") + (UI / "fes_compare.js").read_text(encoding="utf-8")
    assert set(re.findall(r'fetch\("([^"]+)"', js)) == {"/ready", "/v1/enhance"}
    assert not re.search(r"sface|dlib|feature.?match|threshold", js, re.I)
    text = html + js
    assert not re.search(r"\bsafe\b|certified|production[- ](safe|ready)|approved for", text, re.I)
    for m in re.finditer(r"production-approved", text, re.I):           # only ever negated
        assert re.search(r"(not|neither)\b", text[max(0, m.start() - 40):m.start()], re.I), text[m.start() - 40:m.end()]


@pytest.mark.skipif(__import__("shutil").which("node") is None,
                    reason="node not on PATH (run: node --test tests/ui/fes_compare.test.mjs)")
def test_compare_logic_node_suite():
    import subprocess
    p = subprocess.run(["node", "--test", str(repo_path("tests/ui/fes_compare.test.mjs"))], capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
