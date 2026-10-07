"""R5-UI demo page: served same-origin by the existing service, CSP-locked, `/ready` reports the input limits.

The page logic itself (validation, request, response interpretation, health) is tested in tests/ui/fes_ui.test.mjs;
it is run here when `node` is on PATH (it is not in the fes-dev images, so it is skipped inside Docker).
Requires fes-dev:cpu-p5 (FastAPI) for the API tests; they are skipped elsewhere.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess

import pytest

from fes_data.common import repo_path
from fes_service import pipeline as SP

UI = repo_path("src/fes_service/ui")


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    monkeypatch.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    monkeypatch.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    monkeypatch.setenv("FES_WARMUP", "0")
    monkeypatch.setitem(A.STATE, "ready", False)
    with TestClient(A.app) as c:
        yield c


def test_root_redirects_to_ui_and_page_is_served(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "Enhance Image" in page.text and "Try Another Image" in page.text
    for name, ctype in (("app.js", "javascript"), ("fes_ui.js", "javascript"), ("style.css", "text/css")):
        f = client.get(f"/ui/{name}")
        assert f.status_code == 200 and ctype in f.headers["content-type"], name


def test_ui_is_locked_to_this_origin(client):
    h = client.get("/ui/").headers
    csp = h["content-security-policy"]
    assert "default-src 'self'" in csp and "connect-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert h["x-content-type-options"] == "nosniff" and h["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in client.get("/ready", headers={"Origin": "http://evil.example"}).headers


def test_ui_has_no_third_party_or_inline_code():
    for f in UI.iterdir():
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), f"external URL in {f.name}"
        assert "localStorage" not in text and "sessionStorage" not in text and "indexedDB" not in text
    html = (UI / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline script (blocked by the CSP)"
    assert not re.search(r"\son[a-z]+=", html), "inline event handler (blocked by the CSP)"


def test_ui_calls_only_the_existing_endpoints():
    js = (UI / "app.js").read_text(encoding="utf-8") + (UI / "fes_ui.js").read_text(encoding="utf-8")
    assert set(re.findall(r'fetch\("([^"]+)"', js)) == {"/ready", "/v1/enhance"}
    assert not re.search(r"sface|dlib|feature.?match|threshold", js, re.I)


def test_ready_reports_limits_backward_compatibly(client):
    j = client.get("/ready").json()
    assert j["limits"] == dataclasses.asdict(SP.Limits())
    # the R5-1 fields are unchanged
    assert {"ready", "service_version", "error", "seals", "routing_table", "gate", "model", "worker_alive",
            "worker_restarts", "validity"} <= set(j)


def test_enhance_contract_used_by_ui_unchanged(client):
    r = client.post("/v1/enhance", files={"image": ("x.png", b"garbage", "image/png")},
                    data={"input_mode": "full_image", "return_image": "true"})
    assert r.status_code == 422 and r.json()["status"] == SP.REJECTED and r.json()["image_b64"] is None


def test_ui_defaults_match_pipeline_limits():
    js = (UI / "fes_ui.js").read_text(encoding="utf-8")
    L = SP.Limits()
    assert f"max_bytes: {L.max_bytes // (1024 * 1024)} * 1024 * 1024" in js
    assert f"max_pixels: {L.max_pixels:_}" in js and f"min_side: {L.min_side}" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH (run: node --test tests/ui/fes_ui.test.mjs)")
def test_ui_logic_node_suite():
    p = subprocess.run(["node", "--test", str(repo_path("tests/ui/fes_ui.test.mjs"))], capture_output=True, text=True,
                       timeout=120)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]


def test_compare_frames_never_crop_the_image():
    """Regression (user report): tall photos were cut off. The images must be letterboxed (contain), never clipped."""
    css = (UI / "style.css").read_text(encoding="utf-8")
    frame = re.search(r"\.frame \{([^}]*)\}", css).group(1)
    img = re.search(r"\.frame img \{([^}]*)\}", css).group(1)
    assert "overflow: hidden" not in frame and "height: 340px" in frame
    assert "object-fit: contain" in img and "width: 100%" in img and "height: 100%" in img
