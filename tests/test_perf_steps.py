"""PERF Step 1 (FES_BEST_PARALLEL=1: the two candidates of profile "best" infer at the same time) and Step 2
(FES_R6_MIRROR_BATCH=1: the four mirror passes as one batch).

Invariants: both are off by default and an omitted setting changes nothing; with Step 1 the result of a request —
status, bytes, selection, per-candidate fallbacks — is what the sequential path returns, a face the gate rejects starts
no model, and a hang / crash / error of one candidate stays that candidate's fallback; with Step 2 the four runs, their
order, the per-run tracer and the consensus output are unchanged, only how the network is called differs. DEV images.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pytest

from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from fes_service.worker import IsolatedModel
from test_r6_best import strong
from test_r6_candidate import Model, Spy, dev, png, sharpen  # noqa: F401 — `dev` is a fixture

REAL = pytest.mark.skipif(os.environ.get("FES_RUN_NAFNET") != "1", reason="real NAFNet: set FES_RUN_NAFNET=1")
ENV = ("FES_BEST_PARALLEL", "FES_R6_MIRROR_BATCH", "FES_NAFNET_BACKEND", "FES_EXPERIMENTAL_PROFILES",
       "FES_NAFNET_PROFILE", "FES_NAFNET_FACTORY", "FES_NAFNET_R6_FACTORY", "FES_NAFNET_R6_FT_FACTORY",
       "FES_A4_FACTORY", "FES_GATE", "FES_ROUTE_MODE")


@pytest.fixture
def env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


class Timed(Model):
    """Model that records when each call ran (to see whether two candidates overlapped) and may fail."""

    def __init__(self, core, delay=0.0, fail=None):
        super().__init__(core)
        self.delay, self.fail, self.spans, self.threads = delay, fail, [], []

    def __call__(self, img, ctx):
        t0 = time.perf_counter()
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        try:
            if self.fail is not None:
                raise self.fail
            return super().__call__(img, ctx)
        finally:
            self.spans.append((t0, time.perf_counter()))


def build(parallel, m6=None, ft=None):
    m6, ft = m6 or Timed(sharpen), ft or Timed(strong)
    pipe = SP.EnhancementPipeline(Spy(lambda x: (cv2.resize(x, (224, 224)), {})), timeout_s=10, gate="rw",
                                  nafnet=Spy(sharpen), route_mode="auto", experimental_nafnet={"r6": m6, "r6_ft": ft},
                                  best_parallel=parallel)
    return pipe, m6, ft


def view(r):
    """What a client can see of a result, without the per-request id and timings."""
    m = r.metadata()
    for k in ("request_id", "timings_ms"):
        m.pop(k, None)
    return m, r.image, r.image_format


# ------------------------------------------------------------------ Step 1: parallel candidates

def test_parallel_is_off_by_default():
    pipe = SP.EnhancementPipeline(Spy(lambda x: (x, {})), timeout_s=10, gate="rw", nafnet=Spy(sharpen),
                                  route_mode="auto", experimental_nafnet={"r6": Model(sharpen), "r6_ft": Model(strong)})
    assert pipe.best_parallel is False


@pytest.mark.parametrize("core6, core_ft", [(sharpen, strong), (strong, sharpen), (sharpen, MD._fake_unstable_core),
                                            (MD._fake_unstable_core, MD._fake_unstable_core)])
@pytest.mark.parametrize("image, mode", [("blur", "face_chip"), ("edge", "full_image")])
def test_parallel_result_equals_sequential(dev, core6, core_ft, image, mode):  # noqa: F811
    data = png(dev[image])
    seq, s6, sft = build(False, Timed(core6), Timed(core_ft))
    par, p6, pft = build(True, Timed(core6), Timed(core_ft))
    a, b = seq.process(data, mode, profile="best"), par.process(data, mode, profile="best")
    assert view(a) == view(b)
    assert len(s6.ctx) == len(p6.ctx) and len(sft.ctx) == len(pft.ctx)            # the same models ran, once each
    for x, y in zip(s6.ctx + sft.ctx, p6.ctx + pft.ctx):                          # on the same ctx (incl. oob mask)
        assert x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x)


def test_parallel_candidates_really_overlap_and_sequential_ones_do_not(dev):  # noqa: F811
    data = png(dev["blur"])
    for parallel in (False, True):
        pipe, m6, ft = build(parallel, Timed(sharpen, delay=0.4), Timed(strong, delay=0.4))
        t0 = time.perf_counter()
        r = pipe.process(data, "face_chip", profile="best")
        dt = time.perf_counter() - t0
        assert r.status == SP.ENHANCED and len(m6.spans) == len(ft.spans) == 1
        (a0, a1), (b0, b1) = m6.spans[0], ft.spans[0]
        overlap = min(a1, b1) - max(a0, b0)
        if parallel:
            assert overlap > 0.3 and dt < 0.75
            assert all(n.startswith("fes-best") for n in m6.threads + ft.threads)
        else:
            assert overlap <= 0 and dt > 0.8


def test_parallel_starts_no_model_for_a_face_the_gate_rejects(dev):  # noqa: F811
    plain = np.tile(np.linspace(40, 200, 160, dtype=np.uint8)[None, :, None], (160, 1, 3))    # no face: gate rejects
    for parallel in (False, True):
        pipe, m6, ft = build(parallel)
        r = pipe.process(png(plain), "face_chip", profile="best")
        assert not (r.nafnet or {}).get("eligible")
        assert r.status == SP.ORIGINAL_S and not m6.ctx and not ft.ctx and not m6.spans and not ft.spans
        assert r.selection["chosen"] is None and set(r.selection["candidates"]) == {"r6"}


@pytest.mark.parametrize("fail6, fail_ft", [
    (RuntimeError("boom"), None), (None, RuntimeError("boom")), (TimeoutError("slow"), None),
    (None, MemoryError("oom")), (RuntimeError("a"), TimeoutError("b"))])
def test_parallel_failures_stay_with_their_candidate(dev, fail6, fail_ft):  # noqa: F811
    data = png(dev["blur"])
    out = []
    for parallel in (False, True):
        pipe, m6, ft = build(parallel, Timed(sharpen, fail=fail6), Timed(strong, fail=fail_ft))
        out.append(pipe.process(data, "face_chip", profile="best"))
        assert len(m6.spans) == len(ft.spans) == 1                    # a failing candidate never stops the other
    a, b = out
    assert view(a) == view(b)
    want = lambda e: "" if e is None else "TIMEOUT" if isinstance(e, TimeoutError) else \
        "RESOURCE_FAILURE" if isinstance(e, MemoryError) else "MODEL_FAILURE"  # noqa: E731
    c = b.selection["candidates"]
    assert c["r6"]["fallback"] == want(fail6) and c["r6_ft"]["fallback"] == want(fail_ft)
    if fail6 is not None and fail_ft is not None:
        assert b.status == SP.ORIGINAL_S and b.image == data and b.selection["chosen"] is None
    else:
        assert b.status == SP.ENHANCED and b.selection["chosen"] == ("r6_ft" if fail6 is not None else "r6")


def test_parallel_does_not_touch_other_profiles(dev):  # noqa: F811
    data = png(dev["blur"])
    seq, s6, sft = build(False)
    par, p6, pft = build(True)
    for profile in (None, "r6", "r6_ft", "original"):
        assert view(seq.process(data, "face_chip", profile=profile)) == view(par.process(data, "face_chip", profile=profile))
    assert all(n == "MainThread" for n in p6.threads + pft.threads)        # no helper thread outside profile best


def test_parallel_with_real_workers_timeout_and_crash(dev):  # noqa: F811
    """Real isolated workers: one candidate hangs (killed at its own timeout) or is killed; the other is returned."""
    data = png(dev["blur"])
    for bad, fallback in (("fake_hang_factory", "TIMEOUT"), ("fake_crash_factory", "RESOURCE_FAILURE")):
        ok = IsolatedModel("fes_service.models:fake_nafnet_r6_stable_factory", timeout_s=30, load_timeout_s=60)
        ko = IsolatedModel(f"fes_service.models:{bad}", timeout_s=2.0, load_timeout_s=60)
        pipe, _, _ = build(True, ok, ko)
        try:
            ok.start(), ko.start()
            t0 = time.perf_counter()
            r = pipe.process(data, "face_chip", profile="best")
            assert time.perf_counter() - t0 < 10
            c = r.selection["candidates"]
            assert c["r6_ft"]["fallback"] == fallback and c["r6"]["fallback"] == ""
            assert r.status == SP.ENHANCED and r.selection["chosen"] == "r6" and ok.alive() and not ko.alive()
            r2 = pipe.process(data, "face_chip", profile="best")            # the dead worker restarts; still safe
            assert r2.status == SP.ENHANCED and r2.selection["chosen"] == "r6" and ko.restarts == 1
        finally:
            ok.close(), ko.close()


# ------------------------------------------------------------------ settings

@pytest.mark.parametrize("name", ["FES_BEST_PARALLEL", "FES_R6_MIRROR_BATCH"])
@pytest.mark.parametrize("value", ["true", "2", "", "yes"])
def test_invalid_flag_fails_clearly(env, name, value):
    from fes_service import app as A
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6,r6_ft")
    env.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        A.build_pipeline()


def test_settings_select_the_factories_and_default_is_unchanged(env):
    from fes_service import app as A
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6,r6_ft")
    base = A.build_pipeline()
    f = lambda p: {n: x.model.factory for n, x in p.profiles.items()}  # noqa: E731
    assert f(base) == {"r5_6": "fes_service.models:nafnet_factory", "r6": "fes_service.models:nafnet_factory_r6",
                       "r6_ft": "fes_service.models:nafnet_factory_r6_ft"} and base.best_parallel is False
    env.setenv("FES_BEST_PARALLEL", "1")
    p1 = A.build_pipeline()
    assert p1.best_parallel is True and f(p1) == f(base)                          # Step 1 changes no factory
    env.setenv("FES_R6_MIRROR_BATCH", "1")
    p2 = A.build_pipeline()
    assert f(p2) == {"r5_6": "fes_service.models:nafnet_factory", "r6": "fes_service.models:nafnet_factory_r6_batched",
                     "r6_ft": "fes_service.models:nafnet_factory_r6_ft_batched"}
    assert p2.profiles["r6"].cfg["sha256"] == base.profiles["r6"].cfg["sha256"]   # same sealed route config
    env.setenv("FES_NAFNET_BACKEND", "onnx")
    with pytest.raises(ValueError, match="cannot be combined"):
        A.build_pipeline()


def test_api_reports_the_settings(env, dev):  # noqa: F811
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fes_service import app as A
    env.setenv("FES_EXPERIMENTAL_PROFILES", "r6,r6_ft")
    env.setenv("FES_A4_FACTORY", "fes_service.models:fake_ok_factory")
    env.setenv("FES_NAFNET_FACTORY", "fes_service.models:fake_nafnet_ok_factory")
    env.setenv("FES_NAFNET_R6_FACTORY", "fes_service.models:fake_nafnet_r6_stable_factory")
    env.setenv("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:fake_nafnet_r6_stable_factory")
    env.setenv("FES_WARMUP", "0")
    files = {"image": ("x.png", png(dev["blur"]), "image/png")}
    bodies = []
    for par in ("0", "1"):
        env.setenv("FES_BEST_PARALLEL", par)
        env.setitem(A.STATE, "ready", False)
        with TestClient(A.app) as c:
            rd = c.get("/ready").json()
            assert rd["ready"] and rd["best_parallel"] is (par == "1") and rd["r6_mirror_batch"] is False
            b = c.post("/v3/enhance", files=files, data={"input_mode": "face_chip"}).json()
            for k in ("request_id", "timings_ms"):
                b.pop(k)
            b["nafnet"].pop("infer_s")
            bodies.append(b)
    assert bodies[0] == bodies[1] and bodies[0]["selection"]["chosen"] in SP.BEST_CANDIDATES


# ------------------------------------------------------------------ Step 2: batched mirror passes

def as_batch(core):
    return lambda imgs, ctx: [core(x, ctx)[0] for x in imgs]


@pytest.mark.parametrize("core", [sharpen, strong, MD._fake_unstable_core])
def test_batched_consensus_equals_the_four_single_runs(dev, core):  # noqa: F811
    x = dev["blur"]
    mask = np.zeros(x.shape[:2], bool)
    mask[:20, :30] = True
    for ctx in ({}, {"oob_mask": mask}):
        a, ma = MD.r6_consensus(core)(x, ctx)
        b, mb = MD.r6_consensus(None, as_batch(core))(x, ctx)
        assert np.array_equal(a, b) and ma == mb and len(mb["runs_dot_c"]) == len(NR.FLIPS)


def test_batch_receives_the_four_mirrors_in_order(dev):  # noqa: F811
    x, seen = dev["blur"], []

    def batch(imgs, ctx):
        seen.extend(imgs)
        return [im.copy() for im in imgs]
    MD.r6_consensus(None, batch)(x, {})
    want = [x if fl is None else cv2.flip(x, fl) for fl in NR.FLIPS]
    assert len(seen) == 4 and all(np.array_equal(a, b) for a, b in zip(seen, want))


@pytest.mark.parametrize("n", [0, 3, 5])
def test_batch_returning_another_number_of_runs_is_an_error(dev, n):  # noqa: F811
    with pytest.raises(RuntimeError, match="batched mirror runs"):
        MD.r6_consensus(None, lambda imgs, ctx: [imgs[0]] * n)(dev["blur"], {})


def test_batched_unstable_run_still_returns_original(dev):  # noqa: F811
    class Batched(Model):
        def __init__(self, core):
            self.fn, self.ctx = MD.r6_consensus(None, as_batch(core)), []
    data = png(dev["blur"])
    pipe, _, _ = build(False, Batched(sharpen), Batched(MD._fake_unstable_core))
    r = pipe.process(data, "face_chip", profile="r6_ft")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.STABILITY_GUARD
    assert pipe.process(data, "face_chip", profile="r6").status == SP.ENHANCED


@REAL
@pytest.mark.parametrize("profile", ["r6", "r6_ft"])
def test_real_batched_output_matches_sequential(dev, profile):  # noqa: F811
    seq, si = (MD.nafnet_factory_r6 if profile == "r6" else MD.nafnet_factory_r6_ft)()
    bat, bi = (MD.nafnet_factory_r6_batched if profile == "r6" else MD.nafnet_factory_r6_ft_batched)()
    assert bi["weights_sha256"] == si["weights_sha256"] and bi["model"] == si["model"] and bi["mirror_batch"] is True
    for x in (dev["blur"], cv2.resize(dev["blur"], (150, 150), interpolation=cv2.INTER_AREA)):
        a, ma = seq(x, {})
        b, mb = bat(x, {})
        d = np.abs(a.astype(int) - b.astype(int))
        assert a.shape == b.shape == x.shape and d.max() <= 2 and d.mean() < 0.01, (d.max(), d.mean())
        cfg = NR.load_config(NR.R6_CONFIG)
        assert NR.stability_check(ma["runs_dot_c"], cfg).passed == NR.stability_check(mb["runs_dot_c"], cfg).passed
