"""The enhancement pipeline with in-process models: profiles r6 / r6_ft / best, input validation, the 4-run stability
check, the guards, best-of-two selection and the parallel model start.

Invariants: every non-enhanced outcome returns the input bytes unchanged; an unstable mirror run → ORIGINAL
(STABILITY_GUARD); an accepted output keeps the input's colour; `best` returns exactly the chosen candidate's output and
never a rejected one; a face the gate rejects runs no model; metadata carries no landmarks, boxes or pixels.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from fes_service import models as MD
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from fes_service.quality import lapd1
from helpers import Model, build, chroma, decode, png, sharpen, strong


# ------------------------------------------------------------------ input validation

@pytest.mark.parametrize("data, msg", [(b"", "empty"), (b"not an image", "decodable"),
                                       (png(np.zeros((4, 4, 3), np.uint8)), "smaller"),
                                       (png(np.zeros((10, 200, 3), np.uint8)), "aspect")])
def test_invalid_inputs_rejected(data, msg):
    pipe, m6, ft = build()
    r = pipe.process(data, "full_image")
    assert r.status == SP.REJECTED and msg in r.error and r.image is None and not m6.ctx and not ft.ctx


def test_oversized_and_bad_requests_rejected():
    pipe, *_ = build(limits=SP.Limits(max_bytes=1000))
    assert "exceeds" in pipe.process(png(np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8))).error
    pipe, *_ = build()
    data = png(np.zeros((64, 64, 3), np.uint8))
    assert "unknown input_mode" in pipe.process(data, "chip").error
    assert "unknown profile" in pipe.process(data, "full_image", profile="r5_6").error
    assert "square" in pipe.process(png(np.zeros((64, 80, 3), np.uint8)), "face_chip").error


def test_no_face_returns_original_bytes_and_escalates():
    pipe, m6, ft = build()
    data = png(np.full((200, 200, 3), 90, np.uint8))
    r = pipe.process(data, "full_image")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.image_format == "as_received"
    assert r.decision == "UNSAFE_OR_UNCERTAIN" and r.reason.startswith("no_valid_face") and r.escalate
    assert not m6.ctx and not ft.ctx and r.profile == SP.DEFAULT_PROFILE == "best"


def test_busy_when_inflight_slots_exhausted():
    pipe, *_ = build(max_inflight=1)
    pipe._slots.acquire()
    with pytest.raises(SP.Busy):
        pipe.process(png(np.zeros((64, 64, 3), np.uint8)))
    pipe._slots.release()


def test_metadata_has_no_sensitive_fields(dev):
    pipe, *_ = build()
    m = pipe.process(png(dev["edge"]), "full_image").metadata()
    flat = repr(m).lower()
    assert not any(k in flat for k in ("landmark", "bbox", "embedding")) and "image" not in m and "chip" not in m


# ------------------------------------------------------------------ one candidate (r6 / r6_ft)

def test_r6_stable_is_enhanced_with_input_colour(dev):
    pipe, m6, ft = build()
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6")
    assert r.status == SP.ENHANCED and len(m6.ctx) == 1 and not ft.ctx and r.model == NR.MODEL == r.used
    assert r.nafnet["route_version"] == NR.R6_ROUTE_VERSION and r.nafnet["profile"] == "r6"
    assert r.nafnet["stability"]["passed"] and r.nafnet["stability"]["metrics"]["n_unstable"] == 0
    assert r.nafnet["artifact"]["passed"] and r.nafnet["local_artifact"]["passed"] and r.artifact_check == "PASS"
    out = decode(r.image)
    assert out.shape == dev["blur"].shape
    assert np.abs(chroma(out).astype(int) - chroma(dev["blur"]).astype(int)).max() <= 2


def test_r6_ft_runs_its_own_model_and_names_it(dev):
    pipe, m6, ft = build()
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6_ft")
    assert r.status == SP.ENHANCED and len(ft.ctx) == 1 and not m6.ctx
    assert r.model == MD.FACE_FT_MODEL and r.used == NR.MODEL and r.nafnet["profile"] == "r6_ft"


def test_unstable_run_returns_original(dev):
    pipe, *_ = build(core6=MD._fake_unstable_core)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == SP.STABILITY_GUARD
    assert r.error.startswith(NR.A_UNSTABLE) and not r.nafnet["stability"]["passed"]


def test_missing_tracer_rejects(dev):
    class NoMeta(Model):
        def __call__(self, img, ctx):
            return sharpen(img)[0], {"infer_s": 0.01}
    pipe = SP.EnhancementPipeline({"r6": NoMeta(sharpen), "r6_ft": Model(strong)})
    r = pipe.process(png(dev["blur"]), "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.fallback == SP.STABILITY_GUARD and NR.A_UNSTABLE_ERROR in r.error


def test_artifact_guard_rejects_rainbow(dev):
    def rb(img, ctx):                                     # the same streaks in every run: stable, but an artifact
        return MD.rainbow(img, amp=80, period=2.0), {}
    pipe, *_ = build(core6=rb)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.artifact_check == "FAIL"
    assert r.fallback in (SP.ARTIFACT_GUARD, SP.LOCAL_ARTIFACT_GUARD, SP.STABILITY_GUARD)


def test_model_failure_falls_back(dev):
    def boom(img, ctx):
        raise RuntimeError("injected")
    pipe, *_ = build(core6=boom)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="r6")
    assert r.status == SP.ORIGINAL_S and r.image == data and r.fallback == "MODEL_FAILURE"
    assert SP.fallback_code(TimeoutError()) == "TIMEOUT" and SP.fallback_code(MemoryError()) == "RESOURCE_FAILURE"


def test_full_image_passes_oob_mask_and_returns_black_border(dev):
    pipe, m6, _ = build()
    r = pipe.process(png(dev["edge"]), "full_image", profile="r6")
    assert r.nafnet is not None and r.nafnet["eligible"], r.reason
    assert m6.ctx and "oob_mask" in m6.ctx[0] and m6.ctx[0]["oob_mask"].any() and r.nafnet["oob_fill_frac"] > 0
    assert r.status == SP.ENHANCED
    out, chip = decode(r.image), decode(r.chip)
    assert out.shape == chip.shape and (out[m6.ctx[0]["oob_mask"]] == 0).all()


def test_greyscale_input_stays_grey(dev):
    g = cv2.cvtColor(cv2.cvtColor(dev["blur"], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    pipe, *_ = build()
    r = pipe.process(png(g), "face_chip", profile="r6")
    if r.status == SP.ENHANCED:
        assert NR.gray_chroma_p99(decode(r.image)) < 2.0 and "grayscale_preserved" in r.warnings


def test_deterministic(dev):
    pipe, *_ = build()
    a, b = (pipe.process(png(dev["edge"]), "full_image") for _ in range(2))
    assert a.status == b.status and a.image == b.image and a.quality == b.quality


# ------------------------------------------------------------------ best: one output out of two candidates

def test_select_best_prefers_the_sharper_output(dev):
    x = dev["blur"]
    a, b = sharpen(x)[0], strong(x)[0]
    assert lapd1(b) > lapd1(a) > lapd1(x)
    assert NR.select_best(x, {"r6": a, "r6_ft": b})[0] == "r6_ft"
    assert NR.select_best(x, {"r6": b, "r6_ft": a})[0] == "r6"
    name, scores = NR.select_best(x, {"r6": a, "r6_ft": a.copy()})
    assert name == "r6" and scores["r6"] == scores["r6_ft"]               # tie: the earlier candidate


@pytest.mark.parametrize("core6,core_ft,want", [(sharpen, strong, "r6_ft"), (strong, sharpen, "r6")])
def test_best_returns_the_sharper_candidate(dev, core6, core_ft, want):
    pipe, m6, ft = build(core6, core_ft)
    data = png(dev["blur"])
    r = pipe.process(data, "face_chip", profile="best")
    assert r.status == SP.ENHANCED and len(m6.ctx) == len(ft.ctx) == 1
    assert r.selection["chosen"] == want and r.selection["rule"] == NR.SELECT_RULE
    assert r.nafnet["profile"] == want and r.profile == "best" and r.metadata()["selection"] == r.selection
    c = r.selection["candidates"]
    assert set(c) == {"r6", "r6_ft"} and c[want]["score"] == max(v["score"] for v in c.values())
    assert r.image == pipe.process(data, "face_chip", profile=want).image          # exactly that profile's output
    assert r.image == pipe.process(data, "face_chip").image                        # best is the default


def test_best_never_returns_a_rejected_candidate(dev):
    pipe, *_ = build(core_ft=MD._fake_unstable_core)
    r = pipe.process(png(dev["blur"]), "face_chip", profile="best")
    assert r.status == SP.ENHANCED and r.selection["chosen"] == "r6"
    assert r.selection["candidates"]["r6_ft"] == {"status": SP.ORIGINAL_S, "fallback": SP.STABILITY_GUARD,
                                                  "model": MD.FACE_FT_MODEL, "score": None}


def test_best_all_rejected_is_original(dev):
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


def test_single_profiles_carry_no_selection(dev):
    pipe, *_ = build()
    for profile in ("r6", "r6_ft"):
        assert "selection" not in pipe.process(png(dev["blur"]), "face_chip", profile=profile).metadata()


def test_both_candidates_start_at_the_same_moment(dev):
    class Slow(Model):
        def __init__(self, core, log):
            super().__init__(core)
            self.log = log

        def __call__(self, img, ctx):
            self.log.append(("start", threading.current_thread().name, time.perf_counter()))
            time.sleep(0.3)
            return super().__call__(img, ctx)
    log = []
    pipe = SP.EnhancementPipeline({"r6": Slow(sharpen, log), "r6_ft": Slow(strong, log)})
    t0 = time.perf_counter()
    r = pipe.process(png(dev["blur"]), "face_chip", profile="best")
    assert r.status == SP.ENHANCED and len(log) == 2
    assert abs(log[0][2] - log[1][2]) < 0.2 and time.perf_counter() - t0 < 0.6 + 2.0      # overlapped, not 2 × 0.3 s


def test_pipeline_needs_both_models():
    with pytest.raises(AssertionError, match="r6, r6_ft"):
        SP.EnhancementPipeline({"r6": Model(sharpen)})
