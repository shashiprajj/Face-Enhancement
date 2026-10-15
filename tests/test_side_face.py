"""SIDE-FACE research profile "best_side": `best`, plus a detection-box framing where `best` stops on a turned face.

Invariants: a face the landmark framing accepts gives exactly the `best` answer; a turned face `best` returns
unchanged is framed by its detection box and goes through the same gate rules, models, guards and selection; the box
framing still needs a confident detection; no other profile changes. Images: the Indian-expression benchmark
(git-ignored personal data) — tests that need them are skipped when it is absent.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fes_data.common import repo_path
from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from test_r6_best import build
from test_r6_candidate import png

BENCH = "Dataset-and-code/fes_indian_expression_benchmark/degraded"
SIDE = f"{BENCH}/Anger/anger_000108/motion_blur_000/level_3.png"         # full left profile, blurred
FRONT = f"{BENCH}/Anger/anger_000020/motion_blur_000/level_3.png"        # frontal, blurred


def load(rel):
    p = repo_path(rel)
    if not p.exists():
        pytest.skip("benchmark image missing")
    return p.read_bytes()


def test_box_chip_geometry():
    img = np.arange(300 * 400 * 3, dtype=np.uint32).reshape(300, 400, 3).astype(np.uint8)
    chip, side = SP.face_chip_box(img, (100, 50, 80, 100))
    assert side == round(SP.SIDE_BOX_SCALE * 100) == chip.shape[0] == chip.shape[1]
    x0, y0 = round(140 - side / 2), round(100 - side / 2)
    assert np.array_equal(chip, img[y0:y0 + side, x0:x0 + side])            # integer cut, no resampling
    big, side = SP.face_chip_box(np.zeros((900, 900, 3), np.uint8), (100, 100, 600, 500))
    assert side == round(SP.SIDE_BOX_SCALE * 600) and big.shape[:2] == (SP.CHIP_MAX, SP.CHIP_MAX)
    edge, _ = SP.face_chip_box(np.full((200, 200, 3), 200, np.uint8), (-40, 20, 100, 100))
    assert (edge[:, 0] == 0).all() and (edge[:, -1] == 200).all()           # outside the image: black


def test_box_safety_needs_a_confident_detection():
    cfg = NR.load_config(NR.R6_CONFIG)
    f = {"det_status": "FACE_TOO_SMALL", "yaw_proxy": float("nan")}
    assert NR.decide({**f, "det_score": 0.5}, cfg, NR.box_safety).reason == NR.R_LOW_DET
    assert NR.decide({**f, "det_score": float("nan")}, cfg, NR.box_safety).reason == NR.R_NO_FACE
    assert NR.decide({**f, "det_score": 0.95}, cfg).reason == NR.R_NO_FACE                 # default rules unchanged
    assert NR.decide({**f, "det_score": 0.95}, cfg, NR.box_safety).reason.startswith(NR.R_MISSING)   # past safety


def test_side_config_is_sealed_and_only_tighter():
    side, front = NR.load_side_config(), NR.load_config(NR.R6_CONFIG)["thresholds"]
    t = side["thresholds"]
    assert t["blur_max"] == t["clean_lapd1_lo"] <= front["blur_max"]
    assert all(t[k] <= front[k] for k in NR.SIDE_TIGHTER if k in t)
    merged = {**front, **t}
    f = {"det_status": "FACE_TOO_SMALL", "det_score": 0.95, "iod_px": 80.0, "lapd1_c": t["blur_max"] / 2,
         "noise_sigma": 0.0, "luma_mean": 120.0, "rms_contrast": 40.0, "blockiness": 1.0, "pixelation": 0.1,
         "hf_peak_frac": 0.05, "hf_energy": 0.03}
    cfg = NR.load_config(NR.R6_CONFIG)
    assert NR.decide(f, {**cfg, "thresholds": merged}, NR.box_safety).eligible
    noisy = {**f, "noise_sigma": merged["noise_max"] + 1e-3}
    assert NR.decide(noisy, {**cfg, "thresholds": merged}, NR.box_safety).reason == NR.R_NOISE
    blocky = {**f, "blockiness": merged["blockiness_hi"] + 1e-3}
    assert NR.decide(blocky, {**cfg, "thresholds": merged}, NR.box_safety).reason == NR.R_COMP
    assert t["pixelation_max"] < front["pixelation_max"]                    # 1.1.0: JPEG / noise rule for the box
    grid = {**f, "pixelation": (t["pixelation_max"] + front["pixelation_max"]) / 2}
    assert NR.decide(grid, {**cfg, "thresholds": merged}, NR.box_safety).reason == NR.R_PIX
    assert NR.decide(grid, cfg, NR.box_safety).eligible                       # frontal limit alone would enhance


def test_side_face_is_enhanced_only_by_best_side():
    data = load(SIDE)
    pipe, m6, ft = build()
    r = pipe.process(data, "full_image", profile="best")
    assert r.status == SP.ORIGINAL_S and r.image == data and not m6.ctx and not ft.ctx
    s = pipe.process(data, "full_image", profile="best_side")
    assert s.profile == "best_side" and "side_face_box_framing" in s.warnings
    assert s.nafnet["framing"] == "detection_box" and s.nafnet["landmark_framing"]
    assert s.nafnet["eligible"] and len(m6.ctx) == len(ft.ctx) == 1          # same route: both candidates ran once
    assert s.status == SP.ENHANCED and s.selection["chosen"] in SP.BEST_CANDIDATES
    out = cv2.imdecode(np.frombuffer(s.image, np.uint8), cv2.IMREAD_COLOR)
    chip = cv2.imdecode(np.frombuffer(s.chip, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == chip.shape and chip.shape[0] == chip.shape[1] == s.chip_px <= SP.CHIP_MAX
    for profile in (None, "r6", "r6_ft"):                                    # nobody else frames by the box
        o = pipe.process(data, "full_image", profile=profile)
        assert o.status == SP.ORIGINAL_S and "side_face_box_framing" not in o.warnings


def test_frontal_face_best_side_equals_best():
    data = load(FRONT)
    pipe, *_ = build()
    a, b = pipe.process(data, "full_image", profile="best"), pipe.process(data, "full_image", profile="best_side")
    assert a.status == b.status == SP.ENHANCED and a.image == b.image and a.chip == b.chip
    assert "framing" not in b.nafnet and "side_face_box_framing" not in b.warnings
    assert a.selection == b.selection and a.quality == b.quality


def test_best_side_face_chip_mode_equals_best():
    pipe, *_ = build()
    data = png(np.random.default_rng(0).integers(0, 255, (160, 160, 3), dtype=np.uint8))
    a, b = pipe.process(data, "face_chip", profile="best"), pipe.process(data, "face_chip", profile="best_side")
    assert a.status == b.status == SP.ORIGINAL_S and a.reason == b.reason and "framing" not in (b.nafnet or {})


def test_best_side_needs_both_candidates():
    pipe, m6, _ = build(with_ft=False)
    r = pipe.process(png(np.zeros((64, 64, 3), np.uint8)), "full_image", profile="best_side")
    assert r.status == SP.REJECTED and "not enabled" in r.error and not m6.ctx
