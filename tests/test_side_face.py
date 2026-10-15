"""Profile "best_side": `best`, plus a detection-box framing where `best` stops on a turned face.

Invariants: a face the landmark framing accepts gives exactly the `best` answer; a turned face `best` returns
unchanged is framed by its detection box and goes through the same gate rules, models, guards and selection; the box
framing still needs a confident detection. Images: Indian-expression benchmark fixtures (local only; skipped if absent).
"""

from __future__ import annotations

import numpy as np

from fes_service import nafnet_route as NR
from fes_service import pipeline as SP
from helpers import FRONT, SIDE, build, decode, fixture_bytes, png

BOX_OK = {"det_status": "FACE_TOO_SMALL", "det_score": 0.95, "iod_px": 80.0, "noise_sigma": 0.0, "luma_mean": 120.0,
          "rms_contrast": 40.0, "blockiness": 1.0, "pixelation": 0.1, "hf_peak_frac": 0.05, "hf_energy": 0.03}


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
    cfg = NR.load_config()
    f = {"det_status": "FACE_TOO_SMALL", "yaw_proxy": float("nan")}
    assert NR.decide({**f, "det_score": 0.5}, cfg, NR.box_safety).reason == NR.R_LOW_DET
    assert NR.decide({**f, "det_score": float("nan")}, cfg, NR.box_safety).reason == NR.R_NO_FACE
    assert NR.decide({**f, "det_score": 0.95}, cfg).reason == NR.R_NO_FACE                 # default rules unchanged
    assert NR.decide({**f, "det_score": 0.95}, cfg, NR.box_safety).reason.startswith(NR.R_MISSING)   # past safety


def test_side_config_is_sealed_and_only_tighter():
    side, cfg = NR.load_side_config(), NR.load_config()
    front, t = cfg["thresholds"], side["thresholds"]
    assert t["blur_max"] == t["clean_lapd1_lo"] <= front["blur_max"]
    assert all(t[k] <= front[k] for k in NR.SIDE_TIGHTER if k in t)
    merged = {**cfg, "thresholds": {**front, **t}}
    f = {**BOX_OK, "lapd1_c": t["blur_max"] / 2}
    assert NR.decide(f, merged, NR.box_safety).eligible
    assert NR.decide({**f, "noise_sigma": merged["thresholds"]["noise_max"] + 1e-3}, merged,
                     NR.box_safety).reason == NR.R_NOISE
    assert t["pixelation_max"] < front["pixelation_max"]                    # 1.1.0: JPEG / noise rule for the box
    grid = {**f, "pixelation": (t["pixelation_max"] + front["pixelation_max"]) / 2}
    assert NR.decide(grid, merged, NR.box_safety).reason == NR.R_PIX
    assert NR.decide(grid, cfg, NR.box_safety).eligible                       # frontal limit alone would enhance


def test_side_face_is_enhanced_only_by_best_side():
    data = fixture_bytes(SIDE)
    pipe, m6, ft = build()
    r = pipe.process(data, "full_image", profile="best")
    assert r.status == SP.ORIGINAL_S and r.image == data and not m6.ctx and not ft.ctx
    s = pipe.process(data, "full_image", profile="best_side")
    assert s.profile == "best_side" and "side_face_box_framing" in s.warnings
    assert s.nafnet["framing"] == "detection_box" and s.nafnet["landmark_framing"]
    assert s.nafnet["eligible"] and len(m6.ctx) == len(ft.ctx) == 1          # same route: both candidates ran once
    assert s.status == SP.ENHANCED and s.selection["chosen"] in SP.BEST_CANDIDATES
    out, chip = decode(s.image), decode(s.chip)
    assert out.shape == chip.shape and chip.shape[0] == chip.shape[1] == s.chip_px <= SP.CHIP_MAX
    for profile in ("r6", "r6_ft"):                                         # nobody else frames by the box
        o = pipe.process(data, "full_image", profile=profile)
        assert o.status == SP.ORIGINAL_S and "side_face_box_framing" not in o.warnings


def test_frontal_face_best_side_equals_best():
    data = fixture_bytes(FRONT)
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
