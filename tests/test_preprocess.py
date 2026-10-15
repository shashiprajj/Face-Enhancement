"""P2-T6 edge-case contract of the preprocessing pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fes_service import preprocess as P


def _canvas(h: int, w: int, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


# ---------------------------------------------------------------- pure geometry

def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 200, (5, 2))
    ang, s, t = np.radians(17), 1.7, np.array([12.0, -5.0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (s * (R @ src.T)).T + t
    M = P.umeyama_similarity(src, dst)
    np.testing.assert_allclose(M[:, :2], s * R, atol=1e-9)
    np.testing.assert_allclose(M[:, 2], t, atol=1e-9)


def test_decode_error_status(detector, pcfg):
    r = P.preprocess(None, detector, pcfg)[0]
    assert r.status == P.DECODE_ERROR and r.crop is None


# ---------------------------------------------------------------- no face: never invent one

@pytest.mark.parametrize("img", [
    _canvas(300, 300),                                                            # flat grey
    np.random.default_rng(1).integers(0, 256, (300, 300, 3), dtype=np.uint8),     # noise
])
@pytest.mark.parametrize("mode", ["full_image", "face_crop"])
def test_no_face_returns_no_crop(detector, pcfg, img, mode):
    r = P.preprocess(img, detector, pcfg, input_mode=mode)
    assert len(r) == 1 and r[0].status == P.NO_FACE and r[0].crop is None and r[0].n_faces == 0


# ---------------------------------------------------------------- one face

def test_one_face_aligned_to_template(detector, pcfg, ref_faces):
    img = ref_faces[0]
    r = P.preprocess(img, detector, pcfg)[0]
    assert r.status == P.OK and r.aligned and r.n_faces == 1
    assert r.crop.shape == (256, 256, 3)
    lm = np.asarray(r.detection.landmarks)
    M = np.asarray(r.transform_2x3)
    mapped = (M[:, :2] @ lm.T).T + M[:, 2]
    template = P.FFHQ_TEMPLATE_512 * (256 / 512)
    assert np.abs(mapped - template).max() < 12.0   # least-squares fit of 5 points
    rec = r.record()
    for key in ("status", "det_score", "det_bbox_xywh", "det_landmarks", "transform_2x3", "scale", "meta"):
        assert key in rec
    assert rec["meta"]["preproc_version"] == P.PREPROC_VERSION and rec["meta"]["detector"].startswith("yunet:")


def test_deterministic(detector, pcfg, ref_faces):
    a = P.preprocess(ref_faces[0], detector, pcfg)[0]
    b = P.preprocess(ref_faces[0], detector, pcfg)[0]
    assert np.array_equal(a.crop, b.crop) and a.record() == b.record()


# ---------------------------------------------------------------- multiple faces

def _two_faces(ref_faces):
    big = ref_faces[0]
    small = cv2.resize(ref_faces[1], (150, 150), interpolation=cv2.INTER_AREA)
    canvas = _canvas(260, 520)
    canvas[18:242, 10:234] = big
    canvas[55:205, 330:480] = small
    return canvas


def test_multiple_faces_policies(detector, ref_faces):
    img = _two_faces(ref_faces)
    largest = P.preprocess(img, detector, P.PreprocessConfig(out_size=256, selection="largest"))
    assert len(largest) == 1 and largest[0].status == P.OK and largest[0].n_faces == 2
    assert "multiple_faces" in largest[0].warnings
    assert largest[0].detection.bbox_xywh[0] < 260          # the larger (left) face was chosen
    allr = P.preprocess(img, detector, P.PreprocessConfig(out_size=256, selection="all"))
    assert len(allr) == 2 and {r.face_index for r in allr} == {0, 1}
    bad = P.preprocess(img, detector, P.PreprocessConfig(out_size=256, selection="index", face_index=5))
    assert bad[0].status == P.INVALID_INDEX and bad[0].crop is None


# ---------------------------------------------------------------- very small face

def test_very_small_face_is_not_restored(detector, ref_faces):
    tiny = cv2.resize(ref_faces[0], (28, 28), interpolation=cv2.INTER_AREA)   # eye distance ≈ 10 px
    canvas = _canvas(200, 200)
    canvas[80:108, 80:108] = tiny
    r = P.preprocess(canvas, detector, P.PreprocessConfig(out_size=256, min_eye_dist_px=12.0))[0]
    assert r.status in (P.TOO_SMALL, P.NO_FACE)
    assert r.crop is None


# ---------------------------------------------------------------- partial / occluded face

def test_partial_face_is_flagged_or_rejected(detector, pcfg, ref_faces):
    img = ref_faces[0][:, 70:]                      # left part of the face cut off
    r = P.preprocess(img, detector, pcfg)[0]
    if r.status == P.OK:
        assert "partial_face" in r.warnings and r.oob_frac > pcfg.partial_oob_warn
    else:
        assert r.crop is None


def test_occluded_face_never_crashes_and_is_traceable(detector, pcfg, ref_faces):
    img = ref_faces[0].copy()
    lm = np.asarray(P.preprocess(img, detector, pcfg)[0].detection.landmarks)
    y = int(lm[:2, 1].mean())
    img[y - 12:y + 12, :] = 0                        # black bar across the eyes (sunglasses-like)
    r = P.preprocess(img, detector, pcfg)[0]
    assert r.status in (P.OK, P.NO_FACE, "LANDMARK_FAILURE")
    if r.status == P.OK:
        assert r.detection is not None and r.transform_2x3 is not None
    else:
        assert r.crop is None


# ---------------------------------------------------------------- already-cropped face

def test_already_cropped_face(detector, pcfg, ref_faces):
    img = ref_faces[0]
    x, y, w, h = [int(v) for v in P.preprocess(img, detector, pcfg)[0].detection.bbox_xywh]
    tight = cv2.resize(img[max(y, 0):y + h, max(x, 0):x + w], (400, 400), interpolation=cv2.INTER_LINEAR)
    r = P.preprocess(tight, detector, pcfg, input_mode="face_crop")[0]
    assert r.status == P.OK and r.crop is not None
    assert r.meta["input_mode"] == "face_crop"
    # coordinates are reported in the ORIGINAL (unpadded) crop frame
    lm = np.asarray(r.detection.landmarks)
    assert (lm >= -40).all() and (lm <= 440).all()


def test_tight_crop_needs_face_crop_mode_retry(detector, pcfg, ref_faces):
    """A crop filling the frame (eyes→mouth) is missed in full_image mode; face_crop mode pads and retries."""
    img = ref_faces[0]
    lm = np.asarray(P.preprocess(img, detector, pcfg)[0].detection.landmarks)
    eye_d = np.linalg.norm(lm[1] - lm[0])
    x0, x1 = int(lm[0, 0] - 0.2 * eye_d), int(lm[1, 0] + 0.25 * eye_d)
    y0, y1 = int(lm[:2, 1].mean() - 0.2 * eye_d), int(lm[3:, 1].mean() + 0.3 * eye_d)
    tight = cv2.resize(img[max(y0, 0):y1, max(x0, 0):x1], (400, 400), interpolation=cv2.INTER_LINEAR)
    full = P.preprocess(tight, detector, pcfg, input_mode="full_image")[0]
    crop = P.preprocess(tight, detector, pcfg, input_mode="face_crop")[0]
    assert crop.status == P.OK and crop.meta["crop_retry"] is True
    assert "detected_after_crop_retry" in crop.warnings
    assert full.status in (P.NO_FACE, P.OK)   # documents that full_image may miss it; face_crop must not
