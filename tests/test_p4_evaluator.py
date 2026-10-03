"""P4-T3 second evaluator E07 (dlib ResNet): pinned load, output contract, determinism, route agreement,
identity sensitivity. Runs only in fes-dev:cpu-p4 (dlib) with the git-ignored weights present."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fes_data import common as C

dlib = pytest.importorskip("dlib")
from fes_bench import evaluators as EV  # noqa: E402

if not C.repo_path(EV.DLIB_MODEL).exists():
    pytest.skip("E07 weights not acquired", allow_module_level=True)


@pytest.fixture(scope="module")
def ev():
    return EV.DlibResNet()


def test_pinned(ev):
    assert dlib.__version__ == EV.DLIB_VERSION and ev.sha256 == EV.DLIB_MODEL_SHA256


def test_contract_and_determinism(ev, ref_faces):
    a, b = ev.embed_chip(ref_faces[0]), ev.embed_chip(ref_faces[0])
    assert a.shape == (128,) and np.isfinite(a).all() and abs(np.linalg.norm(a) - 1) < 1e-9
    assert np.array_equal(a, b)
    with pytest.raises(AssertionError):
        ev.embed_chip(cv2.resize(ref_faces[0], (200, 200)))       # E-frame only on the chip route


def test_routes_agree_and_identity_separates(ev, ref_faces, detector):
    f = ref_faces[0]
    lm = np.asarray(detector.detect(f)[0].landmarks)
    assert float(ev.embed_chip(f) @ ev.embed(f, lm)) > 0.95
    same = float(ev.embed_chip(f) @ ev.embed_chip(cv2.GaussianBlur(f, (0, 0), 1.0)))
    other = float(ev.embed_chip(f) @ ev.embed_chip(ref_faces[1]))
    assert same > other
