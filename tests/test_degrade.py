"""P2-T4 generator properties: determinism, reference immutability, recorded parameters, output sizes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fes_data import degrade as D


@pytest.fixture
def ref() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return base


@pytest.mark.parametrize("profile,severity", D.all_cells())
def test_every_cell_deterministic_and_reference_untouched(ref, profile, severity):
    before = ref.copy()
    seed = D.derive_seed(1, "ref-x", profile, severity)
    a, pa = D.apply_profile(ref, profile, severity, seed)
    b, pb = D.apply_profile(ref, profile, severity, seed)
    assert np.array_equal(ref, before), "reference was modified"
    assert np.array_equal(a, b) and pa == pb
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
    assert (pa["out_h"], pa["out_w"]) == a.shape[:2]
    factor = D.PROFILES[profile][severity].get("factor")
    expected = 224 // factor if factor else 224
    assert a.shape[:2] == (expected, expected)
    json.loads(D.params_json(pa))  # serializable


def test_seed_depends_on_every_key():
    base = D.derive_seed(1, "r", "noise", "mild")
    assert len({base, D.derive_seed(2, "r", "noise", "mild"), D.derive_seed(1, "q", "noise", "mild"),
                D.derive_seed(1, "r", "jpeg", "mild"), D.derive_seed(1, "r", "noise", "severe")}) == 5


def test_severity_is_monotonic_for_blur(ref):
    errs = []
    for sev in ("mild", "moderate", "severe", "extreme"):
        out, _ = D.apply_profile(ref, "gaussblur", sev, 0)
        errs.append(float(np.abs(out.astype(int) - ref.astype(int)).mean()))
    assert errs == sorted(errs)


def test_motion_kernel_normalized():
    for length in (4, 8, 13, 21):
        for ang in (0.0, 33.3, 90.0, 135.0):
            k = D.motion_kernel(length, ang)
            assert abs(k.sum() - 1.0) < 1e-5 and (k >= 0).all()
