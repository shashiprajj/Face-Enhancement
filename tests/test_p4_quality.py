"""P4-T2 quality features: definitions, monotonicity, determinism, validity rules."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from fes_bench import quality as Q
from fes_data import degrade as D


def test_immerkaer_recovers_known_sigma():
    rng = np.random.default_rng(0)
    flat = np.full((256, 256), 128.0) + rng.normal(0, 10, (256, 256))
    assert abs(Q.noise_sigma_immerkaer(flat) - 10.0) < 0.5


def test_blockiness_validity_and_direction():
    assert math.isnan(Q.blockiness(np.zeros((28, 28))))
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur((rng.random((128, 128, 3)) * 255).astype(np.uint8), (0, 0), 3)
    y = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64)  # noqa: E731
    assert Q.blockiness(y(D.jpeg(img, 10))) > Q.blockiness(y(img)) + 0.2


def test_features_monotone_and_deterministic(ref_faces, detector):
    ref = ref_faces[0]
    f0 = Q.features(ref, detector)
    assert f0 == Q.features(ref, detector)                     # deterministic
    assert f0["det_status"] == "OK" and f0["iod_px"] > 60 and math.isnan(f0["occlusion"])
    blur = Q.features(D.gaussian_blur(ref, 3.0), detector)
    assert blur["lap_norm_c"] < f0["lap_norm_c"] and blur["hf_ratio_c"] < f0["hf_ratio_c"]
    noisy = Q.features(D.add_gaussian_noise(ref, 20.0, np.random.default_rng(0)), detector)
    assert noisy["noise_sigma"] > f0["noise_sigma"] + 10
    dark = Q.features(D.exposure(ref, -2.0), detector)
    assert dark["luma_mean"] < f0["luma_mean"] - 30
    low = Q.features(D.contrast(ref, 0.3), detector)
    assert low["rms_contrast"] < 0.5 * f0["rms_contrast"]
    tiny = Q.features(cv2.resize(ref, (14, 14), interpolation=cv2.INTER_AREA), detector)
    assert tiny["in_min_side"] == 14 and math.isnan(tiny["blockiness"])
    assert tiny["det_status"] != "OK" and math.isnan(tiny["iod_px"])     # no face → geometry NaN, never imputed


@pytest.mark.parametrize("profile", ["lowres", "gaussblur", "noise", "jpeg", "combined"])
def test_taxonomy_maps_every_profile(profile):
    from fes_data.common import load_config
    tax = load_config("configs/phase4/taxonomy_v1.yaml")
    assert profile in tax["synthetic_cell_map"]
    assert set(tax["synthetic_cell_map"][profile]) <= set(tax["families"])
