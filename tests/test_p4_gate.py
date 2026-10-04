"""P4-T6 Do-Not-Enhance gate: fixed safety rules, NaN handling, freeze seal, labels, learner (toy data, not evidence),
and an integration check from real images through the feature extractor to a decision."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from fes_bench import gate as GT

SAFE = {"det_status": "OK", "det_score": 0.95, "yaw_proxy": 0.0}


@pytest.mark.parametrize("f,why", [
    ({"det_status": "NO_FACE_DETECTED"}, "no_valid_face"),
    ({"det_status": "FACE_TOO_SMALL", "det_score": 0.9}, "no_valid_face"),
    ({**SAFE, "det_score": 0.84}, "low_detection_confidence"),
    ({**SAFE, "det_score": math.nan}, "low_detection_confidence"),
    ({**SAFE, "yaw_proxy": 0.26}, "alignment_unreliable"),
    ({**SAFE, "yaw_proxy": math.nan}, "alignment_unreliable")])
def test_unsafe_rules_are_fixed_and_override_rules(f, why):
    rules = [GT.Rule((GT.Cond("noise_sigma", ">=", -1e9),))]
    d, reason = GT.decide({**f, "noise_sigma": 50.0}, rules)
    assert d == GT.UNSAFE and reason.startswith(why)


def test_default_is_do_not_enhance_and_rules_fire():
    rules = [GT.Rule((GT.Cond("noise_sigma", ">=", 8.0),))]
    assert GT.decide({**SAFE, "noise_sigma": 9.0}, rules) == (GT.ENH, "rule0")
    assert GT.decide({**SAFE, "noise_sigma": 1.0}, rules) == (GT.DNE, "default")
    assert GT.decide({**SAFE, "noise_sigma": math.nan}, rules) == (GT.DNE, "default")   # missing evidence never enhances
    assert GT.decide({**SAFE}, []) == (GT.DNE, "default")


def test_freeze_seal_detects_edit():
    gate = {"name": "t", "rules": [{"conds": [["noise_sigma", ">=", 8.0]]}]}
    gate["sha256"] = GT.gate_hash(gate)
    assert len(GT.load_rules(gate)) == 1
    gate["rules"][0]["conds"][0][2] = 7.0
    with pytest.raises(AssertionError):
        GT.load_rules(gate)


def test_pair_label():
    assert GT.pair_label(["CONFIRMED_BENEFIT", "CONFIRMED_HARM"]) == "WORTHY"
    assert GT.pair_label(["CONFIRMED_HARM", "PARTIAL_HARM", "DISAGREE"]) == "UNSAFE"
    assert GT.pair_label(["CONFIRMED_HARM", "NEUTRAL"]) == "NEUTRAL"
    assert GT.pair_label(["PARTIAL_BENEFIT", "CONFIRMED_HARM"]) == "NEUTRAL"


def _toy(n_refs=40, per=12, seed=0):
    rng = np.random.default_rng(seed)
    feats, labels, groups = [], [], []
    for r in range(n_refs):
        for _ in range(per):
            noise, luma = float(rng.uniform(0, 30)), float(rng.uniform(20, 120))
            feats.append({**SAFE, "noise_sigma": noise, "luma_mean": luma, "lap_norm_c": float(rng.random())})
            labels.append("WORTHY" if noise > 10 and luma > 50 else ("UNSAFE" if luma < 40 or noise < 5 else "NEUTRAL"))
            groups.append(f"r{r}")
    return feats, labels, groups


def test_learner_recovers_protective_rule_and_is_deterministic():
    feats, labels, groups = _toy()
    k, cv = GT.choose_k(feats, labels, groups)
    assert k >= 1 and cv["cv_utility_mean_by_k"][k] > cv["cv_utility_mean_by_k"][0]
    rules = GT.fit_rules(feats, labels, k)
    assert rules == GT.fit_rules(feats, labels, k) and GT.choose_k(feats, labels, groups)[0] == k
    fired = [GT.decide(f, rules)[0] == GT.ENH for f in feats]
    unsafe = sum(f and l == "UNSAFE" for f, l in zip(fired, labels))
    worthy = sum(f and l == "WORTHY" for f, l in zip(fired, labels))
    assert unsafe <= 0.05 * max(1, sum(fired)) and worthy > 0.6 * labels.count("WORTHY")


def test_no_rule_when_enhancement_never_helps():
    feats, _, groups = _toy()
    labels = ["UNSAFE" if i % 3 else "NEUTRAL" for i in range(len(feats))]
    assert GT.fit_rules(feats, labels, 4) == []
    assert GT.choose_k(feats, labels, groups)[0] == 0


def test_integration_images_to_decision(ref_faces, detector):
    """Real pipeline: image → p4-quality features → decision; deterministic; tiny input is UNSAFE."""
    from fes_bench import quality as Q
    from fes_data import degrade as D
    rules = [GT.Rule((GT.Cond("noise_sigma", ">=", 8.0),))]
    clean = Q.features(ref_faces[0], detector)
    noisy = Q.features(D.add_gaussian_noise(ref_faces[0], 20.0, np.random.default_rng(0)), detector)
    tiny = Q.features(cv2.resize(ref_faces[0], (14, 14), interpolation=cv2.INTER_AREA), detector)
    assert GT.decide(clean, rules)[0] in (GT.DNE, GT.UNSAFE)
    assert GT.decide(tiny, rules)[0] == GT.UNSAFE
    if GT.safety(noisy) is None:
        assert GT.decide(noisy, rules)[0] == GT.ENH
    assert GT.decide(Q.features(ref_faces[0], detector), rules) == GT.decide(clean, rules)
