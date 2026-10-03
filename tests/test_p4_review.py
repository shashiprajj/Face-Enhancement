"""P4-T5 review analysis on SIMULATED responses (unit test of the arithmetic only — never evidence)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fes_data import common as C

_spec = importlib.util.spec_from_file_location("ra", C.repo_path("scripts/p4_review_analyze.py"))
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def test_kripp_alpha_known_values():
    assert ra.kripp_alpha([[1, 1], [2, 2], [3, 3]], "ordinal") == pytest.approx(1.0)
    # Krippendorff (2011) nominal example: 4 coders-free check — perfect disagreement pattern is below 0
    assert ra.kripp_alpha([[1, 2], [2, 1], [1, 2], [2, 1]], "nominal") < 0
    assert ra.kripp_alpha([[1]], "nominal") is None


def _key():
    key = []
    for c in ("B0", "A1"):
        for i in range(12):
            key.append({"item_id": f"{c}-{i}", "role": "TEST", "selection": "RANDOM", "config_id": c,
                        "pair_id": f"p{i}", "ref_id": f"r{i % 6}", "severity": "mild", "repeat_of": ""})
    for i in range(10):
        key.append({"item_id": f"s{i}", "role": "CONTROL_SAME", "selection": "CONTROL", "config_id": "", "pair_id": "",
                    "ref_id": f"r{i % 6}", "severity": "", "repeat_of": ""})
        key.append({"item_id": f"d{i}", "role": "CONTROL_DIFFERENT", "selection": "CONTROL", "config_id": "",
                    "pair_id": "", "ref_id": f"r{i % 6}", "severity": "", "repeat_of": ""})
    key.append({"item_id": "rep0", "role": "REPEAT", "selection": "REPEAT", "config_id": "B0", "pair_id": "p0",
                "ref_id": "r0", "severity": "mild", "repeat_of": "B0-0"})
    return key


def _resp(key, rng, good=True, a1_subst=True):
    out = {}
    for k in key:
        if k["role"] == "CONTROL_SAME":
            s = 5 if good else 1
        elif k["role"] == "CONTROL_DIFFERENT":
            s = 1 if good else 5
        else:
            s = 2 if (k["config_id"] == "A1" and a1_subst) else 5
        sub = "yes" if s <= 2 and k["role"] in ("TEST", "REPEAT") else "no"
        out[k["item_id"]] = {"same": str(s), "subst": sub, "invented": "major" if sub == "yes" else "none",
                             "attr": ["none"], "appear": ["none"], "artef": "none", "conf": "high"}
    return out


def test_analysis_qualification_rates_and_paired_delta():
    key, rng = _key(), np.random.default_rng(0)
    files = [{"reviewer": "R1", "n_items": len(key), "responses": _resp(key, rng)},
             {"reviewer": "R2", "n_items": len(key), "responses": _resp(key, rng)},
             {"reviewer": "BAD", "n_items": len(key), "responses": _resp(key, rng, good=False)}]
    res = ra.analyse(key, files)
    assert res["qualified"] == ["R1", "R2"] and res["status"] == "EVIDENCE"
    assert res["reviewers"]["R1"]["repeat_same_within_1"] == 1.0
    a1 = res["random_by_config"]["A1"]
    assert a1["subst_yes"][0] == 1.0 and a1["d_same_vs_B0"][0] == pytest.approx(-3.0)
    assert res["random_by_config"]["B0"]["subst_yes"][0] == 0.0
    assert res["agreement"]["alpha_same_ordinal"] == pytest.approx(1.0)


def test_single_reviewer_is_insufficient():
    key = _key()
    res = ra.analyse(key, [{"reviewer": "R1", "n_items": len(key), "responses": _resp(key, None)}])
    assert res["status"].startswith("INSUFFICIENT")
