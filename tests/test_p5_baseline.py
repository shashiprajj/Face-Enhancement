"""P5-T1 baseline regression: the frozen Phase 4 behaviour (T6 gate + T7 routing) must not drift during Phase 5.

Uses the tracked Phase 4 quality features / metrics (synthetic `main` test and `p4dev` dev) and the sealed manifest
configs/phase5/baseline_manifest.json. Adapters are spies — no model is run here (A4 pixel reproduction is checked by
scripts/p5_baseline_freeze.py --write --a4-check N).
"""

from __future__ import annotations

import copy
import importlib.util
import json

import numpy as np
import pytest

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import routing as RT
from fes_data.common import read_csv_gz, repo_path

_spec = importlib.util.spec_from_file_location("p5_baseline_freeze", repo_path("scripts/p5_baseline_freeze.py"))
BF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BF)

MANIFEST = json.loads(repo_path(BF.MANIFEST).read_text(encoding="utf-8"))
G6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
R7 = json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))
RULES = GT.load_rules(G6)
THR, TABLE = RT.load_routing(R7, G6)
ORIG = np.full((224, 224, 3), 100, np.uint8)
IMG = np.full((56, 56, 3), 100, np.uint8)
GOOD = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def pairs() -> dict:
    return {pop: E.load(pop) for pop in ("main", "p4dev")}


# ------------------------------------------------------------------ manifest / freeze integrity

def test_manifest_seal_and_frozen_files_unchanged():
    assert MANIFEST["sha256"] == BF.seal(MANIFEST), "baseline manifest edited after freeze"
    for p, v in MANIFEST["frozen_configs"].items():
        assert BF.text_sha256(p) == v["sha256_text"], f"frozen config changed: {p}"
    for p, h in MANIFEST["code_sha256_text"].items():
        assert BF.text_sha256(p) == h, f"frozen baseline code changed: {p} (bump the baseline manifest instead)"


def test_manifest_matches_loaded_seals_and_versions():
    assert MANIFEST["gate"]["seal"] == G6["sha256"] == R7["t6_gate_seal"]
    assert MANIFEST["routing"]["seal"] == R7["sha256"]
    assert MANIFEST["routing"]["table"] == TABLE and MANIFEST["routing"]["bucket_thresholds"] == THR
    assert MANIFEST["versions"]["gate"] == GT.GATE_VERSION and MANIFEST["versions"]["routing"] == RT.ROUTING_VERSION
    assert all(v["identical"] for v in MANIFEST["reproduction"].values())
    assert all(v["verified"] for v in MANIFEST["assets"].values())
    assert MANIFEST["a4_rerun"]["all_identical"] and MANIFEST["a4_rerun"]["n"] >= 1


def test_tampered_routing_is_rejected():
    bad = copy.deepcopy(R7)
    bad["table"]["PHOTOMETRIC"] = "A1"
    with pytest.raises(AssertionError):
        RT.load_routing(bad, G6)
    bad = copy.deepcopy(G6)
    bad["rules"][0]["conds"][0][2] = 0.0
    with pytest.raises(AssertionError):
        GT.load_rules(bad)


# ------------------------------------------------------------------ frozen routing contract

def test_only_a4_is_routed_and_only_for_noise_blur():
    assert set(TABLE.values()) <= {RT.ORIGINAL, "A4"}
    assert not set(TABLE.values()) & set(BF.NOT_ROUTABLE)
    assert {b for b, c in TABLE.items() if c == "A4"} == {"NOISE", "BLUR"}
    assert MANIFEST["routing"]["routed_candidates"] == ["A4"]


def test_replay_equals_committed_phase4_logs(pairs):
    """Decisions/buckets/routes recomputed now equal the committed T7 decision logs, row by row."""
    for pop in ("main", "p4dev"):
        _, t7, _ = BF.replay_rows(pop, RULES, THR, TABLE)
        committed = read_csv_gz(repo_path(BF.REPLAY_FILES[("T7", pop)]))
        keys = ("pair_id", "decision", "reason", "bucket", "route", "used", "fallback")
        assert [tuple(str(r[k]) for k in keys) for r in t7] == [tuple(r[k] for k in keys) for r in committed]


# ------------------------------------------------------------------ behavioural regressions (spy adapters)

class Spy:
    def __init__(self, out=GOOD, exc=None):
        self.calls, self.out, self.exc = 0, out, exc

    def __call__(self, img, ctx):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.out, {}


def _router(feat: dict, adapters: dict, face_ok=True) -> RT.Router:
    return RT.Router(RULES, THR, TABLE, adapters, features_fn=lambda img: dict(feat),
                     face_check=lambda out: face_ok, timeout_s=5.0)


def test_mild_inputs_are_never_enhanced(pairs):
    for pop, ps in pairs.items():
        mild = [p for p in ps.values() if p.severity == "mild"]
        assert len(mild) == 252
        spies = {c: Spy() for c in RT.CANDIDATES}
        for p in mild:
            out, rec = _router(p.feat, spies).route(IMG, {}, ORIG)
            assert rec.used == RT.ORIGINAL and out is ORIG, (pop, p.pair_id, rec)
            assert rec.decision != GT.ENH, (pop, p.pair_id)
        assert sum(s.calls for s in spies.values()) == 0


def test_a1_a2_a5_are_never_routed_on_any_baseline_input(pairs):
    seen = set()
    for ps in pairs.values():
        r = _router({}, {})
        for p in ps.values():
            rec = r.plan(p.feat)
            seen.add(rec.route)
            assert rec.route not in BF.NOT_ROUTABLE
            if rec.route == "A4":
                assert rec.decision == GT.ENH and rec.bucket in ("NOISE", "BLUR")
    assert seen == {RT.ORIGINAL, "A4"}


def test_a4_routed_input_never_calls_other_candidates(pairs):
    p = next(p for p in pairs["p4dev"].values() if _router({}, {}).plan(p.feat).route == "A4")
    spies = {c: Spy() for c in RT.CANDIDATES}
    out, rec = _router(p.feat, spies).route(IMG, {}, ORIG)
    assert rec.used == "A4" and out is GOOD and spies["A4"].calls == 1
    assert all(spies[c].calls == 0 for c in RT.CANDIDATES if c != "A4")


@pytest.mark.parametrize("upd,why", [
    ({"det_status": "NO_FACE_DETECTED"}, "no_valid_face"),
    ({"det_status": "FACE_TOO_SMALL"}, "no_valid_face"),
    ({"det_score": 0.5}, "low_detection_confidence"),
    ({"det_score": float("nan")}, "low_detection_confidence"),
    ({"yaw_proxy": 0.4}, "alignment_unreliable"),
    ({"yaw_proxy": float("nan")}, "alignment_unreliable")])
def test_unsafe_or_uncertain_returns_original_and_escalates(pairs, upd, why):
    """Take an input the frozen route would send to A4 and make the detection unreliable."""
    p = next(p for p in pairs["main"].values() if _router({}, {}).plan(p.feat).route == "A4")
    spies = {c: Spy() for c in RT.CANDIDATES}
    out, rec = _router({**p.feat, **upd}, spies).route(IMG, {}, ORIG)
    assert rec.decision == GT.UNSAFE and rec.reason.startswith(why) and rec.escalate
    assert rec.used == RT.ORIGINAL and out is ORIG and sum(s.calls for s in spies.values()) == 0


@pytest.mark.parametrize("spy,face_ok,fallback", [
    (Spy(exc=RuntimeError("boom")), True, "MODEL_FAILURE"),
    (Spy(exc=MemoryError()), True, "RESOURCE_FAILURE"),
    (Spy(out=np.zeros((224, 224, 3), np.uint8)), True, "INVALID_OUTPUT"),
    (Spy(out=GOOD[:10]), True, "INVALID_OUTPUT"),
    (Spy(), False, "INVALID_OUTPUT")])
def test_a4_failure_falls_back_to_original(pairs, spy, face_ok, fallback):
    p = next(p for p in pairs["main"].values() if _router({}, {}).plan(p.feat).route == "A4")
    out, rec = _router(p.feat, {"A4": spy}, face_ok=face_ok).route(IMG, {}, ORIG)
    assert rec.fallback == fallback and rec.used == RT.ORIGINAL and out is ORIG
