"""R5-4 harness logic with a FAKE matcher (no Feature-Match, no faces, no model): dataset loading/anonymisation,
rescue / harm / unchanged / wrong-person accounting, UNSAFE inputs never forced, and the blocked exit without a client."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from fes_data.common import repo_path

_spec = importlib.util.spec_from_file_location("r5_fm_validate", repo_path("scripts/r5_fm_validate.py"))
FM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FM)


class FakeFM:
    """'Matches' when the query bytes carry the reference's tag; ENH bytes always match their own person."""

    def verify(self, reference: bytes, query: bytes) -> dict:
        tag = reference[:3]
        ok = query.startswith(tag) or query == b"ENH:" + tag
        wrong = query == b"BAD" and tag == b"bob"      # an altered face that now looks like Bob
        return {"score": 1.0 if ok else 0.0, "match": ok or wrong, "threshold": 0.5, "face_found": True}


def _people(tmp_path):
    spec = {"alice": [b"ali-ok", b"xxx-lost"], "bob": [b"bob-ok", b"BAD"]}
    for name, qs in spec.items():
        (tmp_path / name / "reference").mkdir(parents=True)
        (tmp_path / name / "degraded").mkdir(parents=True)
        (tmp_path / name / "reference" / "r.jpg").write_bytes(name[:3].encode() + b"-ref")
        for i, q in enumerate(qs):
            (tmp_path / name / "degraded" / f"q{i}.jpg").write_bytes(q)
    (tmp_path / "labels.csv").write_text("person_id,file,degradation\nalice,q1.jpg,blur\n", encoding="utf-8")
    return FM.load_dataset(tmp_path)


def test_dataset_is_anonymised_and_labelled(tmp_path):
    people = _people(tmp_path)
    assert [p["person"] for p in people] == ["P01", "P02"]
    assert [q["degradation"] for q in people[0]["queries"]] == ["unlabelled", "blur"]


def test_rescue_harm_unchanged_and_wrong_person(tmp_path):
    people = _people(tmp_path)

    def process(data):
        if data == b"xxx-lost":     # gate routes to A4 and A4 rescues it
            return SimpleNamespace(status="ENHANCED", image=b"ENH:ali", decision="ENHANCE_CANDIDATE", reason="strong_blur",
                                   fallback="", n_faces=1)
        if data == b"bob-ok":       # unsafe: original returned, never forced
            return SimpleNamespace(status="ORIGINAL", image=data, decision="UNSAFE_OR_UNCERTAIN", reason="pose",
                                   fallback="", n_faces=1)
        return SimpleNamespace(status="ORIGINAL", image=data, decision="DO_NOT_ENHANCE", reason="acceptable",
                               fallback="", n_faces=1)
    forced = []

    def force(data):
        forced.append(data)
        return b"BAD" if data == b"ali-ok" else data        # forcing harms a correct match into a wrong person
    res = FM.evaluate(people, FakeFM(), process, force)
    m = res["metrics"]
    assert b"bob-ok" not in forced                           # UNSAFE never forced
    assert m["pipeline_B"]["rescued"] == 1 and m["pipeline_B"]["harmed"] == 0
    assert m["gate_admitted"]["n"] == 1 and m["gate_admitted"]["rescued"] == 1
    assert m["forced_C"]["harmed"] == 1 and m["forced_C"]["new_wrong_person"] == 1
    assert all(r["query_id"].startswith(("P01-", "P02-")) for r in res["rows"])


def test_blocked_without_client(monkeypatch):
    monkeypatch.delenv("FES_FM_CLIENT", raising=False)
    with pytest.raises(SystemExit, match="BLOCKED"):
        FM.load_client()
