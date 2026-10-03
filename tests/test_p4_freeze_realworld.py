"""P4-T1 freeze tool: legal gate, integrity, duplicates/leakage, immutability. Synthetic fixtures only (not data)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import cv2
import pytest

from fes_data import common as C

from test_collection_validator import SPEC, _make, vc  # noqa: F401  (fixture builder + validator module)

_spec = importlib.util.spec_from_file_location("fz", C.repo_path("scripts/p4_freeze_realworld.py"))
fz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fz)

REG_FIELDS = ["source_id", "legal_status", "approval_ref", "reviewer", "approval_date", "approved_uses",
              "blocking_reason"]


def _register(tmp: Path, **row) -> Path:
    base = {"source_id": "DS03", "legal_status": "APPROVED", "approval_ref": "LEGAL-TEST-1", "reviewer": "test",
            "approval_date": "2026-10-03", "approved_uses": "evaluation", "blocking_reason": ""}
    base.update(row)
    p = tmp / "register.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REG_FIELDS)
        w.writeheader()
        w.writerow(base)
    return p


def _acq(tmp: Path, face, caps_mutate=None) -> tuple[Path, Path]:
    root = _make(tmp / "col", face, caps_mutate=caps_mutate)
    errors, _, rows = vc.validate(root, SPEC)
    assert not errors, errors
    acq = tmp / "acq.csv.gz"
    C.write_csv_gz(acq, rows)
    return root, acq


def _run(tmp, root, acq, reg, version="rw-test", extra=()):
    return fz.main(["--source-id", "DS03", "--acq", str(acq), "--root", str(root), "--version", version,
                    "--register", str(reg), "--out-root", str(tmp / "out"), *extra])


@pytest.mark.parametrize("row", [{"legal_status": "PENDING_LEGAL_REVIEW"}, {"approval_ref": ""}, {"reviewer": ""},
                                 {"approved_uses": "training"}])
def test_legal_gate_blocks_without_evidence(tmp_path, row):
    reg = _register(tmp_path, **row)
    # the gate runs before any image is opened: a non-existent collection is fine here
    rc = _run(tmp_path, tmp_path / "nope", tmp_path / "nope.csv.gz", reg)
    assert rc == 2
    blocked = json.loads((tmp_path / "out" / "rw-test" / "BLOCKED.json").read_text())
    assert blocked["status"] == "BLOCKED_G3"


def test_real_register_blocks_every_source():
    """The committed register has no APPROVED real-world source (G-3 open)."""
    with open(C.repo_path(fz.REGISTER), encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
    for r in reg:
        with pytest.raises(fz.Blocked):
            fz.legal_gate(reg, r["source_id"])


def test_freeze_ok_and_immutable(tmp_path, ref_faces):
    root, acq = _acq(tmp_path, ref_faces[0])
    reg = _register(tmp_path)
    # the fixture reuses one face image for both participants → identical files across participants = leakage
    assert _run(tmp_path, root, acq, reg) == 1


def test_freeze_clean_collection(tmp_path, ref_faces):
    def distinct(caps, tmp):
        # participant B gets a different face so no duplicate spans participants
        for c in caps:
            if c["participant_id"] == "P-00000B":
                f = tmp / "images" / c["participant_id"] / c["file"]
                img = ref_faces[1]
                size = 448 if c["role"] == "HQ_REF" else 56
                cv2.imwrite(str(f), cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    root, acq = _acq(tmp_path, ref_faces[0], caps_mutate=distinct)
    reg = _register(tmp_path)
    # same participant, identical LQ files (5 conditions share one image) → allowed (one participant, one split)
    assert _run(tmp_path, root, acq, reg) == 0
    fr = json.loads((tmp_path / "out" / "rw-test" / "freeze.json").read_text())
    assert fr["n_participants"] == 2 and fr["data_kind"] == "REAL_WORLD"
    assert _run(tmp_path, root, acq, reg) == 0                       # identical re-freeze is a no-op
    # tamper: change one file → integrity failure
    f = next((root / "images" / "P-00000A").glob("C-A*.png"))
    f.write_bytes(f.read_bytes() + b"\0")
    assert _run(tmp_path, root, acq, reg, version="rw-test2") == 1


def test_frozen_version_refuses_different_content(tmp_path, ref_faces):
    def distinct(caps, tmp):
        for c in caps:
            if c["participant_id"] == "P-00000B":
                size = 448 if c["role"] == "HQ_REF" else 56
                cv2.imwrite(str(tmp / "images" / c["participant_id"] / c["file"]),
                            cv2.resize(ref_faces[1], (size, size), interpolation=cv2.INTER_AREA))
    root, acq = _acq(tmp_path, ref_faces[0], caps_mutate=distinct)
    reg = _register(tmp_path)
    assert _run(tmp_path, root, acq, reg) == 0
    rows = C.read_csv_gz(acq)[:-1]                                   # drop one capture → different content
    acq2 = tmp_path / "acq2.csv.gz"
    C.write_csv_gz(acq2, rows)
    assert _run(tmp_path, root, acq2, reg) == 3
