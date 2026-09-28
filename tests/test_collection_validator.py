"""P2-T5 validator: synthetic *test fixtures only* (built in a temp dir from a FairFace reference; not data)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import cv2
import pytest
from PIL import Image

from fes_data import common as C

_spec = importlib.util.spec_from_file_location("vc", C.repo_path("scripts/p2_validate_collection.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

SPEC = C.load_config("configs/phase2/realworld_collection_v1.yaml")
PARTS = ["participant_id", "consent_ref_id", "consent_scopes", "consent_date", "withdrawn", "age_band"]
CAPS = ["capture_id", "participant_id", "session_id", "role", "condition", "level", "device_id", "distance_m",
        "lux_measured", "pose", "file", "paired_ref_capture_id"]


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _make(tmp: Path, face, parts_extra=None, caps_mutate=None) -> Path:
    pid, pid2 = "P-00000A", "P-00000B"
    parts = [{"participant_id": pid, "consent_ref_id": "CR-1", "consent_scopes": "evaluation", "consent_date": "2026-10-01",
              "withdrawn": "false", "age_band": "30-39"},
             {"participant_id": pid2, "consent_ref_id": "CR-2", "consent_scopes": "evaluation", "consent_date": "2026-10-01",
              "withdrawn": "false", "age_band": "18-29"}]
    if parts_extra:
        parts_extra(parts)
    hq = cv2.resize(face, (448, 448), interpolation=cv2.INTER_LINEAR)   # eye distance ≈ 158 px ≥ 120
    lq = cv2.resize(face, (56, 56), interpolation=cv2.INTER_AREA)
    caps = []
    for p in (pid, pid2):
        (tmp / "images" / p).mkdir(parents=True, exist_ok=True)
        ref_id = "C-0000000" + ("1" if p == pid else "2")
        cv2.imwrite(str(tmp / "images" / p / f"{ref_id}.png"), hq)
        caps.append({"capture_id": ref_id, "participant_id": p, "session_id": "S-00000A", "role": "HQ_REF", "condition": "",
                     "level": "", "device_id": "D-CAM_01", "distance_m": "1.0", "lux_measured": "500", "pose": "frontal",
                     "file": f"{ref_id}.png", "paired_ref_capture_id": ""})
        for i, cond in enumerate(SPEC["capture"]["required_condition_codes_per_participant"]):
            cid = f"C-{'A' if p == pid else 'B'}000000{i}"
            cv2.imwrite(str(tmp / "images" / p / f"{cid}.png"), lq)
            caps.append({"capture_id": cid, "participant_id": p, "session_id": "S-00000A", "role": "LQ", "condition": cond,
                         "level": "x", "device_id": "D-PHONE_01", "distance_m": "6.0", "lux_measured": "40", "pose": "frontal",
                         "file": f"{cid}.png", "paired_ref_capture_id": ref_id})
    if caps_mutate:
        caps_mutate(caps, tmp)
    _write(tmp / "participants.csv", list(parts[0].keys()), parts)
    _write(tmp / "captures.csv", CAPS, caps)
    return tmp


@pytest.fixture
def face(ref_faces):
    return ref_faces[0]


def test_valid_collection_passes_and_splits_by_participant(tmp_path, face):
    errors, warnings, rows = vc.validate(_make(tmp_path, face), SPEC)
    assert errors == [] and warnings == []
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["participant_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_pid.values())


def test_pii_column_rejected(tmp_path, face):
    def add_name(parts):
        for p in parts:
            p["name"] = "Someone"
    errors, _, _ = vc.validate(_make(tmp_path, face, parts_extra=add_name), SPEC)
    assert any("forbidden PII columns" in e for e in errors)


def test_missing_consent_scope_rejected(tmp_path, face):
    def no_scope(parts):
        parts[0]["consent_scopes"] = "training"
    errors, _, _ = vc.validate(_make(tmp_path, face, parts_extra=no_scope), SPEC)
    assert any("consent scope missing" in e for e in errors)


def test_cross_participant_pairing_rejected(tmp_path, face):
    def swap(caps, tmp):
        lq = next(c for c in caps if c["role"] == "LQ" and c["participant_id"] == "P-00000A")
        lq["paired_ref_capture_id"] = "C-00000002"   # belongs to P-00000B
    errors, _, _ = vc.validate(_make(tmp_path, face, caps_mutate=swap), SPEC)
    assert any("different participant" in e for e in errors)


def test_gps_exif_rejected(tmp_path, face):
    def gps(caps, tmp):
        c = next(c for c in caps if c["role"] == "LQ")
        path = tmp / "images" / c["participant_id"] / c["file"]
        im = Image.open(path).convert("RGB")
        exif = Image.Exif()
        exif.get_ifd(0x8825)[1] = "N"     # GPSLatitudeRef
        jpg = path.with_suffix(".jpg")
        im.save(jpg, exif=exif)
        c["file"] = jpg.name
    errors, _, _ = vc.validate(_make(tmp_path, face, caps_mutate=gps), SPEC)
    assert any("GPS EXIF" in e for e in errors)


def test_low_resolution_hq_reference_rejected(tmp_path, face):
    def shrink(caps, tmp):
        c = next(c for c in caps if c["role"] == "HQ_REF")
        p = tmp / "images" / c["participant_id"] / c["file"]
        cv2.imwrite(str(p), cv2.resize(face, (224, 224)))      # eye distance ≈ 79 px < 120
    errors, _, _ = vc.validate(_make(tmp_path, face, caps_mutate=shrink), SPEC)
    assert any("inter-ocular distance" in e for e in errors)
