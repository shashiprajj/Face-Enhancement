"""P2-T5 — Validate an internal real-world collection (DS03) against the executable spec and emit its
acquisition manifest. Refuses (exit 1) on any hard violation. Never modifies the collection.

Expected layout:
    <root>/participants.csv   participant_id, consent_ref_id, consent_scopes (';'-sep), consent_date (YYYY-MM-DD),
                              withdrawn (true/false) [, age_band, gender_self, skin_tone_mst]
    <root>/captures.csv       capture_id, participant_id, session_id, role (HQ_REF|LQ), condition, level, device_id,
                              distance_m, lux_measured, pose, file, paired_ref_capture_id (LQ only)
    <root>/images/<participant_id>/<capture_id>.<ext>

    scripts/dock.sh python scripts/p2_validate_collection.py --root <collection_root> \
        [--spec configs/phase2/realworld_collection_v1.yaml] [--manifest out.csv.gz]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from fes_data import common as C
from fes_data.preprocess import YuNetDetector

DETECTOR_MODEL = "models/D01_yunet/face_detection_yunet_2026may.onnx"


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _has_gps(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            return bool(exif and exif.get_ifd(0x8825))
    except Exception:
        return False


def split_for(participant_id: str, spec: dict) -> str:
    h = int(hashlib.sha256((spec["splits"]["salt"] + participant_id).encode()).hexdigest(), 16) % 100
    return "dev" if h < spec["splits"]["dev_below"] else "test"


def validate(root: Path, spec: dict) -> tuple[list[str], list[str], list[dict]]:
    errors, warnings, rows = [], [], []
    detector = YuNetDetector(C.repo_path(DETECTOR_MODEL), score_threshold=0.7, max_side=1280)
    min_eye = spec["capture"]["hq_reference"]["min_eye_distance_px"]
    ids, mini = spec["identifiers"], spec["data_minimization"]
    part_path, cap_path = root / "participants.csv", root / "captures.csv"
    for p in (part_path, cap_path):
        if not p.exists():
            return [f"missing {p.name}"], [], []
    parts, caps = _read(part_path), _read(cap_path)

    # data minimization: forbidden columns and PII-looking values anywhere
    for name, table in (("participants.csv", parts), ("captures.csv", caps)):
        cols = {c.lower() for c in (table[0].keys() if table else [])}
        bad = cols & set(mini["forbidden_columns"])
        if bad:
            errors.append(f"{name}: forbidden PII columns {sorted(bad)}")
        for i, r in enumerate(table):
            for k, v in r.items():
                for label, pat in mini["pii_value_patterns"].items():
                    if v and re.search(pat, v):
                        errors.append(f"{name} row {i + 2} column {k}: value looks like {label}")

    # participants & consent
    pmap = {}
    for r in parts:
        pid = r.get("participant_id", "")
        if not re.fullmatch(ids["participant_id_pattern"], pid):
            errors.append(f"bad participant_id {pid!r}")
            continue
        if pid in pmap:
            errors.append(f"duplicate participant_id {pid}")
        scopes = {s.strip() for s in r.get("consent_scopes", "").split(";") if s.strip()}
        missing = set(spec["consent"]["required_scope"]) - scopes
        if not r.get("consent_ref_id"):
            errors.append(f"{pid}: no consent_ref_id")
        if missing:
            errors.append(f"{pid}: consent scope missing {sorted(missing)}")
        if r.get("age_band") and r["age_band"] not in mini["allowed_age_bands"]:
            errors.append(f"{pid}: age_band {r['age_band']!r} not allowed (bands only, no exact age)")
        pmap[pid] = r

    # captures
    cmap = {}
    for r in caps:
        cid, pid = r.get("capture_id", ""), r.get("participant_id", "")
        if not re.fullmatch(ids["capture_id_pattern"], cid):
            errors.append(f"bad capture_id {cid!r}")
            continue
        if cid in cmap:
            errors.append(f"duplicate capture_id {cid}")
        cmap[cid] = r
        if pid not in pmap:
            errors.append(f"{cid}: unknown participant {pid}")
        if not re.fullmatch(ids["session_id_pattern"], r.get("session_id", "")):
            errors.append(f"{cid}: bad session_id")
        if not re.fullmatch(ids["device_id_pattern"], r.get("device_id", "")):
            errors.append(f"{cid}: bad device_id")
        for k in spec["capture"]["record_per_capture"]:
            if k in ("condition", "level") and r.get("role") == "HQ_REF":
                continue  # references are captured under the fixed HQ protocol, not an LQ condition
            if not r.get(k):
                errors.append(f"{cid}: missing {k}")
        if r.get("role") not in ("HQ_REF", "LQ"):
            errors.append(f"{cid}: role must be HQ_REF or LQ")

    # pairing, files, coverage
    coverage = defaultdict(set)
    for cid, r in cmap.items():
        pid = r.get("participant_id")
        f = root / "images" / pid / r.get("file", "")
        if not f.is_file():
            errors.append(f"{cid}: file not found {f.relative_to(root)}")
            continue
        img = C.decode_image(f.read_bytes())
        if img is None:
            errors.append(f"{cid}: unreadable image")
            continue
        if spec["data_minimization"]["strip_exif_gps"] and _has_gps(f):
            errors.append(f"{cid}: image contains GPS EXIF")
        if r["role"] == "HQ_REF":
            dets = detector.detect(img)
            if len(dets) != 1:
                errors.append(f"{cid}: HQ reference must contain exactly one detectable face (found {len(dets)})")
            elif dets[0].eye_dist < min_eye:
                errors.append(f"{cid}: HQ reference inter-ocular distance {dets[0].eye_dist:.0f}px < {min_eye}px")
        if r["role"] == "LQ":
            ref = cmap.get(r.get("paired_ref_capture_id", ""))
            if not ref or ref.get("role") != "HQ_REF":
                errors.append(f"{cid}: paired_ref_capture_id does not point to an HQ_REF capture")
            elif ref.get("participant_id") != pid:
                errors.append(f"{cid}: paired reference belongs to a different participant")
            coverage[pid].add(r.get("condition"))
        withdrawn = pmap.get(pid, {}).get("withdrawn", "false").lower() == "true"
        rows.append({
            "capture_id": cid, "participant_id": pid, "session_id": r["session_id"], "role": r["role"],
            "condition": r.get("condition", ""), "level": r.get("level", ""), "device_id": r["device_id"],
            "distance_m": r.get("distance_m", ""), "lux_measured": r.get("lux_measured", ""), "pose": r.get("pose", ""),
            "paired_ref_capture_id": r.get("paired_ref_capture_id", ""),
            "path": str(f.relative_to(root).as_posix()), "sha256": C.sha256_file(f),
            "width": img.shape[1], "height": img.shape[0],
            "split": split_for(pid, spec) if pid in pmap else "",
            "excluded_withdrawn": withdrawn, "spec_version": spec["spec_version"],
        })

    required = set(spec["capture"]["required_condition_codes_per_participant"])
    for pid in pmap:
        miss = required - coverage[pid]
        if miss:
            warnings.append(f"{pid}: missing required conditions {sorted(miss)}")
    return errors, warnings, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--spec", default="configs/phase2/realworld_collection_v1.yaml")
    ap.add_argument("--manifest", default=None, help="write acquisition manifest (csv.gz) if validation passes")
    a = ap.parse_args()
    spec = C.load_config(a.spec)
    errors, warnings, rows = validate(Path(a.root), spec)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print({"captures": len(rows), "errors": len(errors), "warnings": len(warnings),
           "splits": {s: sum(r["split"] == s for r in rows) for s in ("dev", "test")}})
    if errors:
        raise SystemExit(1)
    if a.manifest:
        C.write_csv_gz(Path(a.manifest), [r for r in rows if not r["excluded_withdrawn"]])


if __name__ == "__main__":
    main()
