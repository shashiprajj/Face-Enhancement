"""P4-T1 — freeze a real-world evaluation manifest. Refuses unless the source's legal evidence is recorded.

Input: an acquisition manifest written by scripts/p2_validate_collection.py (DS03 rw-spec-1.0.0, or a vendor
collection built to the same spec) + the collection root + the source register data/registry/p4_realworld_sources.csv.

Order of checks (any failure → exit ≠ 0, nothing frozen):
  1. LEGAL GATE (G-3): register row must be legal_status=APPROVED with approval_ref, reviewer, approval_date and
     'evaluation' in approved_uses. Otherwise exit 2 and write a BLOCKED report (no image is opened).
  2. Integrity: every file re-hashed against the acquisition manifest (SHA-256).
  3. Exact duplicates (SHA-256) and near duplicates (64-bit pHash, Hamming ≤ near_dup_max_hamming).
  4. Leakage: a participant in two splits; an exact/near-duplicate group spanning splits or participants;
     optional identity screen (SFace, HQ references of different participants ≥ identity_screen_threshold,
     i.e. the same person enrolled twice). Withdrawn participants must already be absent.
  5. Pairing: every LQ capture → HQ_REF of the same participant.
  6. Freeze: manifest csv.gz + freeze.json (hashes of manifest, acquisition manifest, register row, spec, code).
     A frozen version is immutable: re-freezing with different content under the same version is refused.

    scripts/dock.sh python scripts/p4_freeze_realworld.py --source-id DS03 --acq <acq.csv.gz> --root <collection> \
        --version rw-v1 [--identity-screen]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from fes_data import common as C

REGISTER = "data/registry/p4_realworld_sources.csv"
OUT_ROOT = "data/manifests/phase4/realworld"
FIELDS = ["capture_id", "participant_id", "session_id", "role", "condition", "level", "device_id", "pose",
          "paired_ref_capture_id", "split", "path", "sha256", "width", "height", "phash64", "dup_group",
          "source_id", "spec_version", "data_kind"]


class Blocked(Exception):
    """Legal evidence insufficient — final evaluation must stop at this boundary (rule 17)."""


def legal_gate(register: list[dict], source_id: str) -> dict:
    rows = [r for r in register if r["source_id"] == source_id]
    if len(rows) != 1:
        raise Blocked(f"{source_id}: not in the source register (exactly one row required)")
    r = rows[0]
    missing = [k for k in ("approval_ref", "reviewer", "approval_date") if not r.get(k, "").strip()]
    uses = {u.strip() for u in r.get("approved_uses", "").split(";") if u.strip()}
    if r.get("legal_status") != "APPROVED":
        raise Blocked(f"{source_id}: legal_status={r.get('legal_status')!r} (APPROVED required); "
                      f"reason: {r.get('blocking_reason', '')}")
    if missing:
        raise Blocked(f"{source_id}: APPROVED but missing {missing}")
    if "evaluation" not in uses:
        raise Blocked(f"{source_id}: 'evaluation' not in approved_uses {sorted(uses)}")
    return r


def check_and_build(acq: list[dict], root: Path, source_id: str, near_max: int,
                    identity_screen: float | None = None) -> tuple[list[dict], list[str], dict]:
    errors: list[str] = []
    rows, hashes, imgs_hq = [], {}, {}
    for a in acq:
        if str(a.get("excluded_withdrawn", "False")).lower() == "true":
            errors.append(f"{a['capture_id']}: withdrawn participant present in acquisition manifest")
            continue
        f = root / a["path"]
        if not f.is_file() or C.sha256_file(f) != a["sha256"]:
            errors.append(f"{a['capture_id']}: file missing or SHA-256 mismatch")
            continue
        img = C.decode_image(f.read_bytes())
        h = C.phash64(img)
        hashes[a["capture_id"]] = h
        if a["role"] == "HQ_REF":
            imgs_hq[a["capture_id"]] = img
        rows.append({k: a.get(k, "") for k in FIELDS} | {"phash64": f"{h:016x}", "source_id": source_id,
                                                         "data_kind": "REAL_WORLD"})
    by_id = {r["capture_id"]: r for r in rows}

    # participants must not span splits
    psplit = defaultdict(set)
    for r in rows:
        psplit[r["participant_id"]].add(r["split"])
    errors += [f"{p}: participant in several splits {sorted(s)}" for p, s in psplit.items() if len(s) > 1]

    # exact + near duplicates → groups; a group may not span participants or splits
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)

    near = defaultdict(list)
    for cid, g in C.near_duplicate_groups(hashes, max_dist=near_max).items():
        near[g].append(cid)
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r["capture_id"])
    for ids in list(near.values()) + list(by_sha.values()):
        for i in ids[1:]:
            union(ids[0], i)
    members = defaultdict(list)
    for cid in by_id:
        members[find(cid)].append(cid)
    leak = {"duplicate_groups": 0, "cross_participant_groups": [], "cross_split_groups": [], "identity_flags": []}
    for g, ids in members.items():
        if len(ids) < 2:
            continue
        leak["duplicate_groups"] += 1
        for cid in ids:
            by_id[cid]["dup_group"] = f"dg-{g}"
        if len({by_id[c]["participant_id"] for c in ids}) > 1:
            leak["cross_participant_groups"].append(sorted(ids))
        if len({by_id[c]["split"] for c in ids}) > 1:
            leak["cross_split_groups"].append(sorted(ids))
    errors += [f"duplicate group spans participants: {g}" for g in leak["cross_participant_groups"]]
    errors += [f"duplicate group spans splits: {g}" for g in leak["cross_split_groups"]]

    # pairing
    for r in rows:
        if r["role"] == "LQ":
            ref = by_id.get(r["paired_ref_capture_id"])
            if not ref or ref["role"] != "HQ_REF" or ref["participant_id"] != r["participant_id"]:
                errors.append(f"{r['capture_id']}: LQ capture without a same-participant HQ_REF")

    # identity screen: the same person enrolled under two participant ids (embeddings in memory only)
    if identity_screen is not None and imgs_hq:
        import numpy as np

        from fes_bench.metrics import SFace
        from fes_data.preprocess import YuNetDetector
        det = YuNetDetector(C.repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), score_threshold=0.7)
        sf, emb = SFace(), {}
        for cid, img in imgs_hq.items():
            faces = det.detect(img)
            if faces:
                emb[cid] = sf.embed(img, np.asarray(faces[0].landmarks))
        ids = sorted(emb)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = by_id[a]["participant_id"], by_id[b]["participant_id"]
                if pa != pb and float(emb[a] @ emb[b]) >= identity_screen:
                    leak["identity_flags"].append([a, b, round(float(emb[a] @ emb[b]), 4)])
        errors += [f"identity screen: {a} ({by_id[a]['participant_id']}) ~ {b} ({by_id[b]['participant_id']}) "
                   f"cos {c}" for a, b, c in leak["identity_flags"]]
    return sorted(rows, key=lambda r: r["capture_id"]), errors, leak


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--acq", required=True, help="acquisition manifest from p2_validate_collection.py")
    ap.add_argument("--root", required=True, help="collection root")
    ap.add_argument("--version", required=True, help="frozen manifest version, e.g. rw-v1")
    ap.add_argument("--register", default=REGISTER)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--near-dup-max-hamming", type=int, default=3)    # P2-T2 near-duplicate rule
    ap.add_argument("--identity-screen", type=float, nargs="?", const=0.59, default=None,
                    help="SFace 1:N flag threshold (default 0.59, data-derived in P2-T7)")
    a = ap.parse_args(argv)
    out = C.repo_path(a.out_root) / a.version
    with open(C.repo_path(a.register) if not Path(a.register).is_absolute() else a.register, encoding="utf-8") as f:
        register = list(csv.DictReader(f))
    try:
        reg_row = legal_gate(register, a.source_id)
    except Blocked as e:
        C.write_json(out / "BLOCKED.json", {"run": C.run_info(), "source_id": a.source_id, "status": "BLOCKED_G3",
                                            "reason": str(e), "register": a.register})
        print("BLOCKED_G3:", e)
        return 2
    acq_path = Path(a.acq)
    acq = C.read_csv_gz(acq_path)
    rows, errors, leak = check_and_build(acq, Path(a.root), a.source_id, a.near_dup_max_hamming, a.identity_screen)
    for e in errors:
        print("ERROR", e)
    if errors:
        return 1
    man = out / "manifest.csv.gz"
    if (out / "freeze.json").exists():
        prev = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
        tmp = out / "manifest.recheck.csv.gz"
        C.write_csv_gz(tmp, rows, FIELDS)
        same = C.sha256_file(tmp) == prev["manifest_sha256"]
        tmp.unlink()
        if not same:
            print(f"REFUSED: {a.version} is frozen with different content; use a new --version")
            return 3
        print(f"{a.version}: already frozen, content identical")
        return 0
    C.write_csv_gz(man, rows, FIELDS)
    freeze = {"run": C.run_info(), "version": a.version, "source_id": a.source_id, "data_kind": "REAL_WORLD",
              "manifest": C.rel_path(man) if str(man).startswith(str(C.REPO_ROOT)) else str(man),
              "manifest_sha256": C.sha256_file(man), "acquisition_manifest_sha256": C.sha256_file(acq_path),
              "register_row": reg_row, "register_row_sha256": C.sha256_bytes(json.dumps(reg_row, sort_keys=True).encode()),
              "n_captures": len(rows), "n_participants": len({r["participant_id"] for r in rows}),
              "splits": {s: sum(r["split"] == s for r in rows) for s in ("dev", "test")},
              "leakage": leak, "near_dup_max_hamming": a.near_dup_max_hamming,
              "identity_screen_threshold": a.identity_screen,
              "rule": "Frozen BEFORE any final evaluation. Thresholds are calibrated on split=dev only."}
    C.write_json(out / "freeze.json", freeze)
    print(json.dumps({k: freeze[k] for k in ("version", "manifest_sha256", "n_captures", "splits")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
