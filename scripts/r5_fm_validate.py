"""R5-4 — Feature-Match validation harness (evaluation-only; READY, NOT YET RUN: Feature-Match is not accessible).

    FES_FM_CLIENT=<module:factory> FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_fm_validate.py \
        --data <dir> [--no-a4]

Feature-Match is EXTERNAL (Phase 0 X6). This script never imports, edits or configures it: it calls a client object
supplied by the user through FES_FM_CLIENT="package.module:factory". factory() must return an object with
    verify(reference: bytes, query: bytes) -> {"score": float, "match": bool, "threshold": float, "face_found": bool}
using Feature-Match's OWN preprocessing, model and configured threshold (nothing here changes them).
Dataset layout (personal photos stay outside git):
    <data>/<person_id>/reference/<one image>          <data>/<person_id>/degraded/<one or more images>
    optional <data>/labels.csv: person_id,file,degradation   (blur | low_light | noise | compression | small_face |
                                                              pose | other)
Conditions per degraded query (gate p5-rwgate-1.0.0, frozen; R5-1 service, full_image mode):
    A ORIGINAL      original bytes → Feature-Match
    B PIPELINE      service output → Feature-Match (A4 face chip if the gate routes to A4, else original bytes)
    C FORCED A4     A4 on the service's face chip regardless of the gate (diagnostic); never for UNSAFE inputs
Wrong-person check: every condition of every query is also verified against every OTHER person's reference.
Writes data/manifests/phase5_rapid/feature_match_validation/results.json (anonymous ids, hashes, scores only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_client():
    spec = os.environ.get("FES_FM_CLIENT")
    if not spec:
        raise SystemExit("BLOCKED: FES_FM_CLIENT is not set — no Feature-Match access (see R5-4 blocker report)")
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)()


def load_dataset(root: Path) -> list[dict]:
    """→ [{person, ref_path, queries: [{path, degradation}]}]; person ids anonymised as P01, P02 … (sorted order)."""
    labels = {}
    lab = root / "labels.csv"
    if lab.exists():
        with open(lab, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels[(r["person_id"], r["file"])] = r["degradation"]
    people = []
    for i, d in enumerate(sorted(p for p in root.iterdir() if p.is_dir()), 1):
        refs = sorted(p for p in (d / "reference").glob("*") if p.suffix.lower() in IMG_EXT)
        degs = sorted(p for p in (d / "degraded").glob("*") if p.suffix.lower() in IMG_EXT)
        if not refs or not degs:
            continue
        people.append({"person": f"P{i:02d}", "ref_path": refs[0],
                       "queries": [{"path": q, "degradation": labels.get((d.name, q.name), "unlabelled")} for q in degs]})
    return people


def evaluate(people: list[dict], client, process, force_a4) -> dict:
    """process(bytes) → service Result; force_a4(bytes) → PNG bytes or None (UNSAFE / no chip)."""
    refs = {p["person"]: p["ref_path"].read_bytes() for p in people}
    rows = []
    for p in people:
        for j, q in enumerate(p["queries"], 1):
            data = q["path"].read_bytes()
            res = process(data)
            outs = {"A": data, "B": res.image if res.status == "ENHANCED" else data}
            if res.decision != "UNSAFE_OR_UNCERTAIN":
                outs["C"] = force_a4(data)
            row = {"query_id": f"{p['person']}-Q{j}", "person": p["person"], "degradation": q["degradation"],
                   "sha256": hashlib.sha256(data).hexdigest(), "gate_decision": res.decision, "gate_reason": res.reason,
                   "a4_ran": res.status == "ENHANCED", "fallback": res.fallback, "n_faces": res.n_faces}
            for c, b in outs.items():
                if b is None:
                    row[c] = None
                    continue
                g = client.verify(refs[p["person"]], b)
                wrong = [o for o in refs if o != p["person"] and client.verify(refs[o], b)["match"]]
                row[c] = {"score": g["score"], "threshold": g["threshold"], "face_found": g.get("face_found", True),
                          "correct": bool(g["match"]), "wrong_person_matches": len(wrong)}
            rows.append(row)
    return {"rows": rows, "metrics": metrics(rows)}


def metrics(rows: list[dict]) -> dict:
    def block(sel, c):
        pair = [(r["A"], r[c]) for r in sel if r.get(c) is not None and r.get("A") is not None]
        a = sum(x["correct"] for x, _ in pair)
        b = sum(y["correct"] for _, y in pair)
        return {"n": len(pair), "correct_A": a, "correct_X": b,
                "rate_A": round(a / len(pair), 4) if pair else None, "rate_X": round(b / len(pair), 4) if pair else None,
                "rescued": sum(not x["correct"] and y["correct"] for x, y in pair),
                "harmed": sum(x["correct"] and not y["correct"] for x, y in pair),
                "unchanged": sum(x["correct"] == y["correct"] for x, y in pair),
                "new_wrong_person": sum(y["wrong_person_matches"] > 0 and x["wrong_person_matches"] == 0 for x, y in pair)}
    out = {"pipeline_B": block(rows, "B"), "forced_C": block(rows, "C"),
           "gate_admitted": block([r for r in rows if r["a4_ran"]], "B"),
           "gate_rejected_forced": block([r for r in rows if not r["a4_ran"]], "C"),
           "by_degradation": {d: {"B": block([r for r in rows if r["degradation"] == d], "B"),
                                  "C": block([r for r in rows if r["degradation"] == d], "C")}
                              for d in sorted({r["degradation"] for r in rows})},
           "gate_decisions": dict(Counter(r["gate_decision"] for r in rows))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-a4", action="store_true", help="dry run of the harness without A4 (stub)")
    a = ap.parse_args()
    from fes_data import preprocess as P
    from fes_data.common import repo_path, run_info, write_json
    from fes_service import rwgate as RW
    from fes_service.pipeline import EnhancementPipeline, face_chip
    client = load_client()
    people = load_dataset(Path(a.data))
    if a.no_a4:
        enhance = lambda chip, ctx: (cv2.resize(chip, (224, 224), interpolation=cv2.INTER_CUBIC), {})  # noqa: E731
        model = enhance
    else:
        from fes_bench.adapters import build
        from fes_service.worker import IsolatedModel
        ad = build("A4")
        ad.load()
        enhance = ad.enhance
        model = IsolatedModel("fes_service.models:a4_factory", timeout_s=600, threads=4)
    pipe = EnhancementPipeline(model, timeout_s=600, gate="rw")
    assert pipe.seals["rw_gate"].startswith("78837a8d"), "gate is not the frozen p5-rwgate-1.0.0"

    def force_a4(data: bytes):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        fr = P.preprocess(img, pipe.det, pipe.det_cfg, input_mode="face_crop")[0]
        if fr.status != P.OK or not fr.aligned:
            return None
        chip, _ = face_chip(img, np.asarray(fr.detection.landmarks, np.float64))
        out, _ = enhance(chip, {"s": chip.shape[0]})
        if RW.is_gray(chip):
            out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return cv2.imencode(".png", out)[1].tobytes()

    res = evaluate(people, client, lambda b: pipe.process(b, "full_image"), force_a4)
    if hasattr(model, "close"):
        model.close()
    res.update(run=run_info(), status="RUN", people=len(people), queries=len(res["rows"]),
               gate_seal=pipe.seals["rw_gate"], client=os.environ.get("FES_FM_CLIENT"))
    write_json(repo_path("data/manifests/phase5_rapid/feature_match_validation/results.json"), res)
    print(res["metrics"])


if __name__ == "__main__":
    main()
