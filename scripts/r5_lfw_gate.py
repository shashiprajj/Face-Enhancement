"""R5-2 — classify every LFW research query with the FROZEN gate p5-rwgate-1.0.0 (seal 78837a8d) through the R5-1
service path (full_image: detection → integer face chip → rw gate) and prepare the A4 run.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_lfw_gate.py         (no model runs here)

Groups (task R5-2):
  G1 gate → A4                 route A4
  G2 degraded, left original   DO_NOT_ENHANCE with category degraded_no_a4_evidence / degraded_small / photometric /
                               compression
  G0 acceptable (control)      DO_NOT_ENHANCE, category acceptable
  G3 unsafe / unrecognizable   UNSAFE_OR_UNCERTAIN — A4 is NOT run on these (not even forced)
The chip saved for each G0–G2 query is byte-for-byte what the service would hand to A4 (same detection + face_chip).
Writes data/manifests/phase5_rapid/lfw/{gate.csv.gz, population_lfwq.csv.gz} (+ chips in datasets/work/r5lfw/chips).
`population_lfwq` uses the Phase 3 harness columns so scripts/p3_run_inference.py can run B0/A4 on the chips.
"""
from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np

from fes_data import preprocess as P
from fes_data.common import read_csv_gz, repo_path, sha256_bytes, write_csv_gz
from fes_service.pipeline import EnhancementPipeline, face_chip

G2_CATS = {"degraded_no_a4_evidence", "degraded_small", "photometric", "compression"}


def group(decision: str, category: str, route: str) -> str:
    if route == "A4":
        return "G1"
    if decision == "UNSAFE_OR_UNCERTAIN":
        return "G3"
    return "G2" if category in G2_CATS else "G0"


class Stub:
    def __call__(self, img, ctx):
        raise RuntimeError("stub: gate classification only")


def main() -> None:
    q = read_csv_gz(repo_path("data/manifests/phase5_rapid/lfw/queries.csv.gz"))
    pipe = EnhancementPipeline(Stub(), timeout_s=30, gate="rw")
    assert pipe.seals["rw_gate"].startswith("78837a8d"), "gate is not the frozen p5-rwgate-1.0.0"
    gate_rows, pop = [], []
    for r in q:
        data = repo_path(r["query_path"]).read_bytes()
        res = pipe.process(data, "full_image")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        fr = P.preprocess(img, pipe.det, pipe.det_cfg, input_mode="face_crop")[0]
        g = group(res.decision, res.bucket, res.route)
        row = {"query_id": r["query_id"], "identity": r["identity"], "cell": r["cell"], "group": g,
               "decision": res.decision, "reason": res.reason, "category": res.bucket or "no_face_or_detection",
               "route": res.route, "n_faces": res.n_faces, "chip_px": res.chip_px, "warnings": "|".join(res.warnings),
               **{f"q_{k}": v for k, v in (res.quality or {}).items()}}
        gate_rows.append(row)
        if g != "G3" and fr.status == P.OK and fr.aligned:
            chip, side = face_chip(img, np.asarray(fr.detection.landmarks, np.float64))
            assert chip.shape[0] == res.chip_px, (r["query_id"], chip.shape, res.chip_px)
            ok, buf = cv2.imencode(".png", chip, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            path = f"datasets/work/r5lfw/chips/{r['cell']}/{r['identity']}.png"
            repo_path(path).parent.mkdir(parents=True, exist_ok=True)
            repo_path(path).write_bytes(buf.tobytes())
            pop.append({"population": "lfwq", "pair_id": r["query_id"], "ref_id": r["identity"], "profile": r["cell"],
                        "severity": g, "cell": r["cell"], "race": "", "gender": "", "age": "",
                        "ref_path": r["ref_path"], "ref_sha256": "", "degraded_path": path,
                        "degraded_sha256": sha256_bytes(buf.tobytes()), "out_w": str(chip.shape[0]), "seed": "",
                        "degrade_version": "r5-lfw", "det_landmarks": "", "e_coverage": "1.0", "eval_mask_frac": "0",
                        "ffhq512_transform_2x3": json.dumps([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "ffhq512_scale": "1"})
    out = repo_path("data/manifests/phase5_rapid/lfw")
    keys = sorted({k for r in gate_rows for k in r}, key=lambda k: list(gate_rows[0]).index(k) if k in gate_rows[0] else 99)
    write_csv_gz(out / "gate.csv.gz", gate_rows, keys)
    write_csv_gz(out / "population_lfwq.csv.gz", pop, list(pop[0]))
    print("groups", dict(Counter(r["group"] for r in gate_rows)))
    print("by cell:")
    for c in sorted({r["cell"] for r in gate_rows}):
        print(f"  {c:18s}", dict(Counter(r["group"] for r in gate_rows if r["cell"] == c)))
    print("G2 categories", dict(Counter(r["reason"] for r in gate_rows if r["group"] == "G2")))
    print("G3 reasons", dict(Counter(r["reason"].split(":")[0] for r in gate_rows if r["group"] == "G3")))


if __name__ == "__main__":
    main()
