"""Phase 5 Rapid — frozen T6/T7 decisions for the 200 real photos (Dataset-and-code/dataset), full_image mode.

    .venv/Scripts/python.exe scripts/r5_realphotos_gate.py       (numpy/OpenCV only; no model is run)

Uses the R5-1 service pipeline unchanged (detection → integer face chip → p4-quality → frozen gate/router) with a
model stub that is never reached for non-A4 routes. Writes data/manifests/phase5_rapid/realphotos/gate.csv.gz + summary.
No identity labels exist for these photos: this measures routing behaviour only.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

from fes_data.common import repo_path, run_info, sha256_file, write_csv_gz, write_json
from fes_service.pipeline import EnhancementPipeline


class Stub:
    calls = 0

    def __call__(self, img, ctx):
        Stub.calls += 1
        import cv2
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), {}


def main() -> None:
    files = sorted(glob.glob(str(repo_path("Dataset-and-code/dataset/*.jpg"))),
                   key=lambda f: int(os.path.basename(f)[3:-4]))
    pipe = EnhancementPipeline(Stub(), timeout_s=30)
    rows = []
    for f in files:
        r = pipe.process(open(f, "rb").read(), "full_image")
        rows.append({"image": os.path.basename(f), "sha256": sha256_file(f), "status": r.status,
                     "decision": r.decision, "reason": r.reason, "bucket": r.bucket, "route": r.route,
                     "n_faces": r.n_faces, "chip_px": r.chip_px, "warnings": "|".join(r.warnings),
                     **{f"q_{k}": v for k, v in r.quality.items()}})
    out = repo_path("data/manifests/phase5_rapid/realphotos")
    write_csv_gz(out / "gate.csv.gz", rows, list(rows[0]))
    summ = {"run": run_info(), "n": len(rows), "data_kind": "REAL (Kaggle low-resolution-photographs, CC0 as uploaded)",
            "decisions": dict(Counter(r["decision"] for r in rows)), "routes": dict(Counter(r["route"] for r in rows)),
            "buckets": dict(Counter(r["bucket"] for r in rows if r["bucket"])),
            "unsafe_reasons": dict(Counter(r["reason"].split(":")[0] for r in rows if r["decision"] == "UNSAFE_OR_UNCERTAIN")),
            "enhance_rules": dict(Counter(r["reason"] for r in rows if r["decision"] == "ENHANCE_CANDIDATE")),
            "multi_face_images": sum(r["n_faces"] > 1 for r in rows)}
    write_json(out / "gate_summary.json", summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "run"}, indent=1))


if __name__ == "__main__":
    main()
