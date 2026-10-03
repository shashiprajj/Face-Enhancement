"""P4-T2 — compute versioned no-reference quality metadata for every degraded input (and clean reference) of a
frozen population. Labels carry their source (SYNTHETIC_GENERATOR here); features never read the reference.

    scripts/dock.sh python scripts/p4_quality_features.py --population p4dev --root data/manifests/phase4
    scripts/dock.sh python scripts/p4_quality_features.py --population main  --root data/manifests/phase3

Writes data/manifests/phase4/quality/<population>.csv.gz (+ _report.json).
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

from fes_bench import quality as Q
from fes_bench import read_bytes_retry
from fes_data.common import load_config, read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_csv_gz, write_json
from fes_data.preprocess import YuNetDetector

TAX = "configs/phase4/taxonomy_v1.yaml"
YUNET = "models/D01_yunet/face_detection_yunet_2026may.onnx"
ID_FIELDS = ["pair_id", "ref_id", "cell", "profile", "severity", "item", "data_kind", "label_source", "families",
             "mixed_flag", "split"]


def label(tax: dict, profile: str) -> tuple[str, bool]:
    fam = tax["synthetic_cell_map"][profile]
    mixed = "MIXED" in fam or len([f for f in fam if f != "MIXED"]) >= 2
    return ";".join(fam), mixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    cv2.setNumThreads(2)
    tax = load_config(TAX)
    det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)
    pop_path = repo_path(f"{a.root}/population_{a.population}.csv.gz")
    pop = read_csv_gz(pop_path)
    split = "dev" if a.population == "p4dev" else "test"
    rows, seen = [], set()

    def one(path: str, sha: str, base: dict) -> None:
        data = read_bytes_retry(repo_path(path))
        assert sha256_bytes(data) == sha, path
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        rows.append(base | Q.features(img, det))

    for p in pop:
        fam, mixed = label(tax, p["profile"])
        one(p["degraded_path"], p["degraded_sha256"],
            {"pair_id": p["pair_id"], "ref_id": p["ref_id"], "cell": p["cell"], "profile": p["profile"],
             "severity": p["severity"], "item": "degraded", "data_kind": "SYNTHETIC",
             "label_source": "SYNTHETIC_GENERATOR", "families": fam, "mixed_flag": mixed, "split": split})
        if p["ref_id"] not in seen:
            seen.add(p["ref_id"])
            one(p["ref_path"], p["ref_sha256"],
                {"pair_id": "", "ref_id": p["ref_id"], "cell": "clean", "profile": "clean", "severity": "none",
                 "item": "reference", "data_kind": "SYNTHETIC_SOURCE_CLEAN", "label_source": "SYNTHETIC_GENERATOR",
                 "families": "CLEAN", "mixed_flag": False, "split": split})
    out = repo_path(f"data/manifests/phase4/quality/{a.population}.csv.gz")
    fields = ID_FIELDS + Q.FIELDS
    write_csv_gz(out, [{k: ("" if isinstance(r.get(k), float) and math.isnan(r[k]) else r.get(k, "")) for k in fields}
                       for r in rows], fields)
    det_rate = {}
    for r in rows:
        d = det_rate.setdefault(r["cell"], [0, 0])
        d[0] += r["det_status"] == "OK"
        d[1] += 1
    write_json(out.with_name(f"{a.population}_report.json"), {
        "run": run_info({"quality_version": Q.QUALITY_VERSION, "taxonomy": TAX,
                         "taxonomy_version": tax["taxonomy_version"], "taxonomy_sha256": tax["_config_sha256"],
                         "detector": det.id, "detector_sha256": sha256_file(repo_path(YUNET))}),
        "population": a.population, "population_sha256": sha256_file(pop_path), "split": split,
        "n_rows": len(rows), "manifest_sha256": sha256_file(out),
        "detection_ok_rate_by_cell": {k: round(v[0] / v[1], 4) for k, v in sorted(det_rate.items())}})
    print(a.population, len(rows), sha256_file(out)[:12])


if __name__ == "__main__":
    main()
