"""P4-T7 — calibrate and FREEZE selective routing on DEV only, on top of the frozen T6 gate.

    scripts/dock.sh python scripts/p4_route_calibrate.py

1. Bucket thresholds = quantiles (2 % / 98 %) of p4-quality features of the 112 CLEAN dev_pilot references
   (population_noise_floor) — what "not degraded" looks like; no outcome data.
2. Route table: on p4dev pairs the frozen T6 gate sends to ENHANCE_CANDIDATE, per bucket pick the candidate with the
   largest U = #CONFIRMED_BENEFIT − 3·#HARM (E01+E07 verdicts), only if U > 0 and n ≥ 10; else ORIGINAL.
Writes configs/phase4/t7_routing.json (sealed; references the T6 seal) + data/manifests/phase4/t7/calibration_report.json.
"""

from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import quality as Q
from fes_bench import routing as RT
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_file, write_json
from fes_data.preprocess import YuNetDetector


def clean_dev_features() -> list[dict]:
    det = YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    seen, out = set(), []
    for p in read_csv_gz(repo_path("data/manifests/phase3/population_noise_floor.csv.gz")):
        if p["ref_id"] in seen:
            continue
        seen.add(p["ref_id"])
        assert sha256_file(repo_path(p["ref_path"])) == p["ref_sha256"]
        out.append(Q.features(cv2.imread(str(repo_path(p["ref_path"])), cv2.IMREAD_COLOR), det))
    return out


def main() -> None:
    gate = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    rules = GT.load_rules(gate)
    t = RT.bucket_thresholds(clean_dev_features())
    pairs = E.load("p4dev")
    delta = E.deltas_from_validation()
    P = sorted(pairs.values(), key=lambda p: p.pair_id)
    G = [p for p in P if GT.decide(p.feat, rules)[0] == GT.ENH]
    buckets = [RT.bucket_of(p.feat, t) for p in G]
    verdicts = {c: [E.multi_of(p, c, delta) for p in G] for c in E.CANDIDATES}
    table, detail = RT.fit_table(buckets, verdicts)
    cfg = {"name": "t7_routing", "routing_version": RT.ROUTING_VERSION, "t6_gate_seal": gate["sha256"],
           "bucket_thresholds": {k: round(v, 6) for k, v in t.items()},
           "bucket_rule": "PHOTOMETRIC > MIXED (≥2 of LOWRES/NOISE/BLUR/COMPRESS) > LOWRES > NOISE > BLUR > COMPRESS > NONE",
           "table": table, "candidates": list(E.CANDIDATES), "original": RT.ORIGINAL,
           "min_bucket_n": RT.MIN_BUCKET_N, "lambda": GT.LAMBDA,
           "fallbacks": {"MODEL_FAILURE": "original", "TIMEOUT": "original", "RESOURCE_FAILURE": "original",
                         "INVALID_OUTPUT": "original (shape/dtype, non-finite, std < 1, no face on output)",
                         "UNSAFE_OR_UNCERTAIN": "original + escalate", "DO_NOT_ENHANCE": "original"},
           "timeout_s": 120.0,
           "calibration": {"population": "p4dev", "split": "dev", "data_kind": "SYNTHETIC",
                           "clean_refs": "population_noise_floor (112 dev_pilot references)"},
           "validity": "SYNTHETIC dev calibration. NOT valid for real-world inputs until re-calibrated on the rw dev split."}
    cfg["sha256"] = GT.gate_hash(cfg)
    write_json(repo_path("configs/phase4/t7_routing.json"), cfg)
    conf = Counter((RT.bucket_of(p.feat, t), p.profile) for p in P if GT.safety(p.feat) is None)
    write_json(repo_path("data/manifests/phase4/t7/calibration_report.json"), {
        "run": run_info(), "routing_sha256": cfg["sha256"], "t6_gate_seal": gate["sha256"], "thresholds": cfg["bucket_thresholds"],
        "table": table, "per_bucket": detail, "n_dev_enhance_candidate": len(G),
        "dev_bucket_vs_true_family": {f"{b}|{f}": n for (b, f), n in sorted(conf.items())}})
    print("T7 frozen:", cfg["sha256"][:12], json.dumps(table))


if __name__ == "__main__":
    main()
