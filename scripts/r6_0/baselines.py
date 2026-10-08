"""R6-0 — the EXISTING verifiers' decisions on every corpus case (host), using the service's own functions unchanged:
  r5_6  = live R5-6 artifact guard (configs/phase5/r5_nafnet_route.json)
  r5_8  = R5-8/R5-10 candidate: global guard (candidate values) + local-patch guard l_dot_c_max <= 24.45
Applied to the output v0 of every case (whatever build produced it), so all verifiers see the same candidates.

    PYTHONPATH=src python scripts/r6_0/baselines.py  ->  data/manifests/phase5_rapid/r6_0/baselines.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet import load  # noqa: E402
from fes_data import preprocess as P  # noqa: E402
from fes_service import nafnet_route as NR  # noqa: E402
from fes_service import rwgate as RW  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    det = P.YuNetDetector(ROOT / "models/D01_yunet/face_detection_yunet_2026may.onnx", 0.7, 0.3, max_side=640)
    live, cand = NR.load_config(NR.CONFIG), NR.load_config(NR.CANDIDATE_CONFIG)
    face = lambda x: len(det.detect(x)) > 0  # noqa: E731
    out = {}
    for c in json.loads((ROOT / "data/manifests/phase5_rapid/r6_0/cases.json").read_text()):
        chip, work, v0, fl, mask = load(c["id"])
        if v0 is None:
            continue
        gray = bool(RW.rw_features(work, det).get("is_gray"))
        g6 = NR.artifact_guard(chip, v0, live, face, gray)
        g8 = NR.artifact_guard(chip, v0, cand, face, gray)
        l8 = NR.local_artifact_guard(chip, v0, cand, gray)
        out[c["id"]] = {"is_gray": gray, "r5_6_pass": g6.passed, "r5_6_reasons": g6.reasons,
                        "r5_8_global_pass": g8.passed, "r5_8_local_pass": l8.passed,
                        "r5_8_pass": g8.passed and l8.passed, "r5_8_reasons": g8.reasons + l8.reasons,
                        "l_dot_c_max": l8.metrics.get("l_dot_c_max")}
    (ROOT / "data/manifests/phase5_rapid/r6_0/baselines.json").write_text(json.dumps(out, indent=0))
    print(len(out), sum(v["r5_8_pass"] for v in out.values()), sum(v["r5_6_pass"] for v in out.values()))


if __name__ == "__main__":
    main()
