"""P4-T8 — CPU cost of the quality gate (feature extraction + rule evaluation) per input. Descriptive, shared host.

    scripts/dock.sh python scripts/p4_gate_latency.py
"""
import time

import cv2
import numpy as np

from fes_bench import quality as Q
from fes_data.common import read_csv_gz, repo_path, run_info, write_json
from fes_data.preprocess import YuNetDetector

cv2.setNumThreads(1)
det = YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
pop = read_csv_gz(repo_path("data/manifests/phase3/population_main.csv.gz"))
by_size = {}
for p in pop[:448]:
    img = cv2.imread(str(repo_path(p["degraded_path"])), cv2.IMREAD_COLOR)
    Q.features(img, det)                                   # warm
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        Q.features(img, det)
        t.append(time.perf_counter() - t0)
    by_size.setdefault(p["out_w"], []).append(min(t) * 1000)
res = {"run": run_info(), "threads": 1, "note": "min of 3 per image; host shared with other containers; CPU only",
       "ms_per_input_by_size": {s: {"n": len(v), "p50": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2)}
                                for s, v in sorted(by_size.items(), key=lambda x: int(x[0]))}}
write_json(repo_path("data/manifests/phase4/perf/gate_latency.json"), res)
print(res["ms_per_input_by_size"])
