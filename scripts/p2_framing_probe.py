"""P2-T3 design probe 2 — choose template zoom and output size so references need no upsampling and
no out-of-source padding. Uses the landmarks recorded for every candidate in references_v1 (run 1).

    scripts/dock.sh python scripts/p2_framing_probe.py
Output: data/manifests/framing_probe_v1.json
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from fes_data import common as C
from fes_data.preprocess import FFHQ_TEMPLATE_512, umeyama_similarity

ZOOMS = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
CENTERS = ["frame", "centroid"]
SIZES = [192, 224, 256]


def main() -> None:
    rows = [r for r in C.read_csv_gz(C.repo_path("data/manifests/references_v1.csv.gz")) if r["det_landmarks"]]
    lms = [np.array(json.loads(r["det_landmarks"])) for r in rows]
    ones = np.full((224, 224), 255, np.uint8)
    res = {}
    for size in SIZES:
      for center in CENTERS:
        for z in ZOOMS:
            c0 = np.array([256.0, 256.0]) if center == "frame" else FFHQ_TEMPLATE_512.mean(0)
            t = (FFHQ_TEMPLATE_512 - c0) * z * (size / 512) + size / 2
            scales, oobs = [], []
            for lm in lms:
                M = umeyama_similarity(lm, t)
                scales.append(float(np.sqrt(abs(np.linalg.det(M[:, :2])))))
                v = cv2.warpAffine(ones, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
                oobs.append(float((v == 0).mean()))
            s, o = np.array(scales), np.array(oobs)
            res[f"{size}@z{z}@{center}"] = {
                "scale_p50": round(float(np.median(s)), 3),
                "frac_scale_le_1": round(float((s <= 1.0).mean()), 4),
                "frac_oob_le_0.01": round(float((o <= 0.01).mean()), 4),
                "frac_both": round(float(((s <= 1.0) & (o <= 0.01)).mean()), 4),
                "frac_both_oob2": round(float(((s <= 1.0) & (o <= 0.02)).mean()), 4),
                "oob_p50": round(float(np.median(o)), 4), "oob_p90": round(float(np.percentile(o, 90)), 4),
            }
    out = {"candidates": len(lms), "grid": res, "run": C.run_info()}
    C.write_json(C.repo_path("data/manifests/framing_probe_v1.json"), out)
    best = sorted(res.items(), key=lambda kv: -kv[1]["frac_both"])[:10]
    for k, v in best:
        print(k, v)


if __name__ == "__main__":
    main()
