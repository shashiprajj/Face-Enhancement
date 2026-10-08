"""R6-0 — every verifier signal for every corpus case, for the original NAFNet output and the post-processing variants
(host; numpy/OpenCV + the repo's YuNet wrapper).

    PYTHONPATH=src python scripts/r6_0/compute_signals.py
    -> data/manifests/phase5_rapid/r6_0/signals.csv.gz   (one row per case x variant)

Variants (all from the SAME four NAFNet runs in the cache; no new model):
  v0        the output the R5 build returned (reproduced bit-exactly)
  luma      v0 luminance + input chrominance
  mean      mean of the 4 flip runs           meanluma  mean + input chrominance
  med       median of the 4 flip runs         medluma   median + input chrominance
The flip-consistency family (f_*) is a property of the four runs, so it is the same for every variant.
"""
import gzip
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals as S  # noqa: E402
from sheet import load  # noqa: E402
from fes_data import preprocess as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/manifests/phase5_rapid/r6_0/signals.csv.gz"


def variants(work, v0, fl, mask):
    vs = {"v0": v0, "luma": S.luma_only(work, v0), "mean": S.flip_mean(v0, fl), "med": S.flip_median(v0, fl)}
    vs["meanluma"], vs["medluma"] = S.luma_only(work, vs["mean"]), S.luma_only(work, vs["med"])
    if mask is not None:
        for k in vs:
            vs[k] = vs[k].copy()
            vs[k][mask] = 0
    return vs


def main():
    cv2.setNumThreads(4)
    det = P.YuNetDetector(ROOT / "models/D01_yunet/face_detection_yunet_2026may.onnx", 0.7, 0.3, max_side=640)
    cases = json.loads((ROOT / "data/manifests/phase5_rapid/r6_0/cases.json").read_text())
    meta = {}
    for line in (ROOT / "experiments/phase5_rapid/outputs/r6_0/cache/meta.jsonl").read_text().splitlines():
        r = json.loads(line)
        meta[r["id"]] = r
    rows = []
    for i, c in enumerate(cases):
        m = meta.get(c["id"], {})
        if "error" in m or not m:
            rows.append({"id": c["id"], "variant": "v0", "error": m.get("error", "not run")})
            continue
        chip, work, v0, fl, mask = load(c["id"])
        f_flip = S.flip_consistency(v0, fl, np.ones(chip.shape[:2], bool) if mask is None else ~mask, chip)
        for k, out in variants(work, v0, fl, mask).items():
            f = S.all_signals(chip, out, None, mask, det)
            f.update(f_flip)
            rows.append({"id": c["id"], "variant": k, "chip_px": m["chip_px"], "oob": m["oob"],
                         "saved_maxdiff": m.get("saved_maxdiff"),
                         **{a: (round(b, 6) if isinstance(b, float) else b) for a, b in f.items()}})
        if i % 100 == 0:
            print(i, len(cases), flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
