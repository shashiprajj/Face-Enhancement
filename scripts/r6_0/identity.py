"""R6-0 — identity-consistency signals (Docker: SFace + dlib).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r6_0/identity.py
    -> data/manifests/phase5_rapid/r6_0/identity.csv.gz

Per case x variant:
  i_sface_io / i_dlib_io   cos(input chip, candidate), both embedded with the INPUT chip's YuNet landmarks
                           (a SAFETY signal for identity drift; it cannot show that an enhancement is good)
  i_sface_gt / i_dlib_gt   archive face_chip cases only: cos(candidate, sharp GT) - cos(input, GT), GT landmarks
                           (the R5 benefit metric; evaluation only, never available to a verifier)
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_signals import variants  # noqa: E402
from sheet import load  # noqa: E402
from fes_bench import evaluators as EV  # noqa: E402
from fes_bench.metrics import SFace  # noqa: E402
from fes_data import preprocess as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "Dataset-and-code/archive"


def main():
    det = P.YuNetDetector(ROOT / "models/D01_yunet/face_detection_yunet_2026may.onnx", 0.7, 0.3, max_side=640)
    sf, dl = SFace(), EV.DlibResNet()
    emb = lambda im, lm: (sf.embed(im, lm), dl.embed(im, lm))  # noqa: E731
    cases = json.loads((ROOT / "data/manifests/phase5_rapid/r6_0/cases.json").read_text())
    meta = {json.loads(x)["id"]: json.loads(x) for x in
            (ROOT / "experiments/phase5_rapid/outputs/r6_0/cache/meta.jsonl").read_text().splitlines()}
    rows = []
    for i, c in enumerate(cases):
        if "error" in meta.get(c["id"], {"error": 1}):
            continue
        chip, work, v0, fl, mask = load(c["id"])
        a = det.detect(chip)
        if not a:
            continue
        lm = np.asarray(a[0].landmarks, np.float64)
        e_in = emb(chip, lm)
        gt = None
        if c["mode"] == "fc" and c["src"] in ("dev", "dev_mod", "test_r56", "test_r57", "test_r58", "test_r510"):
            split = "val" if c["src"].startswith("dev") else "test"
            g = cv2.imread(str(ARCH / "output" / split / c["name"]), cv2.IMREAD_COLOR)
            gf = det.detect(g)
            if gf:
                glm = np.asarray(gf[0].landmarks, np.float64)
                e_g = emb(g, glm)
                e0 = emb(chip, glm)
                gt = (glm, e_g, float(e0[0] @ e_g[0]), float(e0[1] @ e_g[1]))
        for k, out in variants(work, v0, fl, mask).items():
            e = emb(out, lm)
            r = {"id": c["id"], "variant": k, "i_sface_io": round(float(e[0] @ e_in[0]), 5),
                 "i_dlib_io": round(float(e[1] @ e_in[1]), 5)}
            if gt is not None:
                eo = emb(out, gt[0])
                r["i_sface_gt"] = round(float(eo[0] @ gt[1][0]) - gt[2], 5)
                r["i_dlib_gt"] = round(float(eo[1] @ gt[1][1]) - gt[3], 5)
            rows.append(r)
        if i % 100 == 0:
            print(i, flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(ROOT / "data/manifests/phase5_rapid/r6_0/identity.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
