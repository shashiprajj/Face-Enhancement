"""R6 follow-up (DEV only) — does running NAFNet again on its own output help heavy blur?

    docker: python scripts/r6_0/multipass.py [--n 160]
    -> data/manifests/phase5_rapid/r6_0/multipass_val.json
       experiments/phase5_rapid/outputs/r6_0/multipass/<name>.png  (input | 1 pass | 2 passes | 3 passes | GT)

archive val face_chip inputs, seeded order, split by blur score into heavy (< 0.0015), strong (0.0015-0.003),
moderate (0.003-0.008). Identity: SFace / dlib cos(output, GT) - cos(input, GT), GT landmarks (R5-ARCHIVE method).
Per pass also the colour-dottedness tracer vs the input (does a second pass destabilise?).
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from fes_data.common import repo_path

BANDS = {"heavy": (0, 0.0015), "strong": (0.0015, 0.003), "moderate": (0.003, 0.008)}


def main(n):
    import torch
    torch.set_num_threads(8)
    from fes_bench import evaluators as EV
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    from fes_service.models import nafnet_factory_pad16
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    sf, dl = SFace(), EV.DlibResNet()
    naf, _ = nafnet_factory_pad16()
    out_dir = repo_path("experiments/phase5_rapid/outputs/r6_0/multipass")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(os.listdir(repo_path("Dataset-and-code/archive/input/val")))
    random.Random(7).shuffle(names)
    cnt = {b: 0 for b in BANDS}
    rows = []
    t0 = time.time()
    for name in names:
        if min(cnt.values()) >= n:
            break
        inp = cv2.imread(str(repo_path(f"Dataset-and-code/archive/input/val/{name}")))
        gt = cv2.imread(str(repo_path(f"Dataset-and-code/archive/output/val/{name}")))
        lap = RW.rw_features(inp, det).get("lapd1_c")
        band = next((b for b, (lo, hi) in BANDS.items() if lap is not None and lo <= lap < hi), None)
        if band is None or cnt[band] >= n:
            continue
        gf = det.detect(gt)
        if not gf:
            continue
        lm = np.asarray(gf[0].landmarks, np.float64)
        eg = (sf.embed(gt, lm), dl.embed(gt, lm))
        e0 = (sf.embed(inp, lm), dl.embed(inp, lm))
        b0 = (float(e0[0] @ eg[0]), float(e0[1] @ eg[1]))
        r = {"name": name, "band": band, "lapd1": lap}
        x, outs = inp, []
        for k in (1, 2, 3):
            x = naf(x, {})[0]
            outs.append(x)
            e = (sf.embed(x, lm), dl.embed(x, lm))
            r[f"p{k}_dsface"] = round(float(e[0] @ eg[0]) - b0[0], 5)
            r[f"p{k}_ddlib"] = round(float(e[1] @ eg[1]) - b0[1], 5)
            r[f"p{k}_dotc"] = round(NR.dot_c_max(inp, x), 3)
        rows.append(r)
        cnt[band] += 1
        if cnt[band] <= 12:
            cv2.imwrite(str(out_dir / f"{band}_{name}.png"), np.hstack([inp] + outs + [gt]))
        if len(rows) % 50 == 0:
            print(len(rows), cnt, f"{time.time() - t0:.0f}s", flush=True)
    summ = {}
    for b in BANDS:
        sel = [r for r in rows if r["band"] == b]
        summ[b] = {"n": len(sel)}
        for k in (1, 2, 3):
            d = np.array([r[f"p{k}_dsface"] for r in sel])
            summ[b][f"pass{k}"] = {"dsface": float(d.mean()), "ddlib": float(np.mean([r[f"p{k}_ddlib"] for r in sel])),
                                   "harmed": float(np.mean(d < -0.05)), "improved": float(np.mean(d > 0.05)),
                                   "dotc_gt20": float(np.mean([r[f"p{k}_dotc"] > 20.2863 for r in sel]))}
    repo_path("data/manifests/phase5_rapid/r6_0/multipass_val.json").write_text(
        json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    main(ap.parse_args().n)
