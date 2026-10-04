"""Phase 5 Rapid — side-by-side outputs on REAL photos (no identity labels → visual evidence only).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_realphotos_visual.py [--n 16]

Selection (deterministic): photos whose largest face passes the frozen safety rules (detection OK, score ≥ 0.85,
|yaw| ≤ 0.25), ordered by a seeded hash, first `n`. Per face, production-style geometry only (no recorded transforms):
  * A4        on the R5-1 service chip (integer crop, native scale) → 224
  * A6/A7/A8  on the FFHQ-512 alignment of the detected landmarks (fes_data.preprocess.align_face), model in [-1,1]
Every model runs on every selected face regardless of the gate, to show what each model does. Sheet columns:
input chip | A4 | GFPGAN | CodeFormer w=0.5 | CodeFormer w=1.0, label = frozen gate decision/route.
Writes experiments/phase5_rapid/outputs/realphotos/sheet_*.jpg (git-ignored) and
data/manifests/phase5_rapid/realphotos/visual.json (selection, timings).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time

import cv2
import numpy as np
import torch

from fes_bench import adapters as AD
from fes_bench import gate as GT
from fes_bench import quality as Q
from fes_data import preprocess as P
from fes_data.common import repo_path, run_info, write_json
from fes_service.pipeline import EnhancementPipeline, face_chip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(2)
    pipe = EnhancementPipeline(lambda i, c: None, timeout_s=30)
    files = sorted(glob.glob(str(repo_path("Dataset-and-code/dataset/*.jpg"))),
                   key=lambda f: hashlib.sha256(f"r5-real|{os.path.basename(f)}".encode()).hexdigest())
    sel = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        fr = P.preprocess(img, pipe.det, pipe.det_cfg, input_mode="face_crop")[0]
        if fr.status != P.OK or not fr.aligned:
            continue
        lm = np.asarray(fr.detection.landmarks, np.float64)
        chip, side = face_chip(img, lm)
        feat = Q.features(chip, pipe.det)
        if GT.safety(feat):
            continue
        rec = pipe.router.plan(feat)
        f512 = P.align_face(img, lm, 512)[0]
        sel.append({"image": os.path.basename(f), "chip": chip, "f512": f512, "decision": rec.decision,
                    "bucket": rec.bucket, "route": rec.route, "chip_px": int(chip.shape[0])})
        if len(sel) == a.n:
            break

    outs = {k: [] for k in ("A4", "A6", "A7", "A8")}
    times = {}
    a4 = AD.build("A4")
    a4.load()
    t0 = time.perf_counter()
    for s in sel:
        outs["A4"].append(a4.enhance(s["chip"], {"s": s["chip"].shape[0]})[0])
    times["A4"] = round((time.perf_counter() - t0) / len(sel), 2)
    del a4
    for cid in ("A6", "A7", "A8"):
        ad = AD.build(cid)
        ad.load()
        t0 = time.perf_counter()
        for s in sel:
            x = AD.bgr_u8_to_rgb_t(s["f512"]) * 2.0 - 1.0
            with torch.inference_mode():
                y = ad._forward(x)
            outs[cid].append(cv2.resize(AD.rgb_t_to_bgr_u8(y, -1.0, 1.0), (224, 224), interpolation=cv2.INTER_AREA))
        times[cid] = round((time.perf_counter() - t0) / len(sel), 2)
        del ad

    out_dir = repo_path("experiments/phase5_rapid/outputs/realphotos")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(sel):
        cells = [cv2.resize(s["chip"], (224, 224), interpolation=cv2.INTER_CUBIC),
                 cv2.resize(s["f512"], (224, 224), interpolation=cv2.INTER_AREA)] + [outs[k][i] for k in outs]
        row = np.hstack(cells)
        lab = f"{s['image']}  chip {s['chip_px']}px  gate {s['decision'][:14]} {s['bucket']} -> {s['route']}"
        cv2.putText(row, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
        rows.append(row)
    head = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
    for j, t in enumerate(["input chip", "input FFHQ-aligned", "A4 SwinIR", "GFPGAN v1.4", "CodeFormer w=0.5",
                           "CodeFormer w=1.0"]):
        cv2.putText(head, t, (6 + 224 * j, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for k in range(0, len(rows), 8):
        cv2.imwrite(str(out_dir / f"sheet_{k // 8 + 1}.jpg"), np.vstack([head] + rows[k:k + 8]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    write_json(repo_path("data/manifests/phase5_rapid/realphotos/visual.json"),
               {"run": run_info(), "n": len(sel), "sec_per_face": times,
                "selection": [{k: v for k, v in s.items() if k not in ("chip", "f512")} for s in sel]})
    print(times, [s["image"] for s in sel])


if __name__ == "__main__":
    main()
