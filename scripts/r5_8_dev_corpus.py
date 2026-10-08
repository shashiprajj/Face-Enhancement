"""R5-8 — DEV artifact corpus: the R5-7 candidate (profile r5_8, corpus-stage config, NO local guard) on the WHOLE
archive `val` split in both input modes. Every NAFNet output is saved (passed or blocked by the global guard).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_8_dev_corpus.py [--workers 3] [--limit N]
    → experiments/phase5_rapid/outputs/r5_8/corpus/{records.jsonl, fc/<name>.png, fi/<name>.png (+ fi_chip/)}
    → data/manifests/phase5_rapid/r5_8/corpus.csv.gz

The candidate path is reproduced with the service's own functions (pipeline.face_chip / face_chip_filled,
rwgate.rw_features, nafnet_route.texture_features / decide / artifact_guard, models.nafnet_factory_pad16) and the
sealed candidate config. Identity: SFace/dlib vs the sharp GT with GT landmarks, Δ vs B0 (R5-ARCHIVE method; full_image
outputs pasted back at the chip position). DEV only — archive TEST / ztest / real photos are not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import repo_path, write_csv_gz

ARCH = "Dataset-and-code/archive"
IMG = "experiments/phase5_rapid/outputs/r5_8/corpus"
OUT = "data/manifests/phase5_rapid/r5_8"
W: dict = {}


def _init(threads):
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(threads)
    from fes_bench import evaluators as EV
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service.models import nafnet_factory_pad16
    W["det"] = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    W["sface"], W["dlib"] = SFace(), EV.DlibResNet()
    W["cfg"] = NR.load_config(NR.CANDIDATE_CONFIG)
    assert "local_artifact" not in W["cfg"], "corpus must be generated without the local guard"
    W["naf"], _ = nafnet_factory_pad16()


def chip_geometry(lm):
    from fes_bench.evaluators import DLIB_CHIP_TEMPLATE_150
    from fes_data import preprocess as P
    M = P.umeyama_similarity(lm, DLIB_CHIP_TEMPLATE_150)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    cx, cy = np.linalg.inv(M[:, :2]) @ (np.array([75.0, 75.0]) - M[:, 2])
    side = int(max(8, round(150.0 / scale)))
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def paste_back(inp, out, geo):
    x0, y0, side = geo
    if out.shape[0] != side:
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)
    h, w = inp.shape[:2]
    res = inp.copy()
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    res[ya:yb, xa:xb] = out[ya - y0:yb - y0, xa - x0:xa - x0 + (xb - xa)]
    return res


def _job(name):
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    from fes_service.pipeline import face_chip, face_chip_filled
    det, cfg = W["det"], W["cfg"]
    inp = cv2.imread(str(repo_path(f"{ARCH}/input/val/{name}")), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(repo_path(f"{ARCH}/output/val/{name}")), cv2.IMREAD_COLOR)
    r = {"name": name}
    faces = det.detect(gt)
    glm = np.asarray(faces[0].landmarks, np.float64) if faces else None
    if glm is not None:
        eg = (W["sface"].embed(gt, glm), W["dlib"].embed(gt, glm))
        e0 = (W["sface"].embed(inp, glm), W["dlib"].embed(inp, glm))
        r["b0_sface"], r["b0_dlib"] = float(e0[0] @ eg[0]), float(e0[1] @ eg[1])
    paths = {"fc": (inp, inp, None, None)}
    fr = P.preprocess(inp, det, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    if fr.status == P.OK and fr.detection is not None:
        lm = np.asarray(fr.detection.landmarks, np.float64)
        if fr.aligned and P.landmarks_plausible(lm):
            chip, _ = face_chip(inp, lm)
            filled, mask = face_chip_filled(inp, lm)
            paths["fi"] = (chip, filled if mask.any() else chip, mask if mask.any() else None, chip_geometry(lm))
    for p, (chip, work, mask, geo) in paths.items():
        f = RW.rw_features(work, det)
        f.update(NR.texture_features(work))
        d = NR.decide(f, cfg)
        r[f"{p}_reason"], r[f"{p}_eligible"] = d.reason, d.eligible
        r[f"{p}_chip_px"], r[f"{p}_oob_frac"] = int(chip.shape[0]), round(float(mask.mean()), 4) if mask is not None else 0.0
        r[f"{p}_is_gray"] = bool(f.get("is_gray"))
        r[f"{p}_lapd1"] = f.get("lapd1_c")
        if not d.eligible:
            continue
        t0 = time.perf_counter()
        out, _ = W["naf"](work, {})
        r[f"{p}_infer_s"] = round(time.perf_counter() - t0, 3)
        if mask is not None:
            out = out.copy()
            out[mask] = 0
        g = NR.artifact_guard(chip, out, cfg, lambda x: len(det.detect(x)) > 0, bool(f.get("is_gray")))
        r[f"{p}_guard_pass"], r[f"{p}_guard"] = g.passed, "+".join(g.reasons)
        for k, v in g.metrics.items():
            r[f"{p}_g_{k}"] = v
        cv2.imwrite(str(repo_path(f"{IMG}/{p}/{name}.png")), out)
        if p == "fi":
            cv2.imwrite(str(repo_path(f"{IMG}/fi_chip/{name}.png")), chip)
        if glm is not None:
            full = out if p == "fc" else paste_back(inp, out, geo)
            eo = (W["sface"].embed(full, glm), W["dlib"].embed(full, glm))
            r[f"{p}_d_sface"] = round(float(eo[0] @ eg[0]) - r["b0_sface"], 5)
            r[f"{p}_d_dlib"] = round(float(eo[1] @ eg[1]) - r["b0_dlib"], 5)
    return r


def main(workers, threads, limit):
    for sub in ("fc", "fi", "fi_chip"):
        os.makedirs(repo_path(f"{IMG}/{sub}"), exist_ok=True)
    rec = repo_path(f"{IMG}/records.jsonl")
    done = set()
    if rec.exists():
        for line in rec.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["name"])
            except json.JSONDecodeError:
                pass
    names = sorted(os.listdir(repo_path(f"{ARCH}/input/val")))[: limit or None]
    todo = [n for n in names if n not in done]
    print(f"corpus: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads,)) as ex, open(rec, "a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(_job, todo, chunksize=4)):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if i % 500 == 0:
                print(f"corpus {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    rows = {}
    for line in rec.read_text(encoding="utf-8").splitlines():
        try:
            x = json.loads(line)
            rows[x["name"]] = x
        except json.JSONDecodeError:
            pass
    rows = [rows[k] for k in sorted(rows)]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv_gz(repo_path(f"{OUT}/corpus.csv.gz"), [{k: r.get(k, "") for k in cols} for r in rows], cols)
    print(f"corpus done: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.workers, a.threads, a.limit)
