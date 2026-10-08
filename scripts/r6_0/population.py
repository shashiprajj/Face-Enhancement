"""R6-0 — population study on the archive (Docker): identity benefit of each post-processing variant, and the
distribution of every verifier signal over ALL NAFNet-eligible outputs (not only labelled ones).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r6_0/population.py --split val --n 1500 [--workers 4]
    -> data/manifests/phase5_rapid/r6_0/population_<split>.jsonl   (then: gzip -9 it; the readers use .jsonl.gz)
       (+ --save: experiments/phase5_rapid/outputs/r6_0/pop_<split>/<mode>/<name>.png = chip | v0 | medluma | meanluma)

Build: the R5-8 candidate path (candidate config gate, reflect-filled chip, reflect-pad16 NAFNet); the R5-8 global
guard result is recorded (not applied), so each variant can be scored with or without it. Inputs: a seeded random
order of the split; the first n inputs per mode that the candidate gate admits. Identity vs the sharp GT, R5-ARCHIVE
method (GT landmarks; full_image outputs pasted back at the chip position).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fes_data.common import repo_path  # noqa: E402

ARCH = "Dataset-and-code/archive"
W: dict = {}
SAVE = False
FLIPS = (1, 0, -1)


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
    W["naf"], _ = nafnet_factory_pad16()


def _job(args):
    split, name = args
    import r5_8_dev_corpus as C
    import signals as S
    from compute_signals import variants
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    from fes_service.pipeline import face_chip, face_chip_filled
    det, cfg = W["det"], W["cfg"]
    inp = cv2.imread(str(repo_path(f"{ARCH}/input/{split}/{name}")), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(repo_path(f"{ARCH}/output/{split}/{name}")), cv2.IMREAD_COLOR)
    out_rows = []
    gf = det.detect(gt)
    if not gf:
        return out_rows
    glm = np.asarray(gf[0].landmarks, np.float64)
    emb = lambda im, lm: (W["sface"].embed(im, lm), W["dlib"].embed(im, lm))  # noqa: E731
    eg, e0 = emb(gt, glm), emb(inp, glm)
    b0 = (float(e0[0] @ eg[0]), float(e0[1] @ eg[1]))
    paths = {"fc": (inp, inp, None, None)}
    fr = P.preprocess(inp, det, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    if fr.status == P.OK and fr.detection is not None:
        lm = np.asarray(fr.detection.landmarks, np.float64)
        if fr.aligned and P.landmarks_plausible(lm):
            chip, _ = face_chip(inp, lm)
            filled, mask = face_chip_filled(inp, lm)
            paths["fi"] = (chip, filled if mask.any() else chip, mask if mask.any() else None, C.chip_geometry(lm))
    for p, (chip, work, mask, geo) in paths.items():
        f = RW.rw_features(work, det)
        f.update(NR.texture_features(work))
        if not NR.decide(f, cfg).eligible:
            continue
        runs = []
        t0 = time.perf_counter()
        for fl in (None,) + FLIPS:
            x = work if fl is None else np.ascontiguousarray(cv2.flip(work, fl))
            y = W["naf"](x, {})[0]
            y = y if fl is None else cv2.flip(y, fl)
            if mask is not None:
                y = y.copy()
                y[mask] = 0
            runs.append(y)
        infer = time.perf_counter() - t0
        v0, fl3 = runs[0], runs[1:]
        g = NR.artifact_guard(chip, v0, cfg, lambda x: len(det.detect(x)) > 0, bool(f.get("is_gray")))
        lg = NR.local_artifact_guard(chip, v0, cfg, bool(f.get("is_gray")))
        from run_signals import run_features
        rf = run_features(chip, runs, mask)
        tr = {"run_dot_c_max": max(x["dot_c"] for x in rf.values()), "run_hfcg_max": max(x["hfcg"] for x in rf.values()),
              "run_dot_l_max": max(x["dot_l"] for x in rf.values()),
              "run_dot_c": [round(rf[k]["dot_c"], 4) for k in ("v0", "vh", "vv", "vhv")]}
        if SAVE:
            d = repo_path(f"experiments/phase5_rapid/outputs/r6_0/pop_{split}/{p}")
            os.makedirs(d, exist_ok=True)
            vs = variants(work, v0, fl3, mask)
            cv2.imwrite(str(d / f"{name}.png"), np.hstack([chip, v0, vs["medluma"], vs["meanluma"]]))
        a = det.detect(chip)
        lmc = np.asarray(a[0].landmarks, np.float64) if a else None
        e_in = emb(chip, lmc) if lmc is not None else None
        valid = np.ones(chip.shape[:2], bool) if mask is None else ~mask
        ff = S.flip_consistency(v0, fl3, valid, chip)
        for k, out in variants(work, v0, fl3, mask).items():
            full = out if p == "fc" else C.paste_back(inp, out, geo)
            eo = emb(full, glm)
            r = {"split": split, "name": name, "mode": p, "variant": k, "chip_px": int(chip.shape[0]),
                 "oob": float(mask.mean()) if mask is not None else 0.0, "infer4_s": round(infer, 2),
                 "guard_pass": g.passed, "guard": "+".join(g.reasons), "r58_local_pass": lg.passed,
                 "r58_l_dot_c_max": lg.metrics.get("l_dot_c_max"), **tr,
                 "b0_sface": round(b0[0], 5), "b0_dlib": round(b0[1], 5),
                 "d_sface": round(float(eo[0] @ eg[0]) - b0[0], 5), "d_dlib": round(float(eo[1] @ eg[1]) - b0[1], 5)}
            if e_in is not None:
                ei = emb(out, lmc)
                r["i_sface_io"], r["i_dlib_io"] = round(float(ei[0] @ e_in[0]), 5), round(float(ei[1] @ e_in[1]), 5)
            r.update({kk: round(v, 6) if isinstance(v, float) else v
                      for kk, v in S.all_signals(chip, out, None, mask, det).items()})
            r.update(ff)
            out_rows.append(r)
    return out_rows


def _init_save(threads, save):
    global SAVE
    SAVE = save
    _init(threads)


def main(split, n, workers, threads, seed, save=False):
    out = repo_path(f"data/manifests/phase5_rapid/r6_0/population_{split}.jsonl")
    names = sorted(os.listdir(repo_path(f"{ARCH}/input/{split}")))
    random.Random(seed).shuffle(names)
    done, cnt = set(), {"fc": 0, "fi": 0}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add(r["name"])
            if r["variant"] == "v0":
                cnt[r["mode"]] += 1
    todo = [x for x in names if x not in done]
    print(f"{len(done)} done {cnt}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init_save, initargs=(threads, save)) as ex, open(out, "a") as fh:
        for i, rows in enumerate(ex.map(_job, [(split, x) for x in todo], chunksize=2)):
            for mode in ("fc", "fi"):
                mine = [r for r in rows if r["mode"] == mode]
                if mine and cnt[mode] < n:
                    fh.writelines(json.dumps(r) + "\n" for r in mine)
                    cnt[mode] += 1
            fh.flush()
            if i % 100 == 0:
                print(f"{i} scanned, eligible {cnt}, {time.time() - t0:.0f}s", flush=True)
            if min(cnt.values()) >= n:
                break
        ex.shutdown(cancel_futures=True)
    print("done", cnt, time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="save chip|v0|medluma|meanluma strips for visual tail review")
    a = ap.parse_args()
    main(a.split, a.n, a.workers, a.threads, a.seed, a.save)
