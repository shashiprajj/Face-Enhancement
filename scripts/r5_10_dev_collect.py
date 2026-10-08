"""R5-10 — DEV evidence for the NAFNet blur-eligibility boundary (DEV data ONLY; nothing here touches archive TEST,
ztest, the 200 real photos or the owner's images).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_10_dev_collect.py {archive|clean|p4} [--workers 3]
    → experiments/phase5_rapid/outputs/r5_10/<part>/records.jsonl (resumable) → data/manifests/phase5_rapid/r5_10/

The path is the R5-8 candidate exactly (sealed configs/phase5/r5_8_nafnet_candidate.json, nafnet_factory_pad16,
reflect fill for full_image, global artifact guard, local-patch guard) with ONE difference: the two blur rules
(NOT_BLUR / MILD_BLUR) are switched off in an in-memory copy of the config, so NAFNet also runs on inputs that every
other gate admits but the blur boundary rejects. Every other gate is unchanged and still authoritative.

Parts
  archive  archive `val` (19,290 pairs), face_chip (fc) + full_image (fi): NAFNet on every input whose R5-8 reason is
           NAFNET_REJECT_MILD_BLUR or NAFNET_REJECT_NOT_BLUR (the R5-8-eligible inputs are already in
           data/manifests/phase5_rapid/r5_8/{corpus,local_features}.csv.gz and are not re-run).
           Identity: SFace / dlib vs the sharp GT with GT landmarks, Δ vs B0 (R5-ARCHIVE / R5-8 method).
  clean    the SHARP archive `val` GT images as inputs (fc): which clean faces would a wider boundary admit, and what
           does NAFNet do to them (Δ = cos(out, GT) − cos(GT, GT)).
  p4       p4dev (896) + r5gap (224) synthetic DEV pairs (12 degradation families × severities, IOD ≈ 40–85 px):
           features + R5-8 reason for all; NAFNet on R5-8-eligible and MILD_BLUR inputs (fc = the image). Identity vs
           the clean reference image (YuNet landmarks on each image), Δ vs the input.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, write_csv_gz

sys.path.insert(0, str(repo_path("scripts")))
ARCH = "Dataset-and-code/archive"
IMG = "experiments/phase5_rapid/outputs/r5_10"
OUT = "data/manifests/phase5_rapid/r5_10"
BLUR_REASONS = ("NAFNET_REJECT_MILD_BLUR", "NAFNET_REJECT_NOT_BLUR")
FEAT_KEYS = ("iod_px", "det_score", "lapd1_c", "noise_sigma", "blockiness", "luma_mean", "rms_contrast", "is_gray",
             "pixelation", "hf_peak_frac", "hf_energy")
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
    cfg = NR.load_config(NR.CANDIDATE_CONFIG)                      # sealed R5-8 candidate (with local guard)
    assert "local_artifact" in cfg
    relaxed = copy.deepcopy(cfg)
    relaxed["thresholds"]["blur_max"] = relaxed["thresholds"]["clean_lapd1_lo"] = float("inf")
    W["cfg"], W["relaxed"] = cfg, relaxed
    W["naf"], _ = nafnet_factory_pad16()


def _feat(work):
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    f = RW.rw_features(work, W["det"])
    f.update(NR.texture_features(work))
    return f


def _run(r, p, chip, work, mask, f, sub, name):
    """R5-8 candidate path after the gate: NAFNet → global guard → local guard (pipeline._nafnet_path order)."""
    from fes_service import nafnet_route as NR
    cfg, det = W["cfg"], W["det"]
    t0 = time.perf_counter()
    out, _ = W["naf"](work, {})
    r[f"{p}_infer_s"] = round(time.perf_counter() - t0, 3)
    if mask is not None:
        out = out.copy()
        out[mask] = 0
    g = NR.artifact_guard(chip, out, cfg, lambda x: len(det.detect(x)) > 0, bool(f.get("is_gray")))
    r[f"{p}_guard_pass"], r[f"{p}_guard"] = g.passed, "+".join(g.reasons)
    for k in ("chroma_p99", "chroma_p999", "hf_color_ratio", "face_found"):
        r[f"{p}_g_{k}"] = g.metrics.get(k)
    lf = NR.local_features(chip, out)                              # always computed (also when the global guard fails)
    r[f"{p}_l_dot_c_max"] = lf["l_dot_c_max"]
    r[f"{p}_local_pass"] = bool(lf["l_dot_c_max"] <= cfg["local_artifact"]["limits"]["l_dot_c_max"])
    os.makedirs(repo_path(f"{IMG}/{sub}/{p}"), exist_ok=True)
    cv2.imwrite(str(repo_path(f"{IMG}/{sub}/{p}/{name}.png")), out)
    return out


def _gate(r, p, f):
    from fes_service import nafnet_route as NR
    d = NR.decide(f, W["cfg"])
    rl = NR.decide(f, W["relaxed"])
    r[f"{p}_reason"], r[f"{p}_relaxed_eligible"] = d.reason, rl.eligible
    for k in FEAT_KEYS:
        v = f.get(k)
        r[f"{p}_{k}"] = (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
    return d


def _embed(img, lm):
    return W["sface"].embed(img, lm), W["dlib"].embed(img, lm)


def job_archive(name):
    import r5_8_dev_corpus as C
    from fes_data import preprocess as P
    from fes_service.pipeline import face_chip, face_chip_filled
    det = W["det"]
    inp = cv2.imread(str(repo_path(f"{ARCH}/input/val/{name}")), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(repo_path(f"{ARCH}/output/val/{name}")), cv2.IMREAD_COLOR)
    r = {"name": name}
    faces = det.detect(gt)
    glm = np.asarray(faces[0].landmarks, np.float64) if faces else None
    eg = e0 = None
    if glm is not None:
        eg, e0 = _embed(gt, glm), _embed(inp, glm)
        r["b0_sface"], r["b0_dlib"] = float(e0[0] @ eg[0]), float(e0[1] @ eg[1])
    paths = {"fc": (inp, inp, None, None)}
    fr = P.preprocess(inp, det, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    if fr.status == P.OK and fr.detection is not None:
        lm = np.asarray(fr.detection.landmarks, np.float64)
        if fr.aligned and P.landmarks_plausible(lm):
            chip, _ = face_chip(inp, lm)
            filled, mask = face_chip_filled(inp, lm)
            paths["fi"] = (chip, filled if mask.any() else chip, mask if mask.any() else None, C.chip_geometry(lm))
    for p, (chip, work, mask, geo) in paths.items():
        f = _feat(work)
        d = _gate(r, p, f)
        if d.reason not in BLUR_REASONS or not r[f"{p}_relaxed_eligible"]:
            continue
        out = _run(r, p, chip, work, mask, f, "archive", name)
        if glm is not None:
            full = out if p == "fc" else C.paste_back(inp, out, geo)
            eo = _embed(full, glm)
            r[f"{p}_d_sface"] = round(float(eo[0] @ eg[0]) - r["b0_sface"], 5)
            r[f"{p}_d_dlib"] = round(float(eo[1] @ eg[1]) - r["b0_dlib"], 5)
    return r


def job_clean(name):
    det = W["det"]
    gt = cv2.imread(str(repo_path(f"{ARCH}/output/val/{name}")), cv2.IMREAD_COLOR)
    r = {"name": name}
    f = _feat(gt)
    d = _gate(r, "fc", f)
    if not r["fc_relaxed_eligible"] or d.reason == "NAFNET_REJECT_NOT_BLUR":
        return r                                        # NOT_BLUR is above every candidate boundary
    out = _run(r, "fc", gt, gt, None, f, "clean", name)
    faces = det.detect(gt)
    if faces:
        lm = np.asarray(faces[0].landmarks, np.float64)
        eg, eo = _embed(gt, lm), _embed(out, lm)
        r["fc_d_sface"] = round(float(eo[0] @ eg[0]) - float(eg[0] @ eg[0]), 5)
        r["fc_d_dlib"] = round(float(eo[1] @ eg[1]) - float(eg[1] @ eg[1]), 5)
    return r


def job_p4(row):
    det = W["det"]
    img = cv2.imread(str(repo_path(row["degraded_path"])), cv2.IMREAD_COLOR)
    ref = cv2.imread(str(repo_path(row["ref_path"])), cv2.IMREAD_COLOR)
    r = {"name": row["pair_id"], "population": row["population"], "cell": row["cell"], "profile": row["profile"],
         "severity": row["severity"], "img_px": int(img.shape[1])}
    f = _feat(img)
    d = _gate(r, "fc", f)
    if d.reason not in ("NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR", "NAFNET_REJECT_MILD_BLUR"):
        return r
    out = _run(r, "fc", img, img, None, f, "p4", row["pair_id"])
    fi, fref = det.detect(img), det.detect(ref)
    if fi and fref:
        lm, lr = np.asarray(fi[0].landmarks, np.float64), np.asarray(fref[0].landmarks, np.float64)
        er, e0, eo = _embed(ref, lr), _embed(img, lm), _embed(out, lm)
        r["b0_sface"], r["b0_dlib"] = float(e0[0] @ er[0]), float(e0[1] @ er[1])
        r["fc_d_sface"] = round(float(eo[0] @ er[0]) - r["b0_sface"], 5)
        r["fc_d_dlib"] = round(float(eo[1] @ er[1]) - r["b0_dlib"], 5)
    return r


def main(part, workers, threads, limit):
    os.makedirs(repo_path(f"{IMG}/{part}"), exist_ok=True)
    os.makedirs(repo_path(OUT), exist_ok=True)
    rec = repo_path(f"{IMG}/{part}/records.jsonl")
    done = set()
    if rec.exists():
        for line in rec.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["name"])
            except json.JSONDecodeError:
                pass
    if part == "p4":
        items = (read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))
                 + read_csv_gz(repo_path("data/manifests/phase5_rapid/gap/population_r5gap.csv.gz")))
        todo, fn = [x for x in items[: limit or None] if x["pair_id"] not in done], job_p4
    else:
        names = sorted(os.listdir(repo_path(f"{ARCH}/input/val")))[: limit or None]
        todo, fn = [n for n in names if n not in done], (job_archive if part == "archive" else job_clean)
    print(f"{part}: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads,)) as ex, open(rec, "a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(fn, todo, chunksize=4)):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if i % 500 == 0:
                print(f"{part} {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    rows = {}
    for line in rec.read_text(encoding="utf-8").splitlines():
        try:
            x = json.loads(line)
            rows[x["name"]] = x
        except json.JSONDecodeError:
            pass
    rows = [rows[k] for k in sorted(rows)]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv_gz(repo_path(f"{OUT}/dev_{part}.csv.gz"), [{k: r.get(k, "") for k in cols} for r in rows], cols)
    print(f"{part} done: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("archive", "clean", "p4"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.part, a.workers, a.threads, a.limit)
