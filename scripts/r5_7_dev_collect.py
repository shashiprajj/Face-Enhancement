"""R5-7 — DEV evidence for (1) artifact-guard calibration and (2) the full_image framing investigation.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_7_dev_collect.py collect [--workers 3] [--limit N]
    → experiments/phase5_rapid/outputs/r5_7/val/records.jsonl (resumable) → data/manifests/phase5_rapid/r5_7/val.csv.gz

DEV data ONLY: the archive `val` split (19,290 pairs; never used before — R5-ARCHIVE/R5-6 used test/ztest; train/val
were reserved). NOT used: archive test, ztest, the 200 real photos, any R5-6 result.

Per val pair, with the SERVICE's own functions (read-only use; nothing is modified):
  path A  face_chip   chip = the 128-px archive image (what input_mode=face_chip receives)
  path B  full_image  YuNet + preprocess + fes_service.pipeline.face_chip (the service's re-frame), geometry recorded
  features            rw_features + texture_features on each chip → nafnet_route.decide with the SEALED R5-5 config
  NAFNet              run on the chip of every path that is eligible (same factory as the service), then the R5-5
                      artifact_guard (sealed thresholds) → every guard metric + extra candidate signals
  labels (fixed before any analysis; independent of the guard's output-vs-input metrics)
     gt_chroma_excess  chroma p99(output, GT) − chroma p99(input, GT)   (R5-ARCHIVE artefact signal, > 10 = artefact)
     out_face_found    YuNet on the output
     DESTRUCTIVE       gt_chroma_excess > 10  or  no face on the output
     identity          SFace (E01) / dlib (E07) cosine to the GT with GT landmarks, Δ vs B0 (input vs GT)
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
IMG_BASE = "experiments/phase5_rapid/outputs/r5_7"
IMG = IMG_BASE + "/val"
OUT = "data/manifests/phase5_rapid/r5_7"
W: dict = {}


def _init(threads: int, paths: str):
    W["paths"] = paths
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(threads)
    from fes_bench import evaluators as EV
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service.models import nafnet_factory
    W["det"] = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    W["sface"], W["dlib"] = SFace(), EV.DlibResNet()
    W["cfg"] = NR.load_config()
    W["naf"], _ = nafnet_factory()


def lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)


def chroma_p99(a, b) -> float:
    A, B = lab(a), lab(b)
    return float(np.percentile(np.hypot(A[..., 1] - B[..., 1], A[..., 2] - B[..., 2]), 99))


def extra_signals(inp, out) -> dict:
    """Candidate guard signals beyond the sealed ones (computed on output vs input chip only — no GT)."""
    A, B = lab(out), lab(inp)
    d = np.hypot(A[..., 1] - B[..., 1], A[..., 2] - B[..., 2])
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.5)  # noqa: E731
    ha, hb = hp(A), hp(B)
    e_c_out = float((ha[..., 1] ** 2 + ha[..., 2] ** 2).mean())
    e_c_in = float((hb[..., 1] ** 2 + hb[..., 2] ** 2).mean())
    e_l_out, e_l_in = float((ha[..., 0] ** 2).mean()), float((hb[..., 0] ** 2).mean())
    sat_out = np.hypot(A[..., 1] - 128, A[..., 2] - 128)
    sat_in = np.hypot(B[..., 1] - 128, B[..., 2] - 128)
    return {"x_chroma_p50": round(float(np.percentile(d, 50)), 3), "x_chroma_p95": round(float(np.percentile(d, 95)), 3),
            "x_chroma_p999": round(float(np.percentile(d, 99.9)), 3),
            "x_chroma_frac_gt20": round(float((d > 20).mean()), 5), "x_chroma_frac_gt40": round(float((d > 40).mean()), 5),
            "x_hf_chroma_out": round(e_c_out, 3), "x_hf_chroma_in": round(e_c_in, 3),
            "x_hf_luma_out": round(e_l_out, 3), "x_hf_luma_in": round(e_l_in, 3),
            "x_sat_p99_out": round(float(np.percentile(sat_out, 99)), 3),
            "x_sat_p99_in": round(float(np.percentile(sat_in, 99)), 3),
            "x_mean_abs_diff": round(float(np.abs(out.astype(float) - inp.astype(float)).mean()), 3)}


def geometry(img, det):
    """The service's full_image framing (fes_service.pipeline) + the numbers the framing analysis needs."""
    from fes_bench.evaluators import DLIB_CHIP_TEMPLATE_150
    from fes_data import preprocess as P
    from fes_service.pipeline import face_chip
    fr = P.preprocess(img, det, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    g = {"fi_det_status": fr.status, "fi_n_faces": fr.n_faces}
    if fr.status != P.OK or fr.detection is None:
        return None, g
    lm = np.asarray(fr.detection.landmarks, np.float64)
    g["fi_iod_img"] = round(float(np.linalg.norm(lm[1] - lm[0])), 3)
    g["fi_bbox"] = [round(float(v), 1) for v in fr.detection.bbox_xywh]
    if not fr.aligned or not P.landmarks_plausible(lm):
        g["fi_det_status"] = "ALIGNMENT_UNRELIABLE"
        return None, g
    M = P.umeyama_similarity(lm, DLIB_CHIP_TEMPLATE_150)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    cx, cy = np.linalg.inv(M[:, :2]) @ (np.array([75.0, 75.0]) - M[:, 2])
    side = int(max(8, round(150.0 / scale)))
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    h, w = img.shape[:2]
    inside = max(0, min(w, x0 + side) - max(0, x0)) * max(0, min(h, y0 + side) - max(0, y0))
    chip, side2 = face_chip(img, lm)
    g.update(fi_x0=x0, fi_y0=y0, fi_side=side2, fi_chip_px=int(chip.shape[0]),
             fi_pad_frac=round(1 - inside / float(side * side), 4),
             fi_img_cover_frac=round(inside / float(h * w), 4))
    return chip, g


def paste_back(inp, out, g):
    x0, y0, side = g["fi_x0"], g["fi_y0"], g["fi_side"]
    if out.shape[0] != side:
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)
    h, w = inp.shape[:2]
    res = inp.copy()
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    res[ya:yb, xa:xb] = out[ya - y0:yb - y0, xa - x0:xa - x0 + (xb - xa)]
    return res


def feats(chip):
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    f = RW.rw_features(chip, W["det"])
    f.update(NR.texture_features(chip))
    return f


FK = ("iod_px", "det_score", "yaw_proxy", "lapd1_c", "noise_sigma", "blockiness", "luma_mean", "rms_contrast", "is_gray",
      "pixelation", "hf_peak_frac", "hf_energy")


def ident(img, lm):
    return W["sface"].embed(img, lm), W["dlib"].embed(img, lm)


def _job(name):
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    inp = cv2.imread(str(repo_path(f"{ARCH}/input/val/{name}")), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(repo_path(f"{ARCH}/output/val/{name}")), cv2.IMREAD_COLOR)
    r = {"name": name, "h": inp.shape[0], "w": inp.shape[1]}
    faces = W["det"].detect(gt)
    lm = np.asarray(faces[0].landmarks, np.float64) if faces else None
    egt = ident(gt, lm) if lm is not None else None
    if egt is not None:
        e0 = ident(inp, lm)
        r["b0_sface"], r["b0_dlib"] = float(e0[0] @ egt[0]), float(e0[1] @ egt[1])
    cin_gt = chroma_p99(inp, gt)
    paths = {"fc": inp}
    chip, g = geometry(inp, W["det"])
    r.update(g)
    if chip is not None and W["paths"] == "both":
        paths["fi"] = chip
    for p, c in paths.items():
        f = feats(c)
        d = NR.decide(f, W["cfg"])
        r[f"{p}_reason"], r[f"{p}_eligible"] = d.reason, d.eligible
        for k in FK:
            v = f.get(k)
            r[f"{p}_{k}"] = None if isinstance(v, float) and not np.isfinite(v) else v
        if not d.eligible:
            continue
        t0 = time.perf_counter()
        out, _ = W["naf"](c, {})
        r[f"{p}_infer_s"] = round(time.perf_counter() - t0, 3)
        gr = NR.artifact_guard(c, out, W["cfg"], lambda x: len(W["det"].detect(x)) > 0, RW.is_gray(c))
        r[f"{p}_guard_pass"], r[f"{p}_guard_reasons"] = gr.passed, "+".join(gr.reasons)
        for k, v in gr.metrics.items():
            r[f"{p}_g_{k}"] = v
        for k, v in extra_signals(c, out).items():
            r[f"{p}_{k}"] = v
        full = out if p == "fc" else paste_back(inp, out, g)
        r[f"{p}_gt_chroma_excess"] = round(chroma_p99(full, gt) - cin_gt, 3)
        r[f"{p}_out_face"] = bool(W["det"].detect(out))
        r[f"{p}_destructive"] = bool(r[f"{p}_gt_chroma_excess"] > 10 or not r[f"{p}_out_face"])
        if egt is not None:
            eo = ident(full, lm)
            r[f"{p}_d_sface"] = round(float(eo[0] @ egt[0]) - r["b0_sface"], 5)
            r[f"{p}_d_dlib"] = round(float(eo[1] @ egt[1]) - r["b0_dlib"], 5)
        if p == "fc" and r.get(f"{p}_destructive") or (p == "fc" and not gr.passed):
            os.makedirs(repo_path(f"{IMG}/fc_flagged"), exist_ok=True)
            cv2.imwrite(str(repo_path(f"{IMG}/fc_flagged/{name}.png")), np.hstack([c, out, gt]))
        W.setdefault("outs", {})[p] = out
    # NAFNet input scale/context: both paths eligible → compare the two outputs on the chip's in-image region
    if r.get("fc_eligible") and r.get("fi_eligible") and "outs" in W:
        a = W["outs"]["fc"]
        b = paste_back(inp, W["outs"]["fi"], g)
        x0, y0, side = g["fi_x0"], g["fi_y0"], g["fi_side"]
        h, w = inp.shape[:2]
        sl = (slice(max(0, y0), min(h, y0 + side)), slice(max(0, x0), min(w, x0 + side)))
        diff = a[sl].astype(float) - b[sl].astype(float)
        r["fcfi_region_psnr"] = round(float(10 * np.log10(255 ** 2 / max((diff ** 2).mean(), 1e-9))), 2)
    # measurement diagnostic: the face_chip output restricted to the full_image chip region (rest stays the input).
    # If this loses as much identity as the full_image path, the loss is region coverage, not NAFNet's input.
    if r.get("fc_eligible") and "fi_x0" in r and "outs" in W and "fc" in W["outs"] and egt is not None:
        x0, y0, side = g["fi_x0"], g["fi_y0"], g["fi_side"]
        h, w = inp.shape[:2]
        sl = (slice(max(0, y0), min(h, y0 + side)), slice(max(0, x0), min(w, x0 + side)))
        hyb = inp.copy()
        hyb[sl] = W["outs"]["fc"][sl]
        eh = ident(hyb, lm)
        r["fcreg_d_sface"] = round(float(eh[0] @ egt[0]) - r["b0_sface"], 5)
        r["fcreg_d_dlib"] = round(float(eh[1] @ egt[1]) - r["b0_dlib"], 5)
    W.pop("outs", None)
    return r


def collect(workers: int, limit: int, threads: int, paths: str, sample: int, run: str) -> None:
    global IMG
    IMG = f"{IMG_BASE}/{run}"
    base = repo_path(IMG)
    base.mkdir(parents=True, exist_ok=True)
    rec = base / "records.jsonl"
    done = set()
    if rec.exists():
        for line in rec.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["name"])
            except json.JSONDecodeError:
                pass
    names = sorted(os.listdir(repo_path(f"{ARCH}/input/val")))
    if sample:
        import random
        names = sorted(random.Random(0).sample(names, sample))
    names = names[: limit or None]
    todo = [n for n in names if n not in done]
    print(f"val: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads, paths)) as ex, open(rec, "a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(_job, todo, chunksize=4)):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if i % 500 == 0:
                print(f"val {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    finalize(run)


def finalize(run: str) -> None:
    global IMG
    IMG = f"{IMG_BASE}/{run}"
    rows = []
    for line in repo_path(f"{IMG}/records.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    last = {r["name"]: r for r in rows}
    rows = [last[k] for k in sorted(last)]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv_gz(repo_path(f"{OUT}/{run}.csv.gz"), [{k: r.get(k, "") for k in cols} for r in rows], cols)
    print(f"{len(rows)} rows → {OUT}/{run}.csv.gz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "finalize"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--paths", choices=("fc", "both"), default="both")
    ap.add_argument("--sample", type=int, default=0, help="random val sample (seed 0); 0 = whole split")
    ap.add_argument("--run", default="val")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.workers, a.limit, a.threads, a.paths, a.sample, a.run)
    else:
        finalize(a.run)
