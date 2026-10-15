"""R6-ONNX: PyTorch vs ONNX Runtime on the same blurred faces, with the service's own single-output steps.

  FES_IMAGE=fes-dev:cpu-p5-onnx scripts/dock.sh python scripts/r6_onnx/bench.py [--sets val300,synth160,archive100]

Per face and per backend (torch | onnx), for both models (r6 = NAFNet-GoPro, r6_ft = face fine-tune):
  single   one reflect-pad16 forward pass                                   (models.reflect_pad16)
  r6       the 4-run mirror consensus + stability check                     (models.r6_consensus, NR.stability_check)
  best     the sharper stability-accepted output of the two models         (NR.select_best) — the 1-to-1 page's rule
Times are wall-clock per face on this CPU (2 threads, both backends). Each backend lives in its own worker process
(like the service's isolated model workers) so their thread pools never compete; one face is sent to one worker at a
time and the backend order alternates per face.
Compared: torch vs onnx output pixels, stability decisions, chosen model, PSNR / SFace to the sharp original.
NOT run here (they do not depend on the backend's arithmetic beyond the pixels compared): face detection, the blur
gate, and the artifact guards after the stability check.

Sets (all blurred faces with a sharp original; none was used for training):
  val300      the 300 fixed FairFace VAL pairs (128 px crops; gaussian / defocus / motion / already sharp)
  synth160    experiments/r6_ft/selector/synth (whole faces, 128–224 px)
  archive100  Dataset-and-code/archive TEST, every 192nd pair (heavy shake blur; licence unknown, evaluation only)
Output: experiments/r6_onnx/bench/{rows.json, summary.json, diff/*.png}
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "r6_ft"))

from fes_service import models as M  # noqa: E402
from fes_service import nafnet_route as NR  # noqa: E402

THREADS = 2
ONNX_DIR = ROOT / "models" / "M28_nafnet" / "onnx"
OUT = ROOT / "experiments" / "r6_onnx" / "bench"
MODELS = {"r6": ("nafnet_gopro", {}),
          "r6_ft": ("nafnet_face_ft", {"weights": M.FACE_FT_WEIGHTS, "sha256": M.FACE_FT_SHA256})}
BACKENDS = ("torch", "onnx")
FAIRFACE = "datasets/raw/DS01_fairface"
ARCHIVE = ROOT / "Dataset-and-code" / "archive"
SYNTH = ROOT / "experiments" / "r6_ft" / "selector" / "synth"


# ------------------------------------------------------------------ backends: enhance(img BGR u8, ctx) → (BGR u8, meta)

def torch_backend(profile):
    import torch
    torch.set_num_threads(THREADS)
    enhance, info = M.nafnet_factory(**MODELS[profile][1])
    return enhance, info["load_s"]


def onnx_backend(profile):
    import onnxruntime as ort
    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
    sess = ort.InferenceSession(str(ONNX_DIR / f"{MODELS[profile][0]}.onnx"), so, providers=["CPUExecutionProvider"])
    max_side = int(sess.get_modelmeta().custom_metadata_map["fes_max_side"])

    def enhance(img, ctx):
        assert max(img.shape[:2]) <= max_side, img.shape
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        y = sess.run(None, {"image": np.ascontiguousarray(x.transpose(2, 0, 1))[None]})[0][0]
        rgb = (np.clip(y, 0, 1).transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), {}
    return enhance, round(time.perf_counter() - t0, 2)


def pad16(enhance0):
    def enhance(img, ctx):
        x, h, w = M.reflect_pad16(img)
        return np.ascontiguousarray(enhance0(x, ctx)[0][:h, :w]), {}
    return enhance


def load(backend):
    """→ {profile: (single, consensus)}, {profile: load seconds}"""
    fns, secs = {}, {}
    for p in MODELS:
        e0, secs[p] = (torch_backend if backend == "torch" else onnx_backend)(p)
        e1 = pad16(e0)
        fns[p] = (e1, M.r6_consensus(e1))
    return fns, secs


# ------------------------------------------------------------------ data: (set, name, kind, blurred BGR, sharp BGR)

def to_chip(img):
    """The service's face_chip rule: sides above 224 are reduced with INTER_AREA."""
    if max(img.shape[:2]) > 224:
        s = 224 / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def items(sets):
    if "val300" in sets:
        import zipfile

        from train_nafnet_face import ZIP, FaceBlur
        zp = str(ROOT / FAIRFACE / ZIP)
        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg") and n.startswith("val/"))
        val = names[::max(1, len(names) // 300)][:300]
        ds = FaceBlur(zp, val, 7, fixed=True)
        u8 = lambda t: (t.clamp(0, 1).permute(1, 2, 0).numpy()[..., ::-1] * 255 + 0.5).astype(np.uint8)  # noqa: E731
        for i in range(len(ds)):
            d, s, kind = ds[i][:3]
            yield "val300", f"{i:03d}", kind, np.ascontiguousarray(u8(d)), np.ascontiguousarray(u8(s))
    if "synth160" in sets:
        for n in sorted(os.listdir(SYNTH / "input")):
            yield ("synth160", n[:3], n[4:-4], cv2.imread(str(SYNTH / "input" / n)), cv2.imread(str(SYNTH / "sharp" / n)))
    if "archive100" in sets:
        names = sorted(os.listdir(ARCHIVE / "input" / "test"))
        for i, n in enumerate(names[::max(1, len(names) // 100)][:100]):
            b, s = cv2.imread(str(ARCHIVE / "input" / "test" / n)), cv2.imread(str(ARCHIVE / "output" / "test" / n))
            if b is None or s is None or b.shape != s.shape:
                continue
            yield "archive100", f"{i:03d}", "shake", to_chip(b), to_chip(s)      # index only: no source file name


# ------------------------------------------------------------------ measures

def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))


def pix_diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max": int(d.max()), "mean": round(float(d.mean()), 4), "changed": round(float((d > 0).mean()), 4),
            "gt2": round(float((d > 2).mean()), 5), "psnr": round(psnr(a, b), 2)}


def run_backend(fns, cfg, blur):
    """→ per-backend record + the images needed for the comparison."""
    rec, imgs, accepted = {}, {}, {}
    for p, (single, cons) in fns.items():
        t0 = time.perf_counter()
        y1 = single(blur, {})[0]
        t1 = time.perf_counter()
        y4, meta = cons(blur, {})
        t2 = time.perf_counter()
        sg = NR.stability_check(meta["runs_dot_c"], cfg)
        rec[p] = {"single_ms": round((t1 - t0) * 1e3, 1), "r6_ms": round((t2 - t1) * 1e3, 1),
                  "runs_dot_c": meta["runs_dot_c"], "stable": bool(sg.passed)}
        imgs[f"{p}_single"], imgs[f"{p}_r6"] = y1, y4
        if sg.passed:
            accepted[p] = y4
    t0 = time.perf_counter()
    chosen, scores = NR.select_best(blur, accepted) if accepted else (None, {})
    rec["select_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
    rec["chosen"], rec["scores"] = chosen, scores
    rec["best_ms"] = round(sum(rec[p]["r6_ms"] for p in fns) + rec["select_ms"], 1)
    imgs["best"] = accepted[chosen] if chosen else blur
    return rec, imgs


def worker(backend, conn):
    fns, _ = load(backend)
    cfg = NR.load_config(NR.R6_CONFIG)
    warm = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for p in MODELS:
        fns[p][0](warm, {})
    conn.send("ready")
    while True:
        blur = conn.recv()
        if blur is None:
            return
        conn.send(run_backend(fns, cfg, blur))


def probe(backend):
    """Child process: load both models on one backend, run 3 full 224 px single-output passes → load time and peak RSS."""
    fns, secs = load(backend)
    cfg = NR.load_config(NR.R6_CONFIG)
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    x = cv2.resize(cv2.imread(str(SYNTH / "input" / sorted(os.listdir(SYNTH / "input"))[0])), (224, 224))
    for _ in range(3):
        run_backend(fns, cfg, x)
    print(json.dumps({"backend": backend, "load_s": secs, "rss_loaded_mb": round(rss_loaded / 1024),
                      "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}))


def main(a):
    if a.probe:
        return probe(a.probe)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diff").mkdir(exist_ok=True)
    probes = [json.loads(subprocess.run([sys.executable, __file__, "--probe", b], capture_output=True, text=True,
                                        check=True).stdout.strip().splitlines()[-1]) for b in BACKENDS]
    print("probes", probes, flush=True)

    from fes_bench.metrics import YUNET, SFace
    from fes_data.common import repo_path
    from fes_data.preprocess import YuNetDetector
    det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)
    sf = SFace()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    conns = {}
    for b in BACKENDS:
        conns[b], child = ctx.Pipe()
        ctx.Process(target=worker, args=(b, child), daemon=True).start()
    for b in BACKENDS:
        assert conns[b].recv() == "ready"

    rows = json.loads((OUT / "rows.json").read_text())["rows"] if a.resume else []
    done = {(r["set"], r["id"]) for r in rows}
    for k, (set_, name, kind, blur, sharp) in enumerate(items(set(a.sets.split(",")))):
        if (set_, name) in done:
            continue
        if a.limit and sum(r["set"] == set_ for r in rows) >= a.limit:
            continue
        row = {"set": set_, "id": name, "kind": kind, "h": blur.shape[0], "w": blur.shape[1],
               "psnr_input": round(psnr(blur, sharp), 3)}
        imgs = {}
        for b in (BACKENDS if k % 2 == 0 else BACKENDS[::-1]):
            conns[b].send(blur)
            row[b], imgs[b] = conns[b].recv()
        row["diff"] = {key: pix_diff(imgs["torch"][key], imgs["onnx"][key]) for key in imgs["torch"]}
        row["same_chosen"] = row["torch"]["chosen"] == row["onnx"]["chosen"]
        row["same_stable"] = all(row["torch"][p]["stable"] == row["onnx"][p]["stable"] for p in MODELS)
        faces = det.detect(cv2.resize(sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        lm = np.asarray(faces[0].landmarks, np.float32) / 2.0 if faces else None
        if lm is not None:
            e = sf.embed(sharp, lm)
            row["sface_input"] = round(float(e @ sf.embed(blur, lm)), 4)
        for b in BACKENDS:
            for key, img in imgs[b].items():
                row[b][f"psnr_{key}"] = round(psnr(img, sharp), 3)
                if lm is not None and key == "best":
                    row[b]["sface_best"] = round(float(e @ sf.embed(img, lm)), 4)
        if row["diff"]["best"]["max"] > 2 or not row["same_chosen"]:
            d = np.clip(np.abs(imgs["torch"]["best"].astype(np.int16) - imgs["onnx"]["best"]) * 10, 0, 255).astype(np.uint8)
            tiles = [cv2.resize(t, (336, 336), interpolation=cv2.INTER_CUBIC) for t in (blur, imgs["torch"]["best"], imgs["onnx"]["best"], d)]
            for t, lab in zip(tiles, ("INPUT", f"TORCH ({row['torch']['chosen']})", f"ONNX ({row['onnx']['chosen']})", "DIFF x10")):
                cv2.putText(t, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(OUT / "diff" / f"{set_}_{name}.png"), np.hstack(tiles))
        rows.append(row)
        if len(rows) % 20 == 0:
            (OUT / "rows.json").write_text(json.dumps({"probes": probes, "rows": rows}))
            print(f"{len(rows)} faces · last {set_}/{name} torch {row['torch']['best_ms']:.0f} ms onnx {row['onnx']['best_ms']:.0f} ms "
                  f"diff max {row['diff']['best']['max']}", flush=True)
    for c in conns.values():
        c.send(None)
    (OUT / "rows.json").write_text(json.dumps({"probes": probes, "threads": THREADS, "rows": rows}))
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="val300,synth160,archive100")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", default="")
    ap.add_argument("--resume", action="store_true", help="keep the faces already in rows.json, run the rest")
    main(ap.parse_args())
