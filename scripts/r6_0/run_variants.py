"""R6-0 — re-create every corpus case's NAFNet output with the build that produced it, plus the same NAFNet run on the
flipped input (h / v / hv, flipped back). Research only: reuses the service's own functions, changes nothing.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r6_0/run_variants.py [--workers 3] [--threads 4]
    → experiments/phase5_rapid/outputs/r6_0/cache/<case>/{chip,work,v0,vh,vv,vhv}.png (+ mask.png) and meta.jsonl

Determinism check: v0 must equal the R5 output saved by the original run (max |Δ| recorded per case).
"""
from __future__ import annotations

import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path

C = "data/manifests/phase5_rapid/r6_0/cases.json"
CACHE = "experiments/phase5_rapid/outputs/r6_0/cache"
ARCH = "Dataset-and-code/archive"
SV = "experiments/phase5_rapid/outputs/r5_6"
R10 = "experiments/phase5_rapid/outputs/r5_10"
FLIPS = {"v0": None, "vh": 1, "vv": 0, "vhv": -1}
W: dict = {}


def cid_dir(cid):
    return cid.replace(":", "__").replace("/", "_")


def _init(threads):
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(threads)
    from fes_data import preprocess as P
    from fes_service.models import nafnet_factory, reflect_pad16
    W["det"] = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    live, _ = nafnet_factory()

    def cand(img, ctx):
        x, h, w = reflect_pad16(img)
        return np.ascontiguousarray(live(x, ctx)[0][:h, :w]), {}
    W["fn"] = {"live": live, "cand": cand}
    p4 = read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))
    p4 += read_csv_gz(repo_path("data/manifests/phase5_rapid/gap/population_r5gap.csv.gz"))
    W["p4"] = {r["pair_id"]: r["degraded_path"] for r in p4}


def paths(c):
    n, m, s = c["name"], c["mode"], c["src"]
    fm = {"fc": "face_chip", "fi": "full_image"}[m]
    if s == "dev":
        return f"{ARCH}/input/val/{n}", f"experiments/phase5_rapid/outputs/r5_8/corpus/{m}/{n}.png"
    if s == "dev_mod":
        return f"{ARCH}/input/val/{n}", f"{R10}/archive/{m}/{n}.png"
    if s == "clean":
        return f"{ARCH}/output/val/{n}", f"{R10}/clean/fc/{n}.png"
    if s == "p4":
        return W["p4"][n], f"{R10}/p4/fc/{n}.png"
    if s.startswith("test_"):
        run = {"test_r56": "", "test_r57": "r57_", "test_r58": "r58_", "test_r510": "r510_"}[s] + f"test_{fm}"
        return f"{ARCH}/input/test/{n}", f"{SV}/{run}/out/test__{n}.png"
    if s.startswith("real_"):
        run = {"real_r56": "real_full_image", "real_r510": "r510_real_full_image"}[s]
        return f"Dataset-and-code/dataset/{n}", f"{SV}/{run}/out/real__{n}.png"
    raise KeyError(s)


def chips(c, img):
    """(chip, work, mask) exactly as pipeline.process builds them for this case's build/profile."""
    from fes_data import preprocess as P
    from fes_service.pipeline import face_chip, face_chip_filled
    if c["mode"] == "fc":
        return img, img, None
    fr = P.preprocess(img, W["det"], P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    lm = np.asarray(fr.detection.landmarks, np.float64)
    chip, _ = face_chip(img, lm)
    if c["profile"] == "cand":
        filled, mask = face_chip_filled(img, lm)
        if mask.any():
            return chip, filled, mask
    return chip, chip, None


def _job(c):
    d = repo_path(f"{CACHE}/{cid_dir(c['id'])}")
    r = {"id": c["id"]}
    try:
        ip, op = paths(c)
        img = cv2.imread(str(repo_path(ip)), cv2.IMREAD_COLOR)
        chip, work, mask = chips(c, img)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(str(d / "chip.png"), chip)
        if mask is not None:
            cv2.imwrite(str(d / "work.png"), work)
            cv2.imwrite(str(d / "mask.png"), mask.astype(np.uint8) * 255)
        fn = W["fn"][c["profile"]]
        t0 = time.perf_counter()
        for k, fl in FLIPS.items():
            x = work if fl is None else cv2.flip(work, fl)
            y = fn(np.ascontiguousarray(x), {})[0]
            y = y if fl is None else cv2.flip(y, fl)
            if mask is not None:
                y = y.copy()
                y[mask] = 0
            cv2.imwrite(str(d / f"{k}.png"), y)
            if k == "v0":
                saved = cv2.imread(str(repo_path(op)), cv2.IMREAD_COLOR)
                r["saved_found"] = saved is not None
                if saved is not None:
                    r["saved_maxdiff"] = int(np.abs(saved.astype(int) - y.astype(int)).max()) if saved.shape == y.shape else -1
        r["infer4_s"] = round(time.perf_counter() - t0, 2)
        r["chip_px"], r["oob"] = int(chip.shape[0]), float(mask.mean()) if mask is not None else 0.0
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"[:300]
    return r


def main(workers, threads):
    cases = json.loads(repo_path(C).read_text())
    os.makedirs(repo_path(CACHE), exist_ok=True)
    meta = repo_path(f"{CACHE}/meta.jsonl")
    done = set()
    if meta.exists():
        done = {json.loads(l)["id"] for l in meta.read_text().splitlines() if l.strip()}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads,)) as ex, open(meta, "a") as fh:
        for i, r in enumerate(ex.map(_job, todo, chunksize=2)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s {r}", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    main(a.workers, a.threads)
