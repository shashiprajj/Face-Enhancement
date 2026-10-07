"""R5-6 — service-level validation of the R5-5 build through the REAL FastAPI service (evaluation only).

Every image is POSTed to running `uvicorn fes_service.app:app` containers (default configuration: FES_GATE=rw,
FES_ROUTE_MODE=auto; `route_mode` is NOT sent, so the service default applies). Nothing in src/ or configs/ is used to
decide anything here — the service decides; this script only records, checks and scores.

    run       --dataset test|ztest|real --mode face_chip|full_image --name <run> --services http://h1:8080,...
              → experiments/phase5_rapid/outputs/r5_6/<run>/{records.jsonl, out/, chip/}   (resumable)
              → data/manifests/phase5_rapid/r5_6/<run>.csv.gz
    identity  --name <run>     SFace (E01) + dlib (E07) of every ENHANCED output vs the sharp GT, exactly as
              R5-ARCHIVE stage 2: GT landmarks (stage-1 gt_landmarks.npy), cosine to the GT embedding, compared with
              the stage-1 B0 (input vs GT) cosine. full_image outputs are face chips in FairFace framing: they are pasted
              back into the 128-px input frame at the chip's position (recomputed with the service's own detector and
              chip geometry, verified against the chip the service returned) so the GT landmarks apply.
    guardcheck --name <run>    controlled comparison, only for inputs whose NAFNet output the guard REJECTED: re-runs
              NAFNet in-process (same factory, same chip) so the blocked output can be inspected. Recorded explicitly.
    analyze   → data/manifests/phase5_rapid/r5_6/summary.json + tables.md
    sheets    visual-inspection sheets (git-ignored outputs)

Datasets (identical to R5-ARCHIVE): test = all 19,290 pairs of Dataset-and-code/archive test split (sorted names);
ztest = all 100; real = the 200 CC0 photos in Dataset-and-code/dataset. The R5-ARCHIVE stage-2 sample (300 test, seed 0)
is reused for matched comparisons with the always-on NAFNet (N1) and A4 runs.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, write_csv_gz, write_json

ARCH = "Dataset-and-code/archive"
REAL = "Dataset-and-code/dataset"
OUT = "data/manifests/phase5_rapid/r5_6"
IMG = "experiments/phase5_rapid/outputs/r5_6"
A_OUT = "data/manifests/phase5_rapid/archive"
A_IMG = "experiments/phase5_rapid/outputs/archive"
NAF = "NAFNET_GOPRO"
W: dict = {}
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())          # one id per client process (latency bookkeeping)


def items_of(dataset: str) -> list[tuple[str, str]]:
    """(input_id, path)."""
    if dataset in ("test", "ztest"):
        d = repo_path(f"{ARCH}/input/{dataset}")
        return [(f"{dataset}/{n}", str(d / n)) for n in sorted(os.listdir(d))]
    d = repo_path(REAL)
    return [(f"real/{p.name}", str(p)) for p in sorted(d.glob("*.jpg"))]


# ================================================================== run (HTTP client)

def flatten(j: dict) -> dict:
    nf = j.get("nafnet") or {}
    art = nf.get("artifact") or {}
    r = {k: j.get(k) for k in ("service_version", "status", "route_mode", "input_mode", "decision", "reason", "bucket",
                               "route", "used", "model", "fallback", "error", "escalate", "n_faces", "chip_px",
                               "artifact_check", "forced", "gate_route", "output", "image_format")}
    r["warnings"] = "+".join(j.get("warnings") or [])
    r["nafnet_eligible"] = nf.get("eligible")
    r["nafnet_reason"] = nf.get("reason")
    r["nafnet_infer_s"] = nf.get("infer_s")
    r["artifact_reasons"] = "+".join(art.get("reasons") or [])
    for k, v in (art.get("metrics") or {}).items():
        r[f"art_{k}"] = v
    for k, v in (j.get("quality") or {}).items():
        r[f"q_{k}"] = v
    for k, v in (j.get("timings_ms") or {}).items():
        r[f"ms_{k}"] = v
    return r


def run(dataset: str, mode: str, name: str, services: list[str], inflight: int, limit: int) -> None:
    import httpx
    base = repo_path(f"{IMG}/{name}")
    (base / "out").mkdir(parents=True, exist_ok=True)
    (base / "chip").mkdir(parents=True, exist_ok=True)
    rec_path = base / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") not in ("BUSY", None) and r.get("http") != 0:   # BUSY / failed requests are retried
                done.add(r["input_id"])
    todo = [it for it in items_of(dataset)[: limit or None] if it[0] not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to go, services={services}", flush=True)
    lock = threading.Lock()
    t0, n = time.time(), [0]
    # each client thread is pinned to ONE service, so a service never has more than `inflight` requests in flight
    # (beyond its max_inflight=4 the service answers 503 BUSY — the first round-robin client of this run hit that)
    local = threading.local()
    slot = iter(range(10 ** 9))

    def one(k_item):
        if not hasattr(local, "svc"):
            with lock:
                local.svc = next(slot) % len(services)
            local.client = httpx.Client(base_url=services[local.svc], timeout=900.0)
        svc = local.svc
        k, (iid, path) = k_item
        data = open(path, "rb").read()
        r = {"input_id": iid, "dataset": dataset, "req_input_mode": mode, "service": services[svc], "client_run": RUN_ID}
        t = time.perf_counter()
        try:
            resp = local.client.post("/v1/enhance", files={"image": (os.path.basename(path), data, "application/octet-stream")},
                                     data={"input_mode": mode, "return_image": "true"})
            r["http"] = resp.status_code
            j = resp.json()
        except Exception as e:  # noqa: BLE001 — a failed request is a finding, not a crash of the evaluation
            r.update(http=0, client_error=f"{type(e).__name__}: {e}"[:200], client_s=round(time.perf_counter() - t, 3))
            j = None
        r["client_s"] = round(time.perf_counter() - t, 3)
        if j is not None:
            r.update(flatten(j))
            r.update(validity(j, data, path, base, iid))
        with lock:
            with open(rec_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            n[0] += 1
            if n[0] % 250 == 0 or n[0] == len(todo):
                el = time.time() - t0
                print(f"{name} {n[0]}/{len(todo)} {el:.0f}s ({n[0] / el:.1f}/s)", flush=True)

    with ThreadPoolExecutor(inflight * len(services)) as ex:
        list(ex.map(one, enumerate(todo)))
    finalize(name)


def validity(j: dict, data: bytes, path: str, base, iid: str) -> dict:
    """Output validity as a client sees it + save ENHANCED outputs (and their chips) for scoring/inspection."""
    st = j.get("status")
    fn = iid.replace("/", "__")
    if st == "ORIGINAL":
        ok = j.get("image_b64") is not None and base64.b64decode(j["image_b64"]) == data
        return {"output_valid": ok, "output_check": "original_bytes_identical" if ok else "ORIGINAL_BYTES_DIFFER"}
    if st == "REJECTED":
        return {"output_valid": j.get("image_b64") is None, "output_check": "rejected_no_image"}
    if st != "ENHANCED" or not j.get("image_b64"):
        return {"output_valid": False, "output_check": f"unexpected status {st}"}
    raw = base64.b64decode(j["image_b64"])
    out = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    chip = cv2.imdecode(np.frombuffer(base64.b64decode(j["chip_b64"]), np.uint8), cv2.IMREAD_COLOR) \
        if j.get("chip_b64") else None
    problems = []
    if out is None:
        problems.append("undecodable")
    else:
        exp = (224, 224, 3) if j.get("used") == "A4" else (chip.shape if chip is not None else None)
        if exp is not None and out.shape != exp:
            problems.append(f"shape {out.shape} != {exp}")
        if float(out.std()) < 1.0:
            problems.append("near-constant")
    (base / "out" / f"{fn}.png").write_bytes(raw)
    if chip is not None:
        cv2.imwrite(str(base / "chip" / f"{fn}.png"), chip)
    return {"output_valid": not problems, "output_check": "+".join(problems) or "enhanced_ok"}


def records(name: str) -> list[dict]:
    p = repo_path(f"{IMG}/{name}/records.jsonl")
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    last = {r["input_id"]: r for r in rows}                       # resume-safe: one row per input
    return [last[k] for k in sorted(last)]


def finalize(name: str) -> None:
    rows = records(name)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv_gz(repo_path(f"{OUT}/{name}.csv.gz"), [{k: r.get(k, "") for k in cols} for r in rows], cols)
    print(f"{name}: {len(rows)} records → {OUT}/{name}.csv.gz")


# ================================================================== identity (R5-ARCHIVE methodology)

def _init_id():
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    from fes_bench import evaluators as EV
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    W["det"] = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    W["sface"], W["dlib"] = SFace(), EV.DlibResNet()
    W["lms"] = np.load(repo_path(f"{A_IMG}/gt_landmarks.npy"), allow_pickle=True).item()


def chip_geometry(img: np.ndarray) -> tuple[int, int, int] | None:
    """(x0, y0, side) of the service's face chip — the same detector call and the same arithmetic as
    fes_service.pipeline.face_chip (re-derived here, read-only, to map a full_image output back to the input frame)."""
    from fes_bench.evaluators import DLIB_CHIP_TEMPLATE_150
    from fes_data import preprocess as P
    fr = P.preprocess(img, W["det"], P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    if fr.status != P.OK or fr.detection is None:
        return None
    lm = np.asarray(fr.detection.landmarks, np.float64)
    M = P.umeyama_similarity(lm, DLIB_CHIP_TEMPLATE_150)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    cx, cy = np.linalg.inv(M[:, :2]) @ (np.array([75.0, 75.0]) - M[:, 2])
    side = int(max(8, round(150.0 / scale)))
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def paste_back(inp: np.ndarray, out: np.ndarray, geo) -> np.ndarray:
    x0, y0, side = geo
    if out.shape[0] != side:                                    # chip was downscaled to 224 by the service
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)
    h, w = inp.shape[:2]
    res = inp.copy()
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    res[ya:yb, xa:xb] = out[ya - y0:yb - y0, xa - x0:xa - x0 + (xb - xa)]
    return res


def chroma_p99(img, ref) -> float:
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.percentile(np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]), 99))


def _id_job(args):
    from fes_bench import metrics as M
    name, rec = args
    split, fname = rec["input_id"].split("/", 1)
    inp = cv2.imread(str(repo_path(f"{ARCH}/input/{split}/{fname}")), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(repo_path(f"{ARCH}/output/{split}/{fname}")), cv2.IMREAD_COLOR)
    fn = rec["input_id"].replace("/", "__")
    out = cv2.imread(str(repo_path(f"{IMG}/{name}/out/{fn}.png")), cv2.IMREAD_COLOR)
    r = {"input_id": rec["input_id"], "used": rec["used"]}
    if rec["req_input_mode"] == "full_image":
        geo = chip_geometry(inp)
        chip = cv2.imread(str(repo_path(f"{IMG}/{name}/chip/{fn}.png")), cv2.IMREAD_COLOR)
        if geo is None:
            return {**r, "id_status": "NO_GEOMETRY"}
        # verify the recomputed geometry reproduces the chip the service returned (in-image part)
        x0, y0, side = geo
        pad = max(0, -x0, -y0, x0 + side - inp.shape[1], y0 + side - inp.shape[0])
        src = cv2.copyMakeBorder(inp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mine = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        r["geometry_verified"] = bool(chip is not None and mine.shape == chip.shape and np.array_equal(mine, chip))
        r["chip_pad_px"] = pad
        out = paste_back(inp, out, geo)
    elif out.shape != gt.shape:
        out = cv2.resize(out, gt.shape[1::-1], interpolation=cv2.INTER_AREA)
    lm_l = W["lms"].get(rec["input_id"])
    lm = np.asarray(lm_l, np.float64) if lm_l is not None else None
    if lm is None:
        return {**r, "id_status": "NO_GT_LANDMARKS"}
    eg_s, eg_d = W["sface"].embed(gt, lm), W["dlib"].embed(gt, lm)
    eo_s, eo_d = W["sface"].embed(out, lm), W["dlib"].embed(out, lm)
    r.update(id_status="OK", id_sface=float(np.dot(eo_s, eg_s)), id_dlib=float(np.dot(eo_d, eg_d)),
             psnr=round(M.psnr_rgb(out, gt), 3), ssim=round(M.ssim_y(out, gt), 4),
             chroma_excess_vs_input=round(chroma_p99(out, gt) - chroma_p99(inp, gt), 2),
             out_face_found=bool(W["det"].detect(out)))
    emb = (eo_s, eo_d) if split == "ztest" else None
    return r, emb


def identity(name: str, workers: int) -> None:
    rows = [r for r in records(name) if r.get("status") == "ENHANCED"]
    res, embs = [], {}
    with ProcessPoolExecutor(workers, initializer=_init_id) as ex:
        for x in ex.map(_id_job, [(name, r) for r in rows], chunksize=8):
            if isinstance(x, tuple):
                r, e = x
                if e is not None:
                    embs[r["input_id"].split("/", 1)[1]] = {"out": e}
            else:
                r = x
            res.append(r)
    if res:
        cols = list(dict.fromkeys(k for r in res for k in r))
        write_csv_gz(repo_path(f"{OUT}/{name}_identity.csv.gz"), [{k: r.get(k, "") for k in cols} for r in res], cols)
    np.save(repo_path(f"{IMG}/{name}/ztest_emb.npy"), embs, allow_pickle=True)
    print(f"{name}: identity for {len(res)} enhanced outputs")


# ================================================================== guardcheck (controlled, guard-rejected only)

def guardcheck(name: str) -> None:
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    from fes_service.models import nafnet_factory
    from fes_service.pipeline import face_chip
    fn_naf, info = nafnet_factory()
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    cfg = NR.load_config()
    rows = [r for r in records(name) if r.get("fallback") == "ARTIFACT_GUARD"]
    paths = dict(items_of(rows[0]["dataset"])) if rows else {}
    out_dir = repo_path(f"{IMG}/{name}/guard_blocked")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for r in rows:
        img = cv2.imread(paths[r["input_id"]], cv2.IMREAD_COLOR)
        if r["req_input_mode"] == "full_image":
            fr = P.preprocess(img, det, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
            chip, _ = face_chip(img, np.asarray(fr.detection.landmarks, np.float64))
        else:
            chip = img
        out, _ = fn_naf(chip, {})
        g = NR.artifact_guard(chip, out, cfg, lambda x: len(det.detect(x)) > 0, RW.is_gray(chip))
        fn = r["input_id"].replace("/", "__")
        cv2.imwrite(str(out_dir / f"{fn}.png"), np.hstack([chip, out]))
        res.append({"input_id": r["input_id"], "service_reasons": r.get("artifact_reasons"),
                    "rerun_reasons": "+".join(g.reasons), "reproduced": "+".join(g.reasons) == r.get("artifact_reasons"),
                    **{f"rerun_{k}": v for k, v in g.metrics.items()}})
    if res:
        cols = list(dict.fromkeys(k for r in res for k in r))
        write_csv_gz(repo_path(f"{OUT}/{name}_guardcheck.csv.gz"), [{k: r.get(k, "") for k in cols} for r in res], cols)
    print(f"{name}: guardcheck {len(res)} (reproduced {sum(x['reproduced'] for x in res)})", info["weights_sha256"][:8])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "identity", "guardcheck", "finalize"))
    ap.add_argument("--dataset", choices=("test", "ztest", "real"))
    ap.add_argument("--mode", choices=("face_chip", "full_image"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--services", default="http://fes-r56-s1:8080")
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.dataset, a.mode, a.name, a.services.split(","), a.inflight, a.limit)
    elif a.cmd == "identity":
        identity(a.name, a.workers)
    elif a.cmd == "guardcheck":
        guardcheck(a.name)
    else:
        finalize(a.name)
