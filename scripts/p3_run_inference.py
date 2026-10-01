"""P3-T3/P3-T4 — run one configuration over one frozen population (identical inputs for every configuration).

    scripts/dock.sh python scripts/p3_run_inference.py --config-id A1 --population trial [--workers 3 --threads 4]

Per pair: verify the degraded file's SHA-256 → adapter.enhance (timed) → PNG under
experiments/phase3/outputs/<population>/<config>/<cell>/<ref_id>.png (git-ignored). Failures are recorded, never
skipped silently and never replaced by another model's output. Resumable: a pair whose output PNG and partial
record both exist is not recomputed.

Writes (tracked): data/manifests/phase3/runs/<population>/<config>.csv.gz (+ _run.json with environment, model
load reports, failures and timing summary).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from fes_bench import read_bytes_retry
from fes_data.common import (load_config, read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_csv_gz,
                             write_json)

FIELDS = ["pair_id", "ref_id", "cell", "profile", "severity", "in_size", "config_id", "status", "error",
          "output_path", "output_sha256", "output_pixel_sha256", "infer_s", "meta_json", "worker_pid"]
_ADAPTER = None


def _init(config_id: str, threads: int) -> None:
    global _ADAPTER
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    from fes_bench.adapters import build
    _ADAPTER = build(config_id)
    # one worker loads at a time: concurrent checkpoint loads spike memory past what the shared Docker VM has
    # left (workers were OOM-killed at init, P3-T2 F5); steady-state inference memory is not the problem
    import fcntl
    import gc
    import tempfile
    with open(Path(tempfile.gettempdir()) / "fes_load.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            _ADAPTER.load()
            gc.collect()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _one(task: dict) -> dict:
    p, out_path = task["pair"], task["out_path"]
    rec = {"pair_id": p["pair_id"], "ref_id": p["ref_id"], "cell": p["cell"], "profile": p["profile"],
           "severity": p["severity"], "in_size": int(p["out_w"]), "config_id": _ADAPTER.config_id,
           "status": "OK", "error": "", "output_path": out_path, "output_sha256": "", "output_pixel_sha256": "",
           "infer_s": "", "meta_json": "{}", "worker_pid": os.getpid()}
    try:
        data = read_bytes_retry(repo_path(p["degraded_path"]))
        if sha256_bytes(data) != p["degraded_sha256"]:
            raise RuntimeError("degraded input SHA-256 mismatch")
        deg = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        ctx = {"s": deg.shape[0], "M_e2f": np.array(json.loads(p["ffhq512_transform_2x3"]), np.float64)}
        t0 = time.perf_counter()
        out, meta = _ADAPTER.enhance(deg, ctx)
        rec["infer_s"] = round(time.perf_counter() - t0, 4)
        if out.shape != (224, 224, 3) or out.dtype != np.uint8:
            raise RuntimeError(f"bad output shape/dtype {out.shape} {out.dtype}")
        ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        dst = repo_path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(buf.tobytes())
        rec.update(output_sha256=sha256_bytes(buf.tobytes()), output_pixel_sha256=sha256_bytes(out.tobytes()),
                   meta_json=json.dumps(meta, sort_keys=True))
    except Exception as e:  # recorded, not hidden
        rec.update(status="FAILED", error=f"{type(e).__name__}: {e}"[:500],
                   meta_json=json.dumps({"traceback": traceback.format_exc()[-1500:]}))
    return rec


def _reports() -> tuple[dict, dict]:
    return _ADAPTER.load_reports, _ADAPTER.describe()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--population", required=True, choices=["trial", "main"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N pairs (never used for results)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    workers = args.workers or cfg["runtime"]["workers"]
    threads = args.threads or cfg["runtime"]["torch_threads_per_worker"]
    assert args.config_id in cfg["configurations"], "unknown configuration"

    pop_path = f"data/manifests/phase3/population_{args.population}.csv.gz"
    pairs = read_csv_gz(repo_path(pop_path))
    if args.limit:
        pairs = pairs[:args.limit]
    out_root = f"{cfg['runtime']['outputs_root']}/outputs/{args.population}/{args.config_id}"
    run_dir = repo_path(f"data/manifests/phase3/runs/{args.population}")
    partial = run_dir / f"{args.config_id}.partial.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["status"] == "OK" and repo_path(r["output_path"]).exists():
                done[r["pair_id"]] = r
    tasks = [{"pair": p, "out_path": f"{out_root}/{p['cell']}/{p['ref_id']}.png"}
             for p in pairs if p["pair_id"] not in done]
    # interleave cells so partial progress covers every input size
    tasks.sort(key=lambda t: (t["pair"]["ref_id"], t["pair"]["cell"]))
    print(f"{args.config_id}/{args.population}: {len(pairs)} pairs, {len(done)} done, {len(tasks)} to run, "
          f"workers={workers} threads={threads}", flush=True)

    t_start = time.time()
    load_reports, describe = {}, {}
    broken = None
    if tasks:
        # ProcessPoolExecutor (not multiprocessing.Pool): a worker killed during init/inference (e.g. ENOMEM under
        # host memory pressure) raises BrokenProcessPool instead of hanging forever on the lost task (P3-T4 log).
        # Completed records are already flushed; the caller re-invokes and resume runs only what is missing.
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init,
                                 initargs=(args.config_id, threads)) as ex, open(partial, "a", encoding="utf-8") as fh:
            futs = [ex.submit(_one, t) for t in tasks]
            try:
                for i, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    done[rec["pair_id"]] = rec
                    if i % 25 == 0 or rec["status"] != "OK":
                        print(f"  {i}/{len(tasks)} {rec['pair_id']} {rec['status']} {rec['infer_s']}s "
                              f"{rec['error']}", flush=True)
                load_reports, describe = ex.submit(_reports).result()   # from a live worker
            except BrokenProcessPool as e:
                broken = f"BrokenProcessPool: {e}"
                print(f"  worker pool broken ({e}); completed records kept — re-run to resume", flush=True)
    if not load_reports:
        _init(args.config_id, threads)
        load_reports, describe = _reports()

    # re-read partial so FAILED records (not in `done` on resume) are also kept
    final = {}
    for line in partial.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["pair_id"] in {p["pair_id"] for p in pairs}:
            if r["pair_id"] not in final or r["status"] == "OK":
                final[r["pair_id"]] = r
    rows = [final[p["pair_id"]] for p in pairs if p["pair_id"] in final]
    missing = [p["pair_id"] for p in pairs if p["pair_id"] not in final]
    manifest = f"data/manifests/phase3/runs/{args.population}/{args.config_id}.csv.gz"
    if not args.limit:
        write_csv_gz(repo_path(manifest), [{k: r[k] for k in FIELDS} for r in rows], FIELDS)
    times = np.array([float(r["infer_s"]) for r in rows if r["status"] == "OK"])
    import torch
    summary = {
        "run": run_info({"config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
                         "protocol_version": cfg["protocol_version"], "torch": torch.__version__,
                         "cpu_count_container": os.cpu_count(), "machine": platform.machine()}),
        "population": args.population, "population_manifest": pop_path,
        "population_sha256": sha256_file(repo_path(pop_path)),
        "configuration": describe, "load_reports": load_reports, "workers": workers, "threads_per_worker": threads,
        "n_pairs": len(pairs), "n_ok": int(sum(r["status"] == "OK" for r in rows)),
        "n_failed": int(sum(r["status"] != "OK" for r in rows)), "n_missing": len(missing),
        "failures": [{"pair_id": r["pair_id"], "error": r["error"]} for r in rows if r["status"] != "OK"],
        "wall_s_this_invocation": round(time.time() - t_start, 1), "pool_broken": broken,
        "infer_s_under_load": {"note": "per-image wall time inside a worker, concurrent workers + host background load; "
                                       "NOT the P3-T8 performance measurement",
                               "mean": round(float(times.mean()), 3) if len(times) else None,
                               "p50": round(float(np.median(times)), 3) if len(times) else None,
                               "p95": round(float(np.percentile(times, 95)), 3) if len(times) else None},
        "manifest": manifest if not args.limit else None,
        "manifest_sha256": sha256_file(repo_path(manifest)) if not args.limit else None,
    }
    if not args.limit:
        write_json(run_dir / f"{args.config_id}_run.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_pairs", "n_ok", "n_failed", "n_missing", "infer_s_under_load")}))


if __name__ == "__main__":
    main()
