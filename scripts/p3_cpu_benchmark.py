"""P3-T8 — CPU latency / resource benchmark (one consistent method for every configuration).

    scripts/dock.sh python scripts/p3_cpu_benchmark.py --threads 2 [--configs ...] [--repeats 5]

Each configuration runs in a FRESH process (spawn), one at a time, nothing else of the harness running:
  1. cold start: build networks + safe-load weights (SHA-256 check timed separately)
  2. per input size: 1 warm-up call (excluded), then `repeats` timed calls of adapter.enhance on the same real
     degraded image from the main population (end-to-end adapter time: geometry + model + resampling)
  3. peak RSS of the process (ru_maxrss) after all calls
Host background load is recorded before/after (/proc/loadavg of the Docker VM). CPU only — no GPU extrapolation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from multiprocessing import get_context

import numpy as np

from fes_data.common import load_config, read_csv_gz, repo_path, run_info, write_json

SIZE_CELLS = {224: "gaussblur-moderate", 112: "lowres-mild", 56: "lowres-moderate", 28: "lowres-severe",
              14: "lowres-extreme"}


def _loadavg() -> list[float]:
    try:
        return [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    except OSError:
        return []


def bench(args) -> dict:
    cid, threads, repeats, sizes, pairs = args
    import cv2
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    from fes_bench import adapters as AD
    from fes_data.common import repo_path as rp, sha256_file
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ad = AD.build(cid)
    t_sha = 0.0
    for k in ad.components:
        t = time.perf_counter()
        sha256_file(rp(AD.WEIGHTS[k][1]))
        t_sha += time.perf_counter() - t
    t0 = time.perf_counter()
    if ad.components:
        for k in ad.components:
            net, rep = AD.load_net(k, verify_sha=False)
            ad.load_reports[k] = rep
            if hasattr(ad, "nets"):
                ad.nets[k] = net
            else:
                ad.net = net
    load_s = time.perf_counter() - t0
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"config_id": cid, "threads": threads, "sha_check_s": round(t_sha, 3), "cold_load_s": round(load_s, 3),
           "rss_after_load_mb": round(rss_loaded / 1024, 1), "sizes": {}}
    for s in sizes:
        p = pairs[s]
        deg = cv2.imread(str(rp(p["degraded_path"])))
        ctx = {"s": deg.shape[0], "M_e2f": np.array(json.loads(p["ffhq512_transform_2x3"]))}
        t = time.perf_counter()
        ad.enhance(deg, ctx)
        warm = time.perf_counter() - t
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            ad.enhance(deg, ctx)
            ts.append(time.perf_counter() - t)
        ts = np.array(ts)
        out["sizes"][str(s)] = {"pair_id": p["pair_id"], "first_call_s": round(warm, 3),
                                "mean_s": round(float(ts.mean()), 4), "median_s": round(float(np.median(ts)), 4),
                                "min_s": round(float(ts.min()), 4), "max_s": round(float(ts.max()), 4),
                                "std_s": round(float(ts.std(ddof=1)), 4) if len(ts) > 1 else 0.0,
                                "throughput_img_s_single_process": round(1.0 / float(np.median(ts)), 4)}
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out["baseline_rss_mb"] = round(rss0 / 1024, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    configs = args.configs or list(cfg["configurations"])
    pop = read_csv_gz(repo_path("data/manifests/phase3/population_main.csv.gz"))
    ref = sorted({p["ref_id"] for p in pop})[0]
    pairs = {s: next(p for p in pop if p["ref_id"] == ref and p["cell"] == c) for s, c in SIZE_CELLS.items()}
    import torch
    res = {"run": run_info({"torch": torch.__version__, "machine": platform.machine(),
                            "cpu_count_visible": os.cpu_count(), "cpuinfo_model": _cpu_model()}),
           "method": {"process": "fresh spawned process per configuration, sequential", "warmup_calls": 1,
                      "repeats": args.repeats, "timed_region": "adapter.enhance end-to-end (uint8 in → uint8 out)",
                      "threads": args.threads, "precision": "fp32", "batch": 1, "reference": ref},
           "results": []}
    ctx = get_context("spawn")
    for cid in configs:
        kind = cfg["configurations"][cid]["kind"]
        sizes = [224, 56] if kind == "face" else [224, 112, 56, 28, 14]
        la0 = _loadavg()
        with ctx.Pool(1) as pool:
            r = pool.apply(bench, ((cid, args.threads, args.repeats, sizes, pairs),))
        r["loadavg_before"], r["loadavg_after"] = la0, _loadavg()
        res["results"].append(r)
        print(json.dumps({k: r[k] for k in ("config_id", "cold_load_s", "peak_rss_mb")}),
              {s: v["median_s"] for s, v in r["sizes"].items()}, r["loadavg_before"], flush=True)
    write_json(repo_path(f"data/manifests/phase3/perf/cpu_benchmark_t{args.threads}{args.tag}.json"), res)


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    main()
