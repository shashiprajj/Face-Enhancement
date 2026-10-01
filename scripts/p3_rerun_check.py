"""P3-T10 — reproducibility re-check: re-run a seeded sample of main-population pairs and compare pixels.

    scripts/dock.sh python scripts/p3_rerun_check.py [--per-config 4] [--threads 2] [--configs ...] [--tag t2]

For each configuration: re-hash the degraded input against the population manifest, rebuild the adapter in a fresh
process (weights SHA-verified at load), enhance the sampled pairs, and compare the pixel SHA-256 with the main run
manifest. A mismatch is reported with max |Δ| and PSNR between the original and the re-run output, so a
thread-count-dependent float difference is distinguishable from a real change. Writes
data/manifests/phase3/repro/rerun_<tag>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import get_context

import numpy as np

from fes_data.common import load_config, read_csv_gz, repo_path, run_info, write_json


def _check(args) -> list[dict]:
    cid, threads, pairs = args
    import cv2
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    from fes_bench import adapters as AD
    from fes_bench import read_bytes_retry
    from fes_data.common import repo_path as rp, sha256_bytes
    ad = AD.build(cid)
    ad.load()
    res = []
    for p, run in pairs:
        data = read_bytes_retry(rp(p["degraded_path"]))
        assert sha256_bytes(data) == p["degraded_sha256"], p["pair_id"]
        deg = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        ctx = {"s": deg.shape[0], "M_e2f": np.array(json.loads(p["ffhq512_transform_2x3"]), np.float64)}
        out, _ = ad.enhance(deg, ctx)
        same = sha256_bytes(out.tobytes()) == run["output_pixel_sha256"]
        rec = {"pair_id": p["pair_id"], "cell": p["cell"], "pixel_identical": same}
        if not same:
            orig = cv2.imdecode(np.frombuffer(read_bytes_retry(rp(run["output_path"])), np.uint8), cv2.IMREAD_COLOR)
            d = np.abs(orig.astype(np.int16) - out.astype(np.int16))
            mse = float((d.astype(np.float64) ** 2).mean())
            rec.update(max_abs_diff=int(d.max()), frac_px_diff=round(float((d > 0).any(2).mean()), 6),
                       psnr_vs_original=round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else None)
        res.append(rec)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--per-config", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    configs = args.configs or list(cfg["configurations"])
    pop = {p["pair_id"]: p for p in read_csv_gz(repo_path("data/manifests/phase3/population_main.csv.gz"))}
    out = {"run": run_info(), "threads": args.threads, "per_config": args.per_config,
           "sampling": "sha256('p3-rerun|<config>|<pair_id>') order, first N", "results": {}}
    ctx = get_context("spawn")
    for cid in configs:
        run = {r["pair_id"]: r for r in read_csv_gz(repo_path(f"data/manifests/phase3/runs/main/{cid}.csv.gz"))}
        ids = sorted(run, key=lambda k: hashlib.sha256(f"p3-rerun|{cid}|{k}".encode()).hexdigest())[:args.per_config]
        with ctx.Pool(1) as pool:
            res = pool.apply(_check, ((cid, args.threads, [(pop[i], run[i]) for i in ids]),))
        out["results"][cid] = res
        print(cid, sum(r["pixel_identical"] for r in res), "/", len(res), "pixel-identical",
              [r for r in res if not r["pixel_identical"]], flush=True)
    write_json(repo_path(f"data/manifests/phase3/repro/rerun{args.tag}.json"), out)


if __name__ == "__main__":
    main()
