"""Calibrate the blur boundary of the side-face (detection-box) framing on DEV data and write the sealed config
configs/phase5/side_face_route.json. Evaluation of the result is done on other originals (side_run.py --name test).

    python scripts/indian_bench/calibrate_side.py [--write]

DEV = side_profile_run/dev: profile best_side run with the frontal boundary (0.008) on the turned faces of the DEV
expression folders, blur types at levels 1-4. For every ENHANCED answer: lapd1_c of the box chip (the gate's sharpness
feature) and Δ PSNR = PSNR(output) − PSNR(degraded input) against the clean reference in the same crop.

Rule (fixed before the full DEV set was read): on the grid 0.0010 … 0.0080 in steps of 0.00025, blur_max is the
largest T such that EVERY 0.0005-wide lapd1 bin below T has a positive mean Δ PSNR (bins of fewer than 20 images are
joined with the next one). clean_lapd1_lo = blur_max (no mild-blur band, as in the r6 route). Nothing else changes.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter

import common as C

STEP, WIDTH, MIN_N = 0.00025, 0.0005, 20


def main(a):
    from fes_bench import gate as GT
    from fes_service import nafnet_route as NR
    cfg = C.load_config()
    d = C.bench(cfg, "side_profile_run", "dev")
    recs = list({r["variant_id"]: r for p in sorted(d.glob("results_*.jsonl")) for r in C.read_jsonl(p)}.values())
    enh = [r for r in recs if r["outcome"] == "enhanced" and "psnr_output" in r]
    pts = sorted((json.loads(r["quality"])["lapd1_c"], r["psnr_output"] - r["psnr_degraded"]) for r in enh)
    print(f"DEV: {len(recs)} requests, {len(enh)} enhanced with a comparison, "
          f"{len({r['sample_id'] for r in enh})} originals, expressions {sorted({r['expression'] for r in recs})}")
    bins, lo, T = [], 0.0, None
    edge = WIDTH
    while lo < 0.008:
        g = [dv for x, dv in pts if lo <= x < edge]
        if len(g) < MIN_N and edge < 0.008:
            edge = round(edge + STEP, 6)
            continue
        m = st.fmean(g) if g else float("nan")
        bins.append({"lapd1_from": round(lo, 6), "lapd1_to": round(edge, 6), "n": len(g), "mean_delta_psnr": round(m, 3),
                     "better_share": round(sum(x > 0 for x in g) / len(g), 3) if g else None})
        print(f"  [{lo:.5f}, {edge:.5f})  n={len(g):4}  mean ΔPSNR {m:+.3f}")
        if T is None and not m > 0:
            T = round(lo, 6)
        lo, edge = edge, round(edge + WIDTH, 6)
    T = T if T is not None else 0.008
    below = [dv for x, dv in pts if x < T]
    above = [dv for x, dv in pts if x >= T]
    print(f"blur_max (box framing) = {T}  ·  below: n={len(below)}, mean ΔPSNR {st.fmean(below):+.3f}, better in "
          f"{100 * sum(x > 0 for x in below) / len(below):.0f} %  ·  at/above: n={len(above)}, mean ΔPSNR {st.fmean(above):+.3f}")
    lv = Counter(r["level"] for r in enh if json.loads(r["quality"])["lapd1_c"] < T)
    al = Counter(r["level"] for r in enh)
    print("admitted share of the DEV enhanced by level:", {l: f"{lv[l]}/{al[l]}" for l in sorted(al)})
    out = {"route_version": NR.SIDE_ROUTE_VERSION, "applies_to": "pipeline profile best_side, detection-box framing only",
           "thresholds": {"blur_max": T, "clean_lapd1_lo": T},
           "frontal_route": {"config": NR.R6_CONFIG, "blur_max": NR.load_config(NR.R6_CONFIG)["thresholds"]["blur_max"]},
           "calibration": {"date": "2026-10-13", "data": "DEV only: Indian-expression benchmark, turned faces of "
                           + ", ".join(sorted({r["expression"] for r in recs})) + "; synthetic blur, levels 1-4; "
                           "types " + ", ".join(sorted({r["type"] for r in recs})),
                           "requests": len(recs), "enhanced": len(enh), "originals": len({r["sample_id"] for r in enh}),
                           "rule": "largest T on a 0.00025 grid such that every 0.0005-wide lapd1_c bin below T has "
                                   "a positive mean delta PSNR (output vs degraded, against the clean reference)",
                           "bins": bins,
                           "limits": "synthetic blur on 800 px references of one dataset; persons may repeat between "
                                     "DEV and TEST folders; PSNR is not an identity measure"}}
    out["sha256"] = GT.gate_hash(out)
    if a.write:
        p = C.ROOT / NR.SIDE_CONFIG
        p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("written", p, out["sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args())
