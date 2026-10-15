"""Compression / noise rule for the side-face (detection-box) framing: calibrate on DEV, write the sealed config
configs/phase5/side_face_route.json (route p7-side-face-route-1.1.0), and replay the rule on TEST.

    python scripts/indian_bench/calibrate_side_comp.py [--write]

Why: on TEST the box-framed path enhanced mildly blurred JPEG / noisy side faces and made them worse (motion_noise_jpeg
-2.08 dB, down_motion_noise_jpeg -3.91 dB, jpeg -1.9 dB, down_jpeg -2.12 dB, blur_jpeg -0.31 dB).

DEV (turned faces of Anger, Confused, Excited, Fear; never TEST):
  side_profile_run/dev       the six blur types, run at blur_max 0.008; the requests with lapd1_c < 0.003 are exactly
                             the ones the sealed 1.0.0 side route enhances (same chip, same route, same output)
  side_profile_run/dev_comp  every other degradation type, run on the sealed 1.0.0 side route
For every ENHANCED answer: Δ PSNR = PSNR(output) − PSNR(degraded input) against the clean reference in the box crop.

Attempt 1 (RED, kept for the record: comp_rule/replay_v1_noise_RED.md): bins of 50 by noise_sigma, then blockiness;
reject above the leading run of positive bins. One flat bin at noise 0.28 cut every box chip above it, and on box chips
noise_sigma is high for diagonal motion blur (DEV median 0.45, +2.0 dB) — TEST lost 890 of 1670 improved blur answers.

Attempt 2 (this rule; feature chosen from DEV per-type medians only, before TEST was replayed with it): the gate's
block-grid feature `pixelation` (periodicity of the gradient profile; JPEG 8x8 blocking raises it). On DEV the
JPEG / noise / compression combinations that got worse have median 0.49, every blur type 0.16-0.31.
  pixelation_max = the T (a DEV value, or the frontal limit) that maximises the summed Δ PSNR of the DEV enhancements
                   kept (pixelation ≤ T); ties → the largest T. Never looser than the frontal route.
noise_max and blockiness_hi stay at the frontal values. Nothing else changes (blur boundary 0.003 kept).

TEST acceptance (offline replay: the rule only turns ENHANCED into ORIGINAL before inference, so every other answer is
unchanged; checked live afterwards with side_run.py on a sample):
  A1 the TEST enhancements the rule removes have a negative mean Δ PSNR
  A2 the blur-only types (category blur) keep ≥ 90 % of their enhancements that were better than the input
  A3 the JPEG / noise combination types (blur_jpeg, down_jpeg, motion_noise_jpeg, down_motion_noise_jpeg, jpeg) have a
     mean Δ PSNR ≥ 0 on what is still enhanced (or nothing is still enhanced)
  GREEN = A1 + A2 + A3 · YELLOW = A1 + A2 · RED otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter

import common as C

BIN = 50
OLD_BLUR_MAX = 0.003
COMBO = ("blur_jpeg", "down_jpeg", "motion_noise_jpeg", "down_motion_noise_jpeg", "jpeg")
FEATURES = (("pixelation", "pixelation_max"),)


def load(d) -> list[dict]:
    recs = list({r["variant_id"]: r for p in sorted(d.glob("results_*.jsonl")) for r in C.read_jsonl(p)}.values())
    for r in recs:
        r["q"] = json.loads(r["quality"]) if r.get("quality") else {}
    return recs


def enhanced(recs) -> list[dict]:
    out = []
    for r in recs:
        if r["outcome"] == "enhanced" and r.get("framing") == "detection_box" and "psnr_output" in r:
            r["d"] = r["psnr_output"] - r["psnr_degraded"]
            out.append(r)
    return out


def threshold(pts: list[tuple[float, float]]) -> tuple[float | None, list[dict]]:
    """pts = (feature, Δ PSNR). Returns (largest kept feature value or None if every bin is positive, bins)."""
    pts = sorted(pts)
    edges = list(range(0, len(pts), BIN))
    if len(edges) > 1 and len(pts) - edges[-1] < BIN:
        edges.pop()
    bins, keep = [], None
    for i, s in enumerate(edges):
        g = pts[s:edges[i + 1] if i + 1 < len(edges) else len(pts)]
        m = st.fmean(d for _, d in g)
        bins.append({"from": round(g[0][0], 6), "to": round(g[-1][0], 6), "n": len(g), "mean_delta_psnr": round(m, 3),
                     "better_share": round(sum(d > 0 for _, d in g) / len(g), 3)})
        if keep is None and not m > 0:
            keep = pts[s - 1][0] if s > 0 else g[0][0] - 1e-6
    return keep, bins


def best_cut(pts: list[tuple[float, float]], limit: float) -> float:
    """pts = (feature, Δ PSNR). The T ≤ limit that maximises the summed Δ of the points with feature ≤ T."""
    pts = sorted(p for p in pts if p[0] <= limit)
    best_t, best, run = limit, sum(d for _, d in pts), 0.0
    for i, (x, d) in enumerate(pts):
        run += d
        if (i + 1 == len(pts) or pts[i + 1][0] > x) and run > best:
            best_t, best = x, run
    return best_t


def rejects(r, t) -> bool:
    return all(r["q"].get(f) is not None and r["q"][f] > t[k] for f, k in FEATURES)


def main(a):
    from fes_bench import gate as GT
    from fes_service import nafnet_route as NR
    cfg = C.load_config()
    base = C.bench(cfg, "side_profile_run", "dev")
    matrix = {t["type"]: t["category"] for t in json.loads((C.ROOT / "scripts/indian_bench/degradation_matrix.json")
                                                           .read_text(encoding="utf-8"))["types"]}
    old = NR.load_side_config()
    front = NR.load_config(NR.R6_CONFIG)["thresholds"]
    dev_blur = [r for r in enhanced(load(base)) if r["q"]["lapd1_c"] < OLD_BLUR_MAX]
    dev_comp = enhanced(load(base.parent / "dev_comp"))
    dev = dev_blur + dev_comp
    print(f"DEV enhanced: {len(dev)} ({len(dev_blur)} six blur types + {len(dev_comp)} other types), "
          f"{len({r['sample_id'] for r in dev})} originals, mean Δ PSNR {st.fmean(r['d'] for r in dev):+.3f}")
    t, steps, kept = dict(old["thresholds"]), {}, dev
    for feat, key in FEATURES:
        pts = [(r["q"][feat], r["d"]) for r in kept if r["q"].get(feat) is not None]
        _, bins = threshold(pts)                                  # reported only
        t[key] = round(best_cut(pts, front[key]), 6)
        steps[key] = {"feature": feat, "value": t[key], "frontal": front[key], "bins": bins}
        print(f"\n{feat}: bins of {BIN}")
        for b in bins:
            print(f"  [{b['from']:.4f}, {b['to']:.4f}]  n={b['n']:3}  mean ΔPSNR {b['mean_delta_psnr']:+.3f}  better {b['better_share']:.0%}")
        kept = [r for r in kept if not (r["q"].get(feat) is not None and r["q"][feat] > t[key])]
        print(f"  → {key} = {t[key]} (frontal {front[key]}); DEV kept {len(kept)}")
    gone = [r for r in dev if rejects(r, t)]
    print(f"\nDEV removed {len(gone)}: mean ΔPSNR {st.fmean(r['d'] for r in gone) if gone else float('nan'):+.3f}; kept "
          f"{len(dev) - len(gone)}: mean ΔPSNR {st.fmean(r['d'] for r in dev if not rejects(r, t)):+.3f}")
    print("DEV removed by type:", dict(Counter(r["type"] for r in gone).most_common()))

    # ---------------------------------------------------------------- TEST replay
    test = enhanced(load(base.parent / "test"))
    test = [r for r in test if r["group"] == "turned"]
    rm = [r for r in test if rejects(r, t)]
    keep = [r for r in test if not rejects(r, t)]
    blur = [r for r in test if matrix[r["type"]] == "blur"]
    blur_better = [r for r in blur if r["d"] > 0]
    blur_better_kept = [r for r in blur_better if not rejects(r, t)]
    combo_kept = [r for r in keep if r["type"] in COMBO]
    a1 = bool(rm) and st.fmean(r["d"] for r in rm) < 0
    a2 = len(blur_better_kept) >= 0.9 * len(blur_better)
    a3 = not combo_kept or st.fmean(r["d"] for r in combo_kept) >= 0
    verdict = "GREEN" if a1 and a2 and a3 else ("YELLOW" if a1 and a2 else "RED")
    rows = []
    for ty in sorted({r["type"] for r in test}):
        b, k = [r for r in test if r["type"] == ty], [r for r in keep if r["type"] == ty]
        rows.append({"type": ty, "enhanced before": len(b), "Δ PSNR before": round(st.fmean(r["d"] for r in b), 2),
                     "better before": f"{100 * sum(r['d'] > 0 for r in b) / len(b):.0f} %",
                     "enhanced after": len(k), "Δ PSNR after": round(st.fmean(r["d"] for r in k), 2) if k else "-",
                     "better after": f"{100 * sum(r['d'] > 0 for r in k) / len(k):.0f} %" if k else "-",
                     "worse removed": sum(r["d"] <= 0 for r in b) - sum(r["d"] <= 0 for r in k),
                     "better removed": sum(r["d"] > 0 for r in b) - sum(r["d"] > 0 for r in k)})
    lines = [f"TEST (turned faces, 4 held-back folders): enhanced {len(test)} → {len(keep)}; removed {len(rm)} "
             f"(mean ΔPSNR {st.fmean(r['d'] for r in rm) if rm else float('nan'):+.3f}, worse than input in "
             f"{sum(r['d'] <= 0 for r in rm)} of {len(rm)})",
             f"mean ΔPSNR all enhanced: {st.fmean(r['d'] for r in test):+.3f} → {st.fmean(r['d'] for r in keep):+.3f}; "
             f"better than input: {100 * sum(r['d'] > 0 for r in test) / len(test):.0f} % → {100 * sum(r['d'] > 0 for r in keep) / len(keep):.0f} %",
             f"A1 removed mean Δ < 0: {a1} · A2 blur types keep {len(blur_better_kept)}/{len(blur_better)} better "
             f"({100 * len(blur_better_kept) / len(blur_better):.1f} %) ≥ 90 %: {a2} · A3 combo types still enhanced "
             f"{len(combo_kept)}, mean Δ {st.fmean(r['d'] for r in combo_kept) if combo_kept else float('nan'):+.3f} ≥ 0: {a3}",
             f"VERDICT {verdict}", "", C.md_table(rows, list(rows[0]))]
    print("\n" + "\n".join(lines))
    out_dir = base.parent / "comp_rule"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "replay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "removed_test.json").write_text(json.dumps(sorted(r["variant_id"] for r in rm)), encoding="utf-8")

    new = {**{k: v for k, v in old.items() if k != "sha256"}, "route_version": NR.SIDE_ROUTE_VERSION_COMP,
           "thresholds": t,
           "compression_noise": {"date": "2026-10-14", "rule": "pixelation_max = the DEV value that maximises the summed "
                                 "delta PSNR of the DEV enhancements kept (pixelation <= T); never looser than the "
                                 "frontal route (attempt 2; attempt 1 on noise_sigma / blockiness was RED on TEST)",
                                 "data": f"DEV only ({len(dev)} enhanced turned-face requests of Anger, Confused, "
                                         "Excited, Fear; all degradation types)",
                                 "steps": steps, "test_replay": {"verdict": verdict, "enhanced_before": len(test),
                                                                 "enhanced_after": len(keep)}}}
    new["sha256"] = GT.gate_hash(new)
    if a.write:
        p = C.ROOT / NR.SIDE_CONFIG
        p.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("written", p, new["sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args())
