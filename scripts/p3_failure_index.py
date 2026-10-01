"""P3-T5/T6/T7 — failure taxonomy flags, candidate failure statistics and a representative-sample index.

    python scripts/p3_failure_index.py --population main

Every flag is computed per pair from the metric files, relative to the same pair's B0 value, the evaluator noise floor,
or the clean reference. The flags are DESCRIPTIVE BINS for organising failure analysis. They are not pass/fail
thresholds, and TH-1…TH-11 stay TBD (P0-T4 §10). Writes data/manifests/phase3/results/<population>/:
  failure_flags.csv.gz     one row per (config, pair) with every flag
  failure_stats.csv        per config × flag (and × family): count, rate
  sample_index.csv         per config × flag: the most extreme pairs (pair_id, cell, metrics) → contact sheets
Only stdlib + numpy.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from fes_data.common import load_config, read_csv_gz, repo_path, write_csv_gz

FAMS = ["lowres", "gaussblur", "motionblur", "noise", "jpeg", "underexposure", "lowcontrast", "combined", "realistic2"]
# flag → (description, ordering key for "most extreme" samples; lower = more extreme)
FLAGS = {
    "quality_regression": "LPIPS worse than the same pair's B0 (enhancement made the image further from the reference)",
    "id_drift": "cos(output, ref) dropped vs B0 by more than the evaluator 1-px-shift noise p99",
    "id_flip": "B0 accepted at the FAR 1e-2 operating point, output rejected",
    "id_rescue": "B0 rejected, output accepted (identity recovered)",
    "id_closer_to_impostor": "output more similar to some OTHER main reference than to its own (misidentification risk)",
    "id_collapse": "cos(output, ref) < 0.2 (identity essentially absent; descriptive bin)",
    "geo_detect_lost": "YuNet finds a face on B0 but not on the output",
    "geo_landmark_shift": "landmark NME increased vs B0 by > 0.05 inter-ocular (descriptive bin)",
    "texture_oversharp": "Laplacian-variance ratio vs clean reference > 1.5 while LPIPS worse than B0 (synthetic texture)",
    "texture_oversmooth": "Laplacian-variance ratio vs clean reference < 0.5 (detail missing)",
    "colour_shift_added": "mean-Lab ΔE76 vs reference exceeds B0's by > 5 (colour introduced by the model; descriptive bin)",
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    ap.add_argument("--population", default="main")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mdir = repo_path(f"data/manifests/phase3/metrics/{args.population}")
    odir = repo_path(f"data/manifests/phase3/results/{args.population}")
    thr = json.loads((mdir / "identity_calibration.json").read_text())["threshold_at_far"]
    nf = json.loads(repo_path("data/manifests/phase3/identity_noise_floor.json").read_text())
    drop_cut = nf["perturbations"]["shift_1px"]["one_minus_cos"]["0.99"]
    configs = [c for c in cfg["configurations"] if (mdir / f"{c}.csv.gz").exists()]
    data = {c: read_csv_gz(mdir / f"{c}.csv.gz") for c in configs}
    b0 = {r["pair_id"]: r for r in data["B0"]}

    rows = []
    for c in configs:
        for r in data[c]:
            if r["status"] != "OK":
                continue
            b = b0[r["pair_id"]]
            idc, idb = fnum(r["id_cos"]), fnum(b["id_cos"])
            lp, lpb = fnum(r["lpips"]), fnum(b["lpips"])
            nme, nmeb = fnum(r["lm_nme"]), fnum(b["lm_nme"])
            sh, de, deb = fnum(r["sharp_ratio"]), fnum(r["color_de"]), fnum(b["color_de"])
            fl = {
                "quality_regression": lp > lpb,
                "id_drift": (idc - idb) < -drop_cut,
                "id_flip": idb >= thr > idc,
                "id_rescue": idc >= thr > idb,
                "id_closer_to_impostor": fnum(r["imp_cos_max"]) >= idc,
                "id_collapse": idc < 0.2,
                "geo_detect_lost": b["det_found"] == "True" and r["det_found"] != "True",
                "geo_landmark_shift": (not np.isnan(nme)) and (not np.isnan(nmeb)) and (nme - nmeb) > 0.05,
                "texture_oversharp": sh > 1.5 and lp > lpb,
                "texture_oversmooth": sh < 0.5,
                "colour_shift_added": (de - deb) > 5,
            }
            rows.append({"config_id": c, "pair_id": r["pair_id"], "ref_id": r["ref_id"], "cell": r["cell"],
                         "family": r["cell"].split("-")[0], "severity": r["severity"], "in_size": r["in_size"],
                         "lpips": r["lpips"], "d_lpips": round(lp - lpb, 5), "psnr": r["psnr"],
                         "id_cos": r["id_cos"], "d_id": round(idc - idb, 5), "imp_cos_max": r["imp_cos_max"],
                         "lm_nme": r["lm_nme"], "sharp_ratio": r["sharp_ratio"], "color_de": r["color_de"],
                         **{k: int(bool(v)) for k, v in fl.items()}})
    write_csv_gz(odir / "failure_flags.csv.gz", rows)

    stats = []
    for c in configs:
        rc = [r for r in rows if r["config_id"] == c]
        for f in FLAGS:
            st = {"config_id": c, "flag": f, "n": len(rc), "count": sum(r[f] for r in rc)}
            st["rate"] = round(st["count"] / max(len(rc), 1), 4)
            for fam in FAMS:
                st[fam] = sum(r[f] for r in rc if r["family"] == fam)
            stats.append(st)
    with open(odir / "failure_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(stats[0]))
        w.writeheader()
        w.writerows(stats)

    key = {"quality_regression": lambda r: -r["d_lpips"], "id_drift": lambda r: r["d_id"],
           "id_flip": lambda r: r["d_id"], "id_rescue": lambda r: -r["d_id"],
           "id_closer_to_impostor": lambda r: fnum(r["id_cos"]) - fnum(r["imp_cos_max"]),
           "id_collapse": lambda r: fnum(r["id_cos"]), "geo_detect_lost": lambda r: r["d_id"],
           "geo_landmark_shift": lambda r: -fnum(r["lm_nme"]), "texture_oversharp": lambda r: -fnum(r["sharp_ratio"]),
           "texture_oversmooth": lambda r: fnum(r["sharp_ratio"]), "colour_shift_added": lambda r: -fnum(r["color_de"])}
    idx = []
    for c in configs:
        if c == "B0":
            continue
        for f in FLAGS:
            hit = sorted([r for r in rows if r["config_id"] == c and r[f]], key=key[f])[:args.k]
            for rank, r in enumerate(hit, 1):
                idx.append({"config_id": c, "flag": f, "rank": rank, "pair_id": r["pair_id"], "cell": r["cell"],
                            "in_size": r["in_size"], "lpips": r["lpips"], "d_lpips": r["d_lpips"],
                            "id_cos": r["id_cos"], "d_id": r["d_id"], "sharp_ratio": r["sharp_ratio"],
                            "color_de": r["color_de"]})
    with open(odir / "sample_index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(idx[0]))
        w.writeheader()
        w.writerows(idx)
    print("flags:", len(rows), "stats:", len(stats), "samples:", len(idx))


if __name__ == "__main__":
    main()
