"""P3-T5/P3-T6/P3-T7/P3-T9 — aggregate per-pair metrics into tables with cluster-bootstrap CIs (P3-T1 §8).

    python scripts/p3_aggregate.py --population main

Writes data/manifests/phase3/results/<population>/:
  cells.csv            per configuration × cell: n, failures, mean/median/p5/p95 of every metric
  deltas.csv           paired differences vs B0 and vs B2 per configuration × {cell, family, severity, all}:
                       mean difference + 95 % cluster-bootstrap CI (resampling references) + tie flag
  identity.csv         per configuration × {cell, family, severity, all}: TAR/FAR at the FAR 1e-2 operating
                       point, ID-gain sign rates, drops beyond the 1-px-shift noise p99 (descriptive)
  strata.csv           per configuration × race × gender (descriptive; 2 identities per stratum on main)
  summary.json         provenance + input metric-file hashes
Only stdlib + numpy (runs in the host venv or the container).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fes_data.common import load_config, read_csv_gz, repo_path, run_info, sha256_file, write_csv_gz, write_json

METRICS = ["psnr", "ssim", "lpips", "id_cos", "lm_nme", "sharp_ratio", "color_de"]
DELTA_METRICS = {"psnr": "higher", "ssim": "higher", "lpips": "lower", "id_cos": "higher", "lm_nme": "lower"}
SEV_ORDER = ["mild", "moderate", "severe", "extreme"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def boot_ci(mat: np.ndarray, B: int, seed: int) -> tuple[float, float, float]:
    """mat: refs × cells (NaN = missing). Statistic = mean over all non-NaN entries. Resample refs."""
    rng = np.random.default_rng(seed)
    s = np.nansum(mat, axis=1)
    c = np.sum(~np.isnan(mat), axis=1)
    est = float(s.sum() / max(c.sum(), 1))
    idx = rng.integers(0, len(s), (B, len(s)))
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    ap.add_argument("--population", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    B, seed = cfg["statistics"]["bootstrap"]["B"], cfg["statistics"]["bootstrap"]["seed"]
    mdir = repo_path(f"data/manifests/phase3/metrics/{args.population}")
    odir = repo_path(f"data/manifests/phase3/results/{args.population}")
    calib = json.loads((mdir / "identity_calibration.json").read_text())
    nf = json.loads(repo_path("data/manifests/phase3/identity_noise_floor.json").read_text())
    drop_cut = nf["perturbations"]["shift_1px"]["one_minus_cos"]["0.99"]

    configs = [c for c in cfg["configurations"] if (mdir / f"{c}.csv.gz").exists()]
    data = {c: read_csv_gz(mdir / f"{c}.csv.gz") for c in configs}
    pairs = [r["pair_id"] for r in data["B0"]]
    refs = sorted({r["ref_id"] for r in data["B0"]})
    cells = sorted({r["cell"] for r in data["B0"]},
                   key=lambda c: (c.split("-")[0], SEV_ORDER.index(c.split("-")[1])))
    ri = {r: i for i, r in enumerate(refs)}
    ci_ = {c: i for i, c in enumerate(cells)}
    for c in configs:
        assert [r["pair_id"] for r in data[c]] == pairs, f"{c}: population/order differs from B0"

    def cube(c, m):
        a = np.full((len(refs), len(cells)), np.nan)
        for r in data[c]:
            if r["status"] == "OK":
                a[ri[r["ref_id"]], ci_[r["cell"]]] = fnum(r[m])
        return a

    cubes = {c: {m: cube(c, m) for m in METRICS} for c in configs}
    groups = [("cell", c, [c]) for c in cells]
    fams = sorted({c.split("-")[0] for c in cells})
    groups += [("family", f, [c for c in cells if c.startswith(f + "-")]) for f in fams]
    groups += [("severity", s, [c for c in cells if c.endswith("-" + s)]) for s in SEV_ORDER]
    groups += [("all", "all", cells)]

    # ---- per cell descriptive
    rows = []
    for c in configs:
        st = {r["cell"]: [0, 0] for r in data[c]}
        for r in data[c]:
            st[r["cell"]][0 if r["status"] == "OK" else 1] += 1
        det = {}
        for r in data[c]:
            if r["status"] == "OK":
                det.setdefault(r["cell"], []).append(r["det_found"] == "True")
        for cell in cells:
            row = {"config_id": c, "cell": cell, "n_ok": st[cell][0], "n_not_ok": st[cell][1],
                   "det_rate": round(float(np.mean(det.get(cell, [np.nan]))), 4)}
            for m in METRICS:
                v = cubes[c][m][:, ci_[cell]]
                v = v[~np.isnan(v)]
                for k, f in (("mean", np.mean), ("median", np.median), ("p5", lambda x: np.percentile(x, 5)),
                             ("p95", lambda x: np.percentile(x, 95))):
                    row[f"{m}_{k}"] = round(float(f(v)), 5) if len(v) else ""
            rows.append(row)
    write_csv_gz(odir / "cells.csv.gz", rows)

    # ---- paired deltas vs B0 and B2 with cluster bootstrap
    drows = []
    for base in ("B0", "B2"):
        for c in configs:
            if c == base:
                continue
            for gk, gname, gcells in groups:
                sel = [ci_[x] for x in gcells]
                row = {"config_id": c, "baseline": base, "group_kind": gk, "group": gname}
                for m, better in DELTA_METRICS.items():
                    d = cubes[c][m][:, sel] - cubes[base][m][:, sel]
                    est, lo, hi = boot_ci(d, B, seed)
                    tie = lo <= 0 <= hi
                    good = (est > 0) if better == "higher" else (est < 0)
                    row.update({f"d_{m}": round(est, 5), f"d_{m}_lo": round(lo, 5), f"d_{m}_hi": round(hi, 5),
                                f"d_{m}_verdict": "tie" if tie else ("better" if good else "worse")})
                row["n_pairs"] = int(np.sum(~np.isnan(cubes[c]["id_cos"][:, sel] - cubes[base]["id_cos"][:, sel])))
                drows.append(row)
    write_csv_gz(odir / "deltas.csv.gz", drows)

    # ---- identity operating point + drop rates
    irows = []
    b0 = {r["pair_id"]: r for r in data["B0"]}
    for c in configs:
        for gk, gname, gcells in groups:
            rs = [r for r in data[c] if r["cell"] in set(gcells) and r["status"] == "OK"]
            g = np.array([fnum(r["id_cos"]) - fnum(b0[r["pair_id"]]["id_cos"]) for r in rs])
            tar = np.mean([r["id_ge_thr"] == "True" for r in rs]) if rs else np.nan
            imp_n = sum(int(r["imp_n"]) for r in rs)
            imp_hit = sum(int(r["imp_n_ge_thr"]) for r in rs)
            irows.append({"config_id": c, "group_kind": gk, "group": gname, "n": len(rs),
                          "id_cos_mean": round(float(np.mean([fnum(r["id_cos"]) for r in rs])), 5) if rs else "",
                          "id_cos_p5": round(float(np.percentile([fnum(r["id_cos"]) for r in rs], 5)), 5) if rs else "",
                          "tar_at_far1e-2": round(float(tar), 4), "far_observed": round(imp_hit / max(imp_n, 1), 5),
                          "imp_cos_mean": round(float(np.mean([fnum(r["imp_cos_mean"]) for r in rs])), 5) if rs else "",
                          "gain_mean": round(float(g.mean()), 5) if len(g) else "",
                          "gain_p5": round(float(np.percentile(g, 5)), 5) if len(g) else "",
                          "frac_gain_neg": round(float(np.mean(g < 0)), 4) if len(g) else "",
                          "frac_drop_beyond_shift1px_p99": round(float(np.mean(g < -drop_cut)), 4) if len(g) else ""})
    write_csv_gz(odir / "identity.csv.gz", irows)

    # ---- strata (descriptive)
    srows = []
    for c in configs:
        strata = sorted({(r["race"], r["gender"]) for r in data[c]})
        for race, gender in strata:
            rs = [r for r in data[c] if r["race"] == race and r["gender"] == gender and r["status"] == "OK"]
            g = [fnum(r["id_cos"]) - fnum(b0[r["pair_id"]]["id_cos"]) for r in rs]
            srows.append({"config_id": c, "race": race, "gender": gender, "n_refs": len({r["ref_id"] for r in rs}),
                          "n_pairs": len(rs), "id_cos_mean": round(float(np.mean([fnum(r["id_cos"]) for r in rs])), 5),
                          "gain_mean": round(float(np.mean(g)), 5),
                          "lpips_mean": round(float(np.mean([fnum(r["lpips"]) for r in rs])), 5)})
    write_csv_gz(odir / "strata.csv.gz", srows)

    write_json(odir / "summary.json", {
        "run": run_info(), "population": args.population, "configs": configs, "n_refs": len(refs),
        "n_cells": len(cells), "bootstrap": cfg["statistics"]["bootstrap"],
        "identity_threshold_far1e-2": calib["threshold_at_far"], "drop_cut_shift1px_p99": drop_cut,
        "metric_files": {c: sha256_file(mdir / f"{c}.csv.gz") for c in configs},
        "outputs": {n: sha256_file(odir / n) for n in ("cells.csv.gz", "deltas.csv.gz", "identity.csv.gz",
                                                        "strata.csv.gz")}})
    print("aggregated", configs, len(refs), "refs", len(cells), "cells")


if __name__ == "__main__":
    main()
