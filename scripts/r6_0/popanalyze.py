"""R6-0 — population-level comparison of verifier/output policies on archive val (DEV) and test (held-out).

    python scripts/r6_0/popanalyze.py  ->  data/manifests/phase5_rapid/r6_0/population_summary.json

Policy = (which candidates are returned, which image is returned). Rejected = ORIGINAL (identity change 0).
  R5-8         R5-8 global guard + local guard on v0, return v0           (current best existing verifier)
  T-only       run tracer (run_dot_c_max <= T), return v0
  T+R5-8       both, return v0
  T+luma       run tracer, return luma(v0)
  T+medluma    run tracer, return median of the 4 flip runs, luma-only  (R6-0 proposed "consensus" output)
  T+R5-8+medluma
T = DEV-GOOD 99th percentile of run_dot_c_max from analysis.json (fixed before the TEST population is read).
"""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "data/manifests/phase5_rapid/r6_0"


def load(split):
    rows = {}
    for line in gzip.open(M / f"population_{split}.jsonl.gz", "rt").read().splitlines():
        r = json.loads(line)
        rows.setdefault((r["name"], r["mode"]), {})[r["variant"]] = r
    return rows


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    T = json.loads((M / "analysis.json").read_text())["thresholds"]["run_dot_c_max"]
    out = {"T_run_dot_c_max": T}
    for split in ("val", "test"):
        if not (M / f"population_{split}.jsonl.gz").exists():
            continue
        rows = load(split)
        out[split] = {}
        for mode in ("fc", "fi"):
            cases = [v for (n, m), v in rows.items() if m == mode and "v0" in v and "medluma" in v]
            if not cases:
                continue
            v0 = lambda c: c["v0"]  # noqa: E731
            r58 = lambda c: v0(c)["guard_pass"] and v0(c)["r58_local_pass"]  # noqa: E731
            tr = lambda c: v0(c)["run_dot_c_max"] <= T  # noqa: E731
            pol = {"none (always v0)": (lambda c: True, "v0"), "R5-8": (r58, "v0"), "T-only": (tr, "v0"),
                   "T+R5-8": (lambda c: tr(c) and r58(c), "v0"), "T+luma": (tr, "luma"),
                   "T+meanluma": (tr, "meanluma"), "T+medluma": (tr, "medluma"),
                   "T+R5-8+medluma": (lambda c: tr(c) and r58(c), "medluma")}
            res = {"n_eligible": len(cases), "infer4_s_median": float(np.median([v0(c)["infer4_s"] for c in cases]))}
            for name, (acc, var) in pol.items():
                a = [acc(c) for c in cases]
                ds = [c[var]["d_sface"] if ok else 0.0 for c, ok in zip(cases, a)]
                dd = [c[var]["d_dlib"] if ok else 0.0 for c, ok in zip(cases, a)]
                acc_ds = [c[var]["d_sface"] for c, ok in zip(cases, a) if ok]
                res[name] = {"accept_rate": float(np.mean(a)), "n_reject": int(len(a) - sum(a)),
                             "policy_dsface": float(np.mean(ds)), "policy_dsface_ci": boot_ci(ds),
                             "policy_ddlib": float(np.mean(dd)),
                             "accepted_dsface": float(np.mean(acc_ds)) if acc_ds else None,
                             "harmed_lt_-0.05": int(sum(1 for c, ok in zip(cases, a) if ok and c[var]["d_sface"] < -0.05)),
                             "improved_gt_0.05": int(sum(1 for c, ok in zip(cases, a) if ok and c[var]["d_sface"] > 0.05))}
            out[split][mode] = res
    (M / "population_summary.json").write_text(json.dumps(out, indent=1))
    for split in ("val", "test"):
        for mode in ("fc", "fi"):
            r = out.get(split, {}).get(mode)
            if not r:
                continue
            print(f"== {split} {mode} n={r['n_eligible']} infer4 {r['infer4_s_median']:.2f}s")
            for k, v in r.items():
                if isinstance(v, dict):
                    print(f"  {k:18s} accept {v['accept_rate']:.3f} rej {v['n_reject']:3d}  policy dSF {v['policy_dsface']:+.4f} "
                          f"[{v['policy_dsface_ci'][0]:+.3f},{v['policy_dsface_ci'][1]:+.3f}] ddlib {v['policy_ddlib']:+.4f} "
                          f"acc dSF {v['accepted_dsface'] or 0:+.4f} harmed {v['harmed_lt_-0.05']} improved {v['improved_gt_0.05']}")


if __name__ == "__main__":
    main()
