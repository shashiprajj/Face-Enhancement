"""R5-3 step 8 — synthetic regression of the rw gate vs the frozen T6/T7 policy (no new inference).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_rwgate_synthetic.py

On `main` (synthetic TEST, never used for calibration) and `p4dev` (dev, in-sample): decisions of both policies from
the same inputs (stored p4-quality features + lapd1_c computed from the degraded image), and the identity outcome of
"gate + A4" using the A4/B0 outputs already scored in Phase 3/4 (SFace E01, dlib E07; fes_bench.p4eval rules).
Reported explicitly: enhancement rate, ΔID with CI, confirmed benefit / harm, flips, MILD inputs enhanced, the A4 output
guard (no face on output → original), and which pairs change route between the two policies.
Writes data/manifests/phase5_rapid/rwgate/synthetic_regression.{json,md}.
"""
from __future__ import annotations

import json
from collections import Counter

import cv2

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import routing as RT
from fes_data.common import read_csv_gz, repo_path, run_info, write_json
from fes_service import rwgate as RW


def main() -> None:
    cfg = RW.load_config()
    g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    rules = GT.load_rules(g6)
    t, table = RT.load_routing(json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8")), g6)
    delta = E.deltas_from_validation()
    res = {"run": run_info(), "rw_seal": cfg["sha256"], "populations": {}}
    md = ["# R5-3 synthetic regression — frozen T6/T7 vs rw gate (A4 outputs already scored)", ""]
    for pop in ("main", "p4dev"):
        pairs = E.load(pop)
        rows = read_csv_gz(repo_path(f"{E.POPS[pop]['root']}/population_{pop}.csv.gz"))
        path = {r["pair_id"]: r["degraded_path"] for r in rows}
        thr = E.thresholds(pop)
        used = {"T6/T7": {}, "rw": {}}
        reasons = Counter()
        for pid, p in pairs.items():
            d, _ = GT.decide(p.feat, rules)
            r_old = table.get(RT.bucket_of(p.feat, t), RT.ORIGINAL) if d == GT.ENH else RT.ORIGINAL
            f = dict(p.feat)
            f["lapd1_c"] = RW.lapd1(cv2.imread(str(repo_path(path[pid])), cv2.IMREAD_COLOR))
            _, why, _, r_new = RW.decide(f, cfg)
            reasons[why.split(":")[0]] += 1
            for k, r in (("T6/T7", r_old), ("rw", r_new)):
                # output guard as in the router / p4_route_evaluate: no face on the A4 output → original
                used[k][pid] = r if (r == RT.ORIGINAL or p.det_out.get(r, False)) else RT.ORIGINAL
        P = sorted(pairs.values(), key=lambda p: p.pair_id)
        out = {"reasons_rw": dict(reasons), "policies": {}}
        for k, u in used.items():
            m = E.policy_metrics(P, u, delta, thr)
            m["mild"] = E.policy_metrics([p for p in P if p.severity == "mild"], u, delta, thr)
            m["by_family_enhanced"] = dict(Counter(p.profile for p in P if u[p.pair_id] != RT.ORIGINAL))
            out["policies"][k] = m
        changed = [p for p in P if used["T6/T7"][p.pair_id] != used["rw"][p.pair_id]]
        out["changed"] = {"n": len(changed),
                          "added_by_rw": dict(Counter(f"{p.cell}:{E.multi_of(p, 'A4', delta)}" for p in changed
                                                      if used["rw"][p.pair_id] == "A4")),
                          "removed_by_rw": dict(Counter(f"{p.cell}:{E.multi_of(p, 'A4', delta)}" for p in changed
                                                        if used["T6/T7"][p.pair_id] == "A4"))}
        res["populations"][pop] = out
        role = "TEST (not used for calibration)" if pop == "main" else "DEV (in-sample)"
        md += [f"## `{pop}` — {role}", "", "| Policy | A4 routed | ΔID SFace | ΔID dlib | conf. benefit | harm (any) | "
               "flips SFace/dlib | MILD enhanced |", "|---|---|---|---|---|---|---|---|"]
        for k, m in out["policies"].items():
            f2 = lambda d: f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {d[3]}"  # noqa: E731
            md.append(f"| {k} | {m['enhanced']} | {f2(m['d_E01'])} | {f2(m['d_E07'])} | {m['confirmed_benefit']} | "
                      f"{m['harm_any']} | {m['flips_E01']}/{m['flips_E07']} | {m['mild']['enhanced']} |")
        md += ["", f"Routes changed: {out['changed']['n']} · added by rw: {out['changed']['added_by_rw']} · "
                   f"removed by rw: {out['changed']['removed_by_rw']}", "",
               f"rw reasons: {out['reasons_rw']}", ""]
    out_dir = repo_path("data/manifests/phase5_rapid/rwgate")
    write_json(out_dir / "synthetic_regression.json", res)
    (out_dir / "synthetic_regression.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
