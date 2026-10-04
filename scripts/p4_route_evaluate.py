"""P4-T7 — evaluate the FROZEN routing ONCE on the untouched SYNTHETIC test population `main` (offline replay of
already-benchmarked outputs; the runtime router is validated separately on dev by p4_router_validate.py).

    scripts/dock.sh python scripts/p4_route_evaluate.py [--population main|p4dev]

Policies compared (P4-T7): 1 always original · 2 always <candidate> · 3 T6 gate + <candidate> · 4 T7 route
(T6 gate + degradation-specific candidate). The INVALID_OUTPUT guard of the router is applied in replay with the same
detector rule (YuNet on the output, score ≥ 0.7): no face → original.
Reported (SFace E01 and dlib E07 separately): enhancement rate, route distribution, ΔID, rescues/flips, confirmed
benefit/harm, MILD damage, evaluator agreement on routed pairs, failure cases, bucket-vs-family confusion.
Writes data/manifests/phase4/t7/{evaluation_<pop>.json/.md, decision_log_<pop>.csv.gz}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import routing as RT
from fes_data.common import repo_path, run_info, sha256_file, write_csv_gz, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="main", choices=list(E.POPS))
    a = ap.parse_args()
    g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    rules = GT.load_rules(g6)
    r7 = json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))
    t, table = RT.load_routing(r7, g6)
    pairs = E.load(a.population)
    delta, thr = E.deltas_from_validation(), E.thresholds(a.population)
    P = sorted(pairs.values(), key=lambda p: p.pair_id)

    log, used7 = [], {}
    for p in P:
        d, why = GT.decide(p.feat, rules)
        b = RT.bucket_of(p.feat, t) if d == GT.ENH else ""
        route = table.get(b, RT.ORIGINAL) if d == GT.ENH else RT.ORIGINAL
        fb, used = "", route
        if route != RT.ORIGINAL and not p.det_out.get(route, False):
            fb, used = "INVALID_OUTPUT", RT.ORIGINAL
        used7[p.pair_id] = used
        log.append({"pair_id": p.pair_id, "ref_id": p.ref_id, "cell": p.cell, "decision": d, "reason": why,
                    "bucket": b, "route": route, "used": used, "fallback": fb, "escalate": d == GT.UNSAFE,
                    "d_E01": round(p.cos[(used, "E01")] - p.cos[("B0", "E01")], 5),
                    "d_E07": round(p.cos[(used, "E07")] - p.cos[("B0", "E07")], 5),
                    "verdict": E.multi_of(p, used, delta) if used != RT.ORIGINAL else ""})
    pols = {"1_always_original": {p.pair_id: "B0" for p in P}}
    for c in E.CANDIDATES:
        pols[f"2_always_{c}"] = {p.pair_id: c for p in P}
    for c in E.CANDIDATES:
        pols[f"3_T6+{c}"] = {p.pair_id: (c if GT.decide(p.feat, rules)[0] == GT.ENH else "B0") for p in P}
    pols["4_T7_route"] = used7
    groups = {"all": lambda p: True} | {f"severity={s}": (lambda s: lambda p: p.severity == s)(s) for s in E.SEV_ORDER} \
        | {f"family={f}": (lambda f: lambda p: p.profile == f)(f) for f in E.FAMILIES}
    res = {"population": a.population, "data_kind": "SYNTHETIC", "split": E.POPS[a.population]["split"],
           "role": "FINAL single evaluation of frozen routing" if a.population == "main" else "IN-SAMPLE dev reference",
           "t6_seal": g6["sha256"], "t7_seal": r7["sha256"], "table": table, "delta": delta, "thr": thr,
           "policies": {n: {g: E.policy_metrics([p for p in P if f(p)], u, delta, thr) for g, f in groups.items()}
                        for n, u in pols.items()}}
    routed = [r for r in log if r["used"] != RT.ORIGINAL]
    res["route_distribution"] = dict(Counter(r["route"] for r in log))
    res["used_distribution"] = dict(Counter(r["used"] for r in log))
    res["decision_distribution"] = dict(Counter(r["decision"] for r in log))
    res["fallbacks"] = dict(Counter(r["fallback"] for r in log if r["fallback"]))
    res["bucket_distribution"] = dict(Counter(r["bucket"] for r in log if r["bucket"]))
    res["bucket_vs_family"] = {f"{b}|{f}": n for (b, f), n in sorted(Counter(
        (r["bucket"], pairs[r["pair_id"]].profile) for r in log if r["bucket"]).items())}
    v1 = [E.verdict(r["d_E01"], delta["E01"]) for r in routed]
    v7 = [E.verdict(r["d_E07"], delta["E07"]) for r in routed]
    res["agreement_on_routed"] = {"n": len(routed), "crosstab": {f"{x}|{y}": n for (x, y), n in sorted(Counter(zip(v1, v7)).items())},
                                  "exact_agreement": round(sum(x == y for x, y in zip(v1, v7)) / max(1, len(routed)), 4)}
    res["failure_cases"] = sorted([r for r in routed if r["verdict"] in GT.HARM], key=lambda r: r["d_E01"])[:40]
    out = repo_path("data/manifests/phase4/t7")
    write_csv_gz(out / f"decision_log_{a.population}.csv.gz", log, list(log[0]))
    res["decision_log_sha256"] = sha256_file(out / f"decision_log_{a.population}.csv.gz")
    res["run"] = run_info()
    write_json(out / f"evaluation_{a.population}.json", res)
    (out / f"evaluation_{a.population}.md").write_text(render(res), encoding="utf-8")
    print(render(res))


def render(r: dict) -> str:
    f = lambda d: f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {d[3]}"  # noqa: E731
    L = [f"# T7 routing evaluation — `{r['population']}` ({r['data_kind']}, {r['split']}; {r['role']})", "",
         f"T6 `{r['t6_seal'][:12]}…` · T7 `{r['t7_seal'][:12]}…` · table {json.dumps(r['table'])}", "",
         f"Decisions {r['decision_distribution']} · routes {r['route_distribution']} · used {r['used_distribution']} · "
         f"fallbacks {r['fallbacks']} · buckets {r['bucket_distribution']}", "",
         "| Policy | enh. rate | ΔE01 | ΔE07 | conf. benefit | harm (any) | conf. harm | rescues/flips E01 | rescues/flips E07 | MILD harm / enhanced |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, p in r["policies"].items():
        a, m = p["all"], p["severity=mild"]
        L.append(f"| {n} | {a['enhancement_rate']:.3f} | {f(a['d_E01'])} | {f(a['d_E07'])} | {a['confirmed_benefit']} | {a['harm_any']} | "
                 f"{a['confirmed_harm']} | {a['rescues_E01']}/{a['flips_E01']} | {a['rescues_E07']}/{a['flips_E07']} | {m['harm_any']} / {m['enhanced']} |")
    L += ["", f"Evaluator agreement on routed pairs: {r['agreement_on_routed']}", "",
          "## T7 route by family (benefit − harm(any), E01/E07 marks)", ""]
    for fam in E.FAMILIES:
        x = r["policies"]["4_T7_route"][f"family={fam}"]
        L.append(f"- {fam}: enhanced {x['enhanced']}/{x['n']}, {x['confirmed_benefit']}−{x['harm_any']}, {x['d_E01'][3]}/{x['d_E07'][3]}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
