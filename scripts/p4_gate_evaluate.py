"""P4-T6 — evaluate the FROZEN Do-Not-Enhance gate ONCE on the untouched SYNTHETIC test population `main`.

    scripts/dock.sh python scripts/p4_gate_evaluate.py [--population main|p4dev]   (p4dev = in-sample reference)

The gate file's SHA-256 seal is verified; nothing is fitted here. Reported (SFace E01 and dlib E07 separately):
  * decision mix (DO_NOT_ENHANCE / ENHANCE_CANDIDATE / UNSAFE_OR_UNCERTAIN) overall, by severity and family
  * gate correctness vs the candidate-agnostic labels: false enhancement (ENHANCE on an UNSAFE pair),
    missed enhancement (DO_NOT_ENHANCE on a WORTHY pair), enhancement rate on MILD inputs
  * identity outcome of 'T6 gate + candidate c' vs always-original and always-c, for every candidate:
    ΔID per evaluator (cluster CI), TAR, flips/rescues, confirmed benefit / harm — and explicitly MILD-input damage
Writes data/manifests/phase4/t6/{evaluation_<pop>.json, evaluation_<pop>.md, decisions_<pop>.csv.gz}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_data.common import repo_path, run_info, sha256_file, write_csv_gz, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="main", choices=list(E.POPS))
    a = ap.parse_args()
    gpath = repo_path("configs/phase4/t6_gate.json")
    gate = json.loads(gpath.read_text(encoding="utf-8"))
    rules = GT.load_rules(gate)                                   # asserts the seal
    pairs = E.load(a.population)
    delta, thr = E.deltas_from_validation(), E.thresholds(a.population)
    P = sorted(pairs.values(), key=lambda p: p.pair_id)
    dec = {p.pair_id: GT.decide(p.feat, rules) for p in P}
    lab = {p.pair_id: GT.pair_label([E.multi_of(p, c, delta) for c in E.CANDIDATES]) for p in P}

    groups = {"all": lambda p: True}
    groups |= {f"severity={s}": (lambda s: lambda p: p.severity == s)(s) for s in E.SEV_ORDER}
    groups |= {f"family={f}": (lambda f: lambda p: p.profile == f)(f) for f in E.FAMILIES}
    res = {"population": a.population, "data_kind": "SYNTHETIC", "split": E.POPS[a.population]["split"],
           "role": "FINAL single evaluation of the frozen gate" if a.population == "main" else "IN-SAMPLE dev reference",
           "gate_file_sha256": sha256_file(gpath), "gate_seal": gate["sha256"], "rules": gate["rules"],
           "delta": delta, "thr": thr, "gate": {}, "policies": {}}
    for g, sel_fn in groups.items():
        sel = [p for p in P if sel_fn(p)]
        mix = Counter(dec[p.pair_id][0] for p in sel)
        cross = Counter((dec[p.pair_id][0], lab[p.pair_id]) for p in sel)
        res["gate"][g] = {"n": len(sel), "decisions": dict(mix),
                          "label_by_decision": {f"{d}|{l}": n for (d, l), n in sorted(cross.items())},
                          "false_enhancement": cross.get((GT.ENH, "UNSAFE"), 0),
                          "missed_enhancement": cross.get((GT.DNE, "WORTHY"), 0),
                          "worthy_total": sum(lab[p.pair_id] == "WORTHY" for p in sel),
                          "unsafe_reasons": dict(Counter(dec[p.pair_id][1].split(":")[0] for p in sel
                                                         if dec[p.pair_id][0] == GT.UNSAFE))}
    pols = {"always_original": {p.pair_id: "B0" for p in P}}
    for c in E.CANDIDATES:
        pols[f"always_{c}"] = {p.pair_id: c for p in P}
        pols[f"T6+{c}"] = {p.pair_id: (c if dec[p.pair_id][0] == GT.ENH else "B0") for p in P}
    for name, used in pols.items():
        res["policies"][name] = {g: E.policy_metrics([p for p in P if f(p)], used, delta, thr)
                                 for g, f in groups.items() if g == "all" or g.startswith("severity=")}
    out = repo_path("data/manifests/phase4/t6")
    write_csv_gz(out / f"decisions_{a.population}.csv.gz",
                 [{"pair_id": p.pair_id, "ref_id": p.ref_id, "cell": p.cell, "decision": dec[p.pair_id][0],
                   "reason": dec[p.pair_id][1], "label": lab[p.pair_id]} for p in P],
                 ["pair_id", "ref_id", "cell", "decision", "reason", "label"])
    res["decisions_sha256"] = sha256_file(out / f"decisions_{a.population}.csv.gz")
    res["run"] = run_info()
    write_json(out / f"evaluation_{a.population}.json", res)
    (out / f"evaluation_{a.population}.md").write_text(render(res), encoding="utf-8")
    print(render(res))


def render(r: dict) -> str:
    f = lambda d: f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {d[3]}"  # noqa: E731
    L = [f"# T6 gate evaluation — `{r['population']}` ({r['data_kind']}, {r['split']}; {r['role']})", "",
         f"Gate seal `{r['gate_seal'][:12]}…` · rules {json.dumps(r['rules'])}", "",
         "## Decisions and gate correctness", "",
         "| Group | n | DNE | ENH | UNSAFE | false enh. (ENH on UNSAFE) | missed (DNE on WORTHY) / worthy |", "|---|---|---|---|---|---|---|"]
    for g, x in r["gate"].items():
        d = x["decisions"]
        L.append(f"| {g} | {x['n']} | {d.get('DO_NOT_ENHANCE', 0)} | {d.get('ENHANCE_CANDIDATE', 0)} | "
                 f"{d.get('UNSAFE_OR_UNCERTAIN', 0)} | {x['false_enhancement']} | {x['missed_enhancement']} / {x['worthy_total']} |")
    L += ["", "## Identity outcome per policy (pooled; SFace E01 and dlib E07 reported separately)", "",
          "| Policy | enh. rate | ΔE01 | ΔE07 | conf. benefit | harm (any) | conf. harm | rescues/flips E01 | rescues/flips E07 | MILD harm (any) / MILD enhanced |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, p in r["policies"].items():
        a, m = p["all"], p["severity=mild"]
        L.append(f"| {n} | {a['enhancement_rate']:.3f} | {f(a['d_E01'])} | {f(a['d_E07'])} | {a['confirmed_benefit']} | "
                 f"{a['harm_any']} | {a['confirmed_harm']} | {a['rescues_E01']}/{a['flips_E01']} | {a['rescues_E07']}/{a['flips_E07']} | "
                 f"{m['harm_any']} / {m['enhanced']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
