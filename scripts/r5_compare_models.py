"""Phase 5 Rapid — head-to-head identity comparison: A4 (SwinIR) vs GFPGAN v1.4 (A6) vs CodeFormer w=0.5/1.0 (A7/A8)
(+ the Phase 3/4 candidates A1–A3, A5 for reference) on the SYNTHETIC test population `main` (896 pairs, 28 refs).

    .venv/Scripts/python.exe scripts/r5_compare_models.py [--configs A6 A7 A8]   (numpy only; reads tracked metrics)

Same rules as P4-T4/T6/T7 (src/fes_bench/p4eval.py): SFace E01 and dlib E07 reported separately, ΔID vs the
untouched input B0 with reference-cluster bootstrap CIs, per-evaluator δ verdicts, TAR at FAR 1e-2, flips/rescues.
Added here: FALSE-ACCEPT risk = pairs where the output's best impostor score crosses the FAR-1e-2 threshold although the
input's did not (identity drifts towards someone else).
Consistency check: the re-run B0 metrics must equal the Phase 3/4 B0 metrics (same inputs, same evaluators).
Policies: always-on · frozen T6 gate + model · T6 + NOISE/BLUR route (T7 table) with the model swapped in.
Writes data/manifests/phase5_rapid/compare/{summary.json, tables.md}.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import routing as RT
from fes_data.common import read_csv_gz, repo_path, run_info, write_json

ROOT5 = "data/manifests/phase5_rapid"
LABEL = {"A1": "RestoreFormer++", "A2": "VQFR v1", "A3": "Real-ESRGAN", "A4": "SwinIR-L (A4)", "A5": "Chain",
         "A6": "GFPGAN v1.4", "A7": "CodeFormer w=0.5", "A8": "CodeFormer w=1.0"}


def add_configs(pairs: dict, configs: list[str]) -> dict:
    """Load Phase 5 metrics; verify the re-run B0 against the Phase 3/4 B0."""
    chk = {}
    for c in ["B0"] + configs:
        for e, path, col, imp in (("E01", f"{ROOT5}/metrics/main/{c}.csv.gz", "id_cos", "imp_cos_max"),
                                  ("E07", f"{ROOT5}/metrics_e07/main/{c}.csv.gz", "id_cos_e07", "imp_cos_max_e07")):
            if not repo_path(path).exists():
                continue
            diffs = []
            for r in read_csv_gz(repo_path(path)):
                p = pairs[r["pair_id"]]
                v = E.fnum(r[col]) if r["status"] == "OK" else float("nan")
                if c == "B0":
                    diffs.append(abs(v - p.cos[("B0", e)]) if not (math.isnan(v) and math.isnan(p.cos[("B0", e)])) else 0.0)
                    continue
                p.cos[(c, e)] = v
                p.imp_max[(c, e)] = E.fnum(r[imp])
                if e == "E01":
                    p.det_out[c] = r["det_found"] == "True"
            if c == "B0":
                chk[f"B0_{e}_max_abs_diff_vs_phase3_4"] = max(diffs) if diffs else None
    return chk


def false_accepts(P, used, thr) -> dict:
    out = {}
    for e in E.EVALUATORS:
        out[f"FA_new_{e}"] = int(sum(p.imp_max[(used[p.pair_id], e)] >= thr[e] > p.imp_max[("B0", e)] for p in P
                                     if used[p.pair_id] != "B0"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["A6", "A7", "A8"])
    a = ap.parse_args()
    pairs = E.load("main")
    chk = add_configs(pairs, a.configs)
    P = sorted(pairs.values(), key=lambda p: p.pair_id)
    have = [c for c in E.CANDIDATES + tuple(a.configs)
            if all((c, e) in P[0].cos for e in E.EVALUATORS)]
    delta, thr = E.deltas_from_validation(), E.thresholds("main")
    g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    rules = GT.load_rules(g6)
    t, _ = RT.load_routing(json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8")), g6)
    dec = {p.pair_id: GT.decide(p.feat, rules)[0] for p in P}
    bkt = {p.pair_id: RT.bucket_of(p.feat, t) if dec[p.pair_id] == GT.ENH else "" for p in P}

    pols = {}
    for c in have:
        pols[f"always_{c}"] = {p.pair_id: c for p in P}
        pols[f"T6+{c}"] = {p.pair_id: (c if dec[p.pair_id] == GT.ENH else "B0") for p in P}
        pols[f"T6+NOISE/BLUR->{c}"] = {p.pair_id: (c if bkt[p.pair_id] in ("NOISE", "BLUR") else "B0") for p in P}
    groups = {"all": lambda p: True} | {f"sev={s}": (lambda s: lambda p: p.severity == s)(s) for s in E.SEV_ORDER} \
        | {f"fam={f}": (lambda f: lambda p: p.profile == f)(f) for f in E.FAMILIES}
    res = {"run": run_info(), "data_kind": "SYNTHETIC (fes-syn-v1 main, test split)", "n_pairs": len(P),
           "n_refs": len({p.ref_id for p in P}), "configs": have, "delta": delta, "thr": thr, "consistency": chk,
           "policies": {}}
    for name, used in pols.items():
        res["policies"][name] = {}
        for g, f in groups.items():
            sel = [p for p in P if f(p)]
            m = E.policy_metrics(sel, used, delta, thr)
            m.update(false_accepts(sel, used, thr))
            res["policies"][name][g] = m
    out = repo_path(f"{ROOT5}/compare")
    write_json(out / "summary.json", res)
    md = render(res)
    (out / "tables.md").write_text(md, encoding="utf-8")
    print(md)


def render(r: dict) -> str:
    f = lambda d: f"{d[0]:+.4f} {d[3]}"  # noqa: E731
    L = [f"# Model comparison on synthetic test `main` ({r['n_pairs']} pairs, {r['n_refs']} identities)", "",
         f"Consistency (re-run B0 vs Phase 3/4): {r['consistency']}", "",
         "▲ = CI above 0 (identity gain), ▼ = CI below 0 (identity loss), ≈ = no detectable difference.", ""]
    for kind in ("always", "T6+", "T6+NOISE/BLUR->"):
        L += [f"## Policy: {kind.rstrip('+->') if kind != 'T6+' else 'T6 gate + model'}", "",
              "| Model | enh. | ΔID SFace | ΔID dlib | conf. benefit | harm (any) | flips SFace/dlib | rescues SFace/dlib | new false-accepts SFace/dlib | MILD harm/enh |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in r["configs"]:
            key = f"always_{c}" if kind == "always" else f"{kind}{c}"
            a, m = r["policies"][key]["all"], r["policies"][key]["sev=mild"]
            L.append(f"| {c} {LABEL[c]} | {a['enhanced']} | {f(a['d_E01'])} | {f(a['d_E07'])} | {a['confirmed_benefit']} | "
                     f"{a['harm_any']} | {a['flips_E01']}/{a['flips_E07']} | {a['rescues_E01']}/{a['rescues_E07']} | "
                     f"{a['FA_new_E01']}/{a['FA_new_E07']} | {m['harm_any']}/{m['enhanced']} |")
        L.append("")
    L += ["## Always-on, by degradation family: confirmed benefit − harm(any), ΔID marks SFace/dlib", "",
          "| Family | " + " | ".join(r["configs"]) + " |", "|---|" + "---|" * len(r["configs"])]
    for fam in E.FAMILIES:
        cells = []
        for c in r["configs"]:
            x = r["policies"][f"always_{c}"][f"fam={fam}"]
            cells.append(f"{x['confirmed_benefit']}−{x['harm_any']} {x['d_E01'][3]}/{x['d_E07'][3]}")
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    L += ["", "## Always-on, by severity", "", "| Severity | " + " | ".join(r["configs"]) + " |",
          "|---|" + "---|" * len(r["configs"])]
    for s in E.SEV_ORDER:
        cells = []
        for c in r["configs"]:
            x = r["policies"][f"always_{c}"][f"sev={s}"]
            cells.append(f"{x['confirmed_benefit']}−{x['harm_any']} {x['d_E01'][3]}/{x['d_E07'][3]} R{x['rescues_E01']}/F{x['flips_E01']}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
